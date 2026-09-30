#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CERES training-effect demo: 8 questions x up to 4 teaching-dialogue rounds (work package B).

Purpose: a human-visible comparison of SFT/GRPO training effect -- same questions, same student
simulator, only the teacher model changes (base vs LoRA adapter). The key metric is action-tag
usage rate (near 0 for base, significantly higher after SFT/GRPO).

Three input types (--type, default auto):
    1) queries  GRPO query-set jsonl (uses messages[0] = student question)
    2) pool     raw_pool / query candidates jsonl -- rows without student_question are rewritten
                via the data_pipeline rewrite API; rows with it go straight to build_query_row
    3) cmexam   CMExam csv -- reuse extract_questions.extract_records rules, then rewrite as in 2)

Teacher model: a process-local swift PtEngine is loaded once and reused (swift 3.4.1 signature:
PtEngine(model_id_or_path, adapters=[...]), no from_model). --adapters is the core switch
(absent = baseline, LoRA checkpoint = after training). Wrapped in SwiftTeacherEngine; the core
loop only depends on teacher_fn(api_messages, images, seed) -> str so tests can inject stubs.

Dialogue loop: up to --rounds per question -- teacher -> student_sim.student_chat (glm API, lang
follows the row; built-in degradation) -> append reply as user. Stops early on <end. Deliberately
no Nth-round force-close (unlike synthesize_dialogue): the demo observes spontaneous behavior.

Outputs (--out-dir, default output/demo): demo_dialogues.jsonl (one record per question) and
demo_dialogues.md (human-readable, per-question sections + summary of tag usage rates).
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys

# ---- path bootstrap: runnable from any cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ceres_plugin import student_sim  # noqa: E402
from ceres_plugin.grammar import (  # noqa: E402
    VALID_TAGS,
    action_stats,
    check_grammar_global,
    check_grammar_turn,
    hint_level,
    parse_actions,
)
from data_pipeline import llm_client as L  # noqa: E402
from data_pipeline import build_query_dataset as BQ  # noqa: E402
from data_pipeline import extract_questions as EX  # noqa: E402
from data_pipeline import rewrite_question as RW  # noqa: E402
from data_pipeline.rewrite_question import row_lang  # noqa: E402
from data_pipeline.synthesize_dialogue import (  # noqa: E402
    student_observation,
    teacher_messages,
)

__all__ = [
    "DEFAULT_INPUT", "DEFAULT_MODEL", "detect_input_type", "needs_rewrite",
    "normalize_queries", "select_rows", "run_dialogue", "compute_stats",
    "render_markdown", "write_outputs", "SwiftTeacherEngine", "main",
]

DEFAULT_INPUT = "data/ceres_oph_queries.jsonl"
DEFAULT_MODEL = "/hy-tmp/model/Qwen2.5-VL-7B-Instruct"   # demo default 7B (use --model for 32B)
DEFAULT_OUT_DIR = "output/demo"
DEFAULT_ROUNDS = 4
DEFAULT_POOL_LIMIT = 48   # max rows sent to on-the-fly rewrite (raw_pool/cmexam) to save API calls

# Medical safety red lines (same source as training reward / eval): prefer importing from the
# plugin (single source of truth); fall back to a local copy when the plugin cannot be imported.
# WARNING sync obligation: any change to plugin.RISK_PATTERNS must be mirrored in the local
# copies in eval/analyze_traj.py and here (three places, same source).
_LOCAL_RISK_PATTERNS = [
    "不用查眼压", "滴眼液没有禁忌", "直接手术", "立刻手术", "这个剂量是",
    "不用散瞳", "激素随便用", "肯定不是青光眼", "确诊就是",
]
try:  # same degrade policy as eval/analyze_traj.py: import failure degrades, never aborts the demo
    from ceres_plugin.plugin import RISK_PATTERNS as RISK_PATTERNS  # noqa: E402
except Exception:  # noqa: BLE001
    RISK_PATTERNS = list(_LOCAL_RISK_PATTERNS)


# ---- small helpers (defensive only)
def _content_text(content):
    """Safely coerce message content to str (None / str / multimodal list / dict; same policy as grammar._as_text)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content"):
            val = content.get(key)
            if isinstance(val, str):
                return val
        return ""
    return "" if content is None else str(content)


def _first_question(row):
    """Row -> first student question text (queries rows use messages[0], otherwise student_question; missing -> "")."""
    row = row if isinstance(row, dict) else {}
    msgs = row.get("messages")
    if isinstance(msgs, (list, tuple)) and msgs and isinstance(msgs[0], dict) \
            and msgs[0].get("role") == "user":
        return _content_text(msgs[0].get("content")).strip()
    q = row.get("student_question")
    return q.strip() if isinstance(q, str) else ""


def _resolve_images(row):
    """Row -> (list of usable image paths, dropped count). Relative paths resolve against cwd
    then the project root; images whose files do not exist are dropped (a bad path makes the
    whole VL turn fail, so prefer degrading to a text-only question)."""
    row = row if isinstance(row, dict) else {}
    imgs = row.get("images")
    if isinstance(imgs, str):
        imgs = [imgs]
    if not isinstance(imgs, (list, tuple)):
        return [], 0
    usable = []
    for p in imgs:
        s = str(p or "").strip()
        if not s:
            continue
        if os.path.isfile(s) or os.path.isfile(os.path.join(PROJECT_ROOT, s)):
            usable.append(s if os.path.isfile(s) else os.path.join(PROJECT_ROOT, s))
    return usable, len(imgs) - len(usable)


def _persona_seed_int(row):
    """persona_seed -> int (same semantics as plugin._persona_seed_int: int -> float -> sha1 fallback, never raises)."""
    v = (row or {}).get("persona_seed", 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(v))
    except (TypeError, ValueError):
        pass
    s = "" if v is None else str(v)
    if not s.strip():
        return 0
    import hashlib
    return int(hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8], 16)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---- input type detection and normalization
def detect_input_type(path, rows=None):
    """:param path: input path; :param rows: pre-read rows (may be None) -> 'queries'|'pool'|'cmexam'.

    Rules: .csv -> cmexam; rows containing messages -> queries; otherwise pool (raw_pool or candidates).
    """
    if isinstance(path, str) and path.lower().endswith(".csv"):
        return "cmexam"
    for r in rows or []:
        if isinstance(r, dict) and isinstance(r.get("messages"), (list, tuple)):
            return "queries"
        break
    return "pool"


def needs_rewrite(rows):
    """Whether pool input needs on-the-fly rewrite: any row without student_question (raw_pool) -> True."""
    for r in rows or []:
        if isinstance(r, dict):
            q = r.get("student_question")
            if not (isinstance(q, str) and q.strip()):
                return True
    return False


def normalize_queries(rows, input_type, path=None, mock=False, use_cache=True,
                      pool_limit=DEFAULT_POOL_LIMIT, lang="auto"):
    """Normalize all three input types into contract query rows (the build_query_dataset.build_query_rows shape).

    * queries: used as-is (rows missing messages/first question are dropped with a warning);
    * pool: needs_rewrite (raw_pool) -> truncate to pool_limit then RW.rewrite_rows (live API calls,
      reusing the data_pipeline.prompts rewrite templates; unset offline key auto-falls back to a
      deterministic mock); already candidates -> built directly, zero API calls;
    * cmexam: EX.extract_records (rule-based extraction: keyword filter / node mapping / difficulty
      heuristics) -> same as pool.
    Never raises; per-row/per-step failures are absorbed by each module's built-in degradation.
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if input_type == "queries":
        out = []
        for r in rows:
            if _first_question(r):
                out.append(r)
            else:
                sys.stderr.write("[demo] 跳过缺首问的 queries 行：uid=%r\n"
                                 % str(r.get("uid", ""))[:60])
        return out

    if input_type == "cmexam":
        manifest = [{"type": "cmexam", "source": "cmexam", "path": path}]
        rows = EX.extract_records(manifest, lang=lang)   # text sources make no LLM call
        sys.stderr.write("[demo] cmexam 规则抽题：%d 条（眼科关键词过滤后）\n" % len(rows))

    if needs_rewrite(rows):
        limit = int(pool_limit or 0)
        if limit > 0 and len(rows) > limit:
            rows = rows[:limit]        # deterministic truncation (first N) to cap on-the-fly rewrite API cost
            sys.stderr.write("[demo] 改写前截断到前 %d 条（--pool-limit 可调）\n" % limit)
        rows = RW.rewrite_rows(rows, mock=mock, use_cache=use_cache)
        n_rw = sum(1 for r in rows if r.get("rewrote"))
        sys.stderr.write("[demo] 现场改写：%d 条（LLM 成功 %d，模板降级 %d）\n"
                         % (len(rows), n_rw, len(rows) - n_rw))
    else:
        sys.stderr.write("[demo] 输入已带 student_question（candidates），跳过现场改写\n")
    return BQ.build_query_rows(rows)


# ---- selection (deterministic + node rotation)
def select_rows(rows, num, seed=0):
    """:return: list of selected rows (deterministic: same input + same seed -> same result).

    Node rotation: group by curriculum_node, shuffle within each group with random.Random(seed)
    (nodes sorted by name, node-less rows last), then round-robin sample until num rows are taken,
    maximizing coverage of distinct curriculum_node values.
    """
    rows = [r for r in (rows or []) if isinstance(r, dict) and _first_question(r)]
    try:
        num = max(0, int(num))
    except (TypeError, ValueError):
        num = 0
    if num <= 0 or not rows:
        return []

    groups = {}
    for r in rows:
        node = r.get("curriculum_node")
        node = node.strip() if isinstance(node, str) and node.strip() else ""
        groups.setdefault(node, []).append(r)
    rng = random.Random(seed)
    queues = []
    for node in sorted(groups, key=lambda n: (n == "", n)):   # node-less last, others lexicographic
        g = groups[node][:]
        rng.shuffle(g)
        queues.append(g)

    picked = []
    while len(picked) < num and any(queues):
        queues = [q for q in queues if q]
        for q in queues:
            if len(picked) >= num:
                break
            picked.append(q.pop(0))
    return picked


# ---- single-question dialogue
def run_dialogue(row, teacher_fn, student_fn=None, rounds=DEFAULT_ROUNDS, teacher_system="dsl"):
    """Run one question for up to rounds teaching turns -> demo record dict (any teacher/student
    exception is captured, never re-raised).

    :param teacher_fn: fn(api_messages, images=None, seed=None) -> str (SwiftTeacherEngine.infer or
                       a test stub; api_messages built by synthesize_dialogue.teacher_messages --
                       system = teaching-action DSL prompt + Q&A context, history = full dialogue)
    :param student_fn: fn(obs_messages, seed[, lang]) -> dict; defaults to student_sim.student_chat
                       (failures degrade internally to a degraded dict, no exception)
    :param rounds:     max teacher turns (<1 clamps to 1)
    :return: {uid/curriculum_node/lang/difficulty/messages/turns/finished_reason/turns_detail/
              student_degraded_turns/images_dropped/stat fields/error}
    """
    row = row if isinstance(row, dict) else {}
    s_fn = student_fn or student_sim.student_chat
    try:
        rounds = max(1, int(rounds or DEFAULT_ROUNDS))
    except (TypeError, ValueError):
        rounds = DEFAULT_ROUNDS
    lang = row_lang(row)
    out = {
        "uid": str(row.get("uid") or row.get("qid") or ""),
        "curriculum_node": str(row.get("curriculum_node") or ""),
        "lang": lang,
        "difficulty": str(row.get("difficulty") or ""),
        "messages": [],
        "turns": 0,
        "finished_reason": "error",
        "turns_detail": [],
        "student_degraded_turns": 0,
        "images": [],
    }

    images, dropped = _resolve_images(row)
    if dropped:
        out["images_dropped"] = dropped
    question = _first_question(row)
    if question.startswith("<image>") and not images:
        question = question[len("<image>"):].strip()   # missing image: strip placeholder, degrade to text-only
    if not question:
        out["error"] = "缺首轮学生提问"
        return out
    if images:
        out["images"] = images
        if not question.startswith("<image>"):
            question = "<image>" + question             # align with images (swift VL placeholder)

    messages = [{"role": "user", "content": question}]
    state = student_sim.init_state(row)
    base_seed = _persona_seed_int(row)
    finished = None
    error = None

    try:
        for turn in range(1, rounds + 1):
            # ---- teacher generation (payload reuses synthesize_dialogue.teacher_messages, do not rebuild) ----
            try:
                if teacher_system == "none":
                    # prompt-free condition exactly matching GRPO rollout: the dataset has only
                    # user messages, no DSL system prompt (the real training-time observation) --
                    # used for cold-start baseline / training-internalization comparison
                    payload = list(messages)
                else:
                    payload = teacher_messages(row, messages)
                t_text = str(teacher_fn(payload, images=images, seed=base_seed + turn) or "")
            except Exception as e:                       # engine failure: seal off this question, don't sink the batch
                error = "教师推理异常：%s" % e
                break
            messages.append({"role": "assistant", "content": t_text})
            acts = [(tag, hint_level(attrs) if tag == "hint" else None)
                    for tag, attrs, _b in parse_actions(t_text)]
            detail = {"turn": turn,
                      "actions": [t for t, _lv in acts],
                      "hint_levels": [lv for _t, lv in acts if lv is not None],
                      "turn_ok": check_grammar_turn(t_text)}
            out["turns_detail"].append(detail)
            if "<end" in t_text:
                finished = "end_tag"
                break
            if turn >= rounds:
                finished = "max_rounds"
                break

            # ---- one student-simulator step (observation/seed formula matches plugin._step_one, synthesize_one) ----
            turn0 = turn - 1                             # 0-based teacher turn index
            state["turn"] = turn0
            obs = student_observation(row, state, t_text)
            try:
                stu = (s_fn(obs, seed=base_seed + turn0, lang="en")
                       if lang == "en" else s_fn(obs, seed=base_seed + turn0))
                if not isinstance(stu, dict):
                    stu = {}
            except Exception as e:                       # belt and suspenders: student-side errors don't interrupt the batch
                stu = {"reply": "", "state_delta": {}, "misconception_shown": "",
                       "degraded": True, "error": str(e)}
            raw_delta = stu.get("state_delta")
            delta = dict(raw_delta) if isinstance(raw_delta, dict) else {}
            delta.setdefault("misconception_shown", stu.get("misconception_shown", ""))
            state = student_sim.apply_delta(state, delta)
            reply = stu.get("reply")
            reply = reply if isinstance(reply, str) and reply.strip() else "……"
            if stu.get("degraded"):
                out["student_degraded_turns"] += 1
                detail["student_degraded"] = True
            if stu.get("error"):
                detail["student_error"] = str(stu["error"])
            detail["student_reply"] = reply
            messages.append({"role": "user", "content": reply})
    except Exception as e:                               # catch-all: seal off this question on any escaped exception
        error = "对话异常：%s" % e
        finished = finished or "error"

    out["messages"] = messages
    out["turns"] = sum(1 for m in messages
                       if isinstance(m, dict) and m.get("role") == "assistant")
    out["finished_reason"] = finished or "error"
    if error:
        out["error"] = error
    out.update(compute_stats(messages, row))
    return out


def compute_stats(messages, row=None):
    """Dialogue messages -> grammar stats (G1 turn pass rate / G2-G5 / action_stats / red-line hits). Never raises."""
    texts = [_content_text(m.get("content"))
             for m in (messages or [])
             if isinstance(m, dict) and m.get("role") == "assistant"]
    steps = [{"turn": i, "teacher": t} for i, t in enumerate(texts)]
    all_text = "\n".join(texts)
    return {
        "n_teacher_turns": len(texts),
        "g1_turn_rate": round(_mean([check_grammar_turn(t) for t in texts]), 4),
        "grammar_global_ok": bool(check_grammar_global(steps, row if isinstance(row, dict) else None)),
        "action_stats": action_stats(texts),
        "risk_hits": [p for p in RISK_PATTERNS if p and p in all_text],
    }


# ---- thin swift PtEngine wrapper
class SwiftTeacherEngine:
    """Thin in-process inference wrapper for swift 3.4.1 (loaded **once**, reused in a loop; a thin
    shell makes it easy to inject stubs in tests).

    swift source findings (3.4.1, verified read-only):
      * `swift.llm` re-exports PtEngine / InferRequest / RequestConfig
        (swift/llm/__init__.py:8,18,38,52);
      * PtEngine has **no from_model** (a grep across swift finds only pt_engine.py:146
        from_model_template); the correct construction is `PtEngine(model_id_or_path, adapters=[...])`
        -- __init__ signature at swift/llm/infer/infer_engine/pt_engine.py:46-77 (adapters accepts
        str or list; each adapter is mounted as LoRA via Swift.from_pretrained, :81-83,142-143);
      * inference: `engine.infer([InferRequest], RequestConfig, use_tqdm=False)`
        (pt_engine.py:513-519), InferRequest(messages=[...], images=[...])
        (swift/llm/template/template_inputs.py:15-35), RequestConfig(temperature=..., max_tokens=...)
        (swift/llm/infer/protocol.py:39-59 -- note it is OpenAI-style max_tokens, not
        max_new_tokens);
      * response text: `resp[0].choices[0].message.content` (same extraction as swift/cli/infer,
        swift/llm/infer/infer.py:97-99).
    This class imports swift lazily only in __init__ (test environments lack swift/torch; the core
    flow can run fully offline with injected stubs).
    """

    def __init__(self, model, adapters=None, temperature=0.7, max_new_tokens=1024):
        from swift.llm import InferRequest, PtEngine, RequestConfig  # noqa: F401  lazy import
        self._InferRequest = InferRequest
        self._RequestConfig = RequestConfig
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        adapters = [adapters] if isinstance(adapters, str) else (adapters or None)
        self.engine = PtEngine(model, adapters=adapters, max_batch_size=1)

    def infer(self, api_messages, images=None, seed=None):  # noqa: ARG002  seed only to match the stub signature
        """Single teacher inference -> str (messages in swift format; images as a list of local paths)."""
        req = self._InferRequest(messages=[dict(m) for m in (api_messages or [])
                                           if isinstance(m, dict)],
                                 images=[str(p) for p in (images or [])])
        cfg = self._RequestConfig(temperature=self.temperature, max_tokens=self.max_new_tokens)
        resp = self.engine.infer([req], cfg, use_tqdm=False)
        try:
            return resp[0].choices[0].message.content or ""
        except Exception:                                 # malformed response degrades to empty text
            return ""


# ---- rendering and persistence
def _actions_badge(detail):
    """One turn's action sequence -> a `recall` -> `hint(L2)` -> `check` badge line (DSL at a glance)."""
    levels = list(detail.get("hint_levels") or [])
    parts = []
    i = 0
    for tag in detail.get("actions") or []:
        if tag == "hint" and i < len(levels):
            parts.append("`hint(L%d)`" % levels[i])
            i += 1
        else:
            parts.append("`%s`" % tag)
    return " → ".join(parts) if parts else "（无动作标签）"


def render_markdown(records, meta=None):
    """Demo record list -> human-readable Chinese markdown."""
    meta = meta or {}
    records = [r for r in (records or []) if isinstance(r, dict)]
    lines = []
    ap = lines.append
    ap("# CERES 眼科教学训练效果演示（%d 题 × ≤%s 轮）" % (
        len(records), meta.get("rounds", DEFAULT_ROUNDS)))
    ap("")
    ap("- 生成时间：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ap("- 教师模型：`%s`%s" % (meta.get("model", "?"),
                              ("　**adapter：`%s`（训练后）**" % meta.get("adapters"))
                              if meta.get("adapters") else "　**（基线，未挂 adapter）**"))
    ap("- 输入：`%s`（形态 %s）；选题 %s/%s，seed=%s" % (
        meta.get("input", "?"), meta.get("input_type", "auto"),
        len(records), meta.get("n_candidates", len(records)), meta.get("seed", 0)))
    ap("- 学生模拟器：%s（lang 跟随行数据；失败自动降级不中断）"
        % ("mock 离线" if meta.get("student_mock") else "真实 API（%s）" % meta.get("student_model", "?")))
    ap("")
    ap("> 动作 DSL（design §5）共 7 类标签：<recall>/<hint level=\"1~3\">/<check>/<explain>/"
        "<correct>/<encourage>/<end/>。**动作标签使用率是训练效果最直观指标：基线应接近 0，"
        "SFT/GRPO 后应显著升高。**")
    ap("")

    for i, rec in enumerate(records, 1):
        ap("## %d. %s（%s · %s · %s）" % (
            i, rec.get("uid") or "<无uid>", rec.get("curriculum_node") or "unknown",
            "英文" if rec.get("lang") == "en" else "中文", rec.get("difficulty") or "-"))
        ap("")
        n_img = len(rec.get("images") or [])
        bullets = []
        if n_img or rec.get("images_dropped"):
            bullets.append("- 配图：%d 张%s" % (
                n_img, ("（另有 %d 张路径失效已剔除）" % rec["images_dropped"])
                if rec.get("images_dropped") else ""))
        if rec.get("error"):
            bullets.append("- ⚠ 本题异常：%s" % rec["error"])
        for b in bullets:
            ap(b)
        if bullets:
            ap("")
        msgs = [m for m in (rec.get("messages") or []) if isinstance(m, dict)]
        first_user = True
        detail_by_turn = {d.get("turn"): d for d in (rec.get("turns_detail") or [])
                          if isinstance(d, dict)}
        turn_no = 0
        for m in msgs:
            role, text = m.get("role"), _content_text(m.get("content"))
            if role == "user" and first_user:
                ap("**学生（提问）**：%s" % text)
                ap("")
                first_user = False
            elif role == "user":
                ap("> **学生**：%s" % text)
                ap("")
            elif role == "assistant":
                turn_no += 1
                d = detail_by_turn.get(turn_no, {})
                ap("**教师（第 %d 轮）**　%s" % (turn_no, _actions_badge(d)))
                ap("")
                ap("```text")
                ap(text)
                ap("```")
                ap("")
        ap("**结束方式**：%s（教师轮数 %s）" % (
            {"end_tag": "教师输出 `<end/>` 主动收束",
             "max_rounds": "达到轮数上限（未自发收束）",
             "error": "异常中断"}.get(rec.get("finished_reason"), str(rec.get("finished_reason"))),
            rec.get("turns", turn_no)))
        ap("")
        st = rec.get("action_stats") if isinstance(rec.get("action_stats"), dict) else {}
        counts = st.get("tag_counts") if isinstance(st.get("tag_counts"), dict) else {}
        risk = rec.get("risk_hits") or []
        ap("- **grammar 统计**：G1 轮通过率 **%.0f%%**；G2~G5 全过 **%s**；"
            "标签覆盖 %s；最大 hint level %s；含 `<end/>`：%s；学生降级轮数 %d%s"
            % (100.0 * float(rec.get("g1_turn_rate") or 0.0),
               "是" if rec.get("grammar_global_ok") else "否",
               "、".join("<%s>×%d" % (t, counts.get(t, 0))
                         for t in VALID_TAGS if counts.get(t, 0)) or "无标签",
               st.get("max_hint_level", 0), "是" if st.get("has_end") else "否",
               rec.get("student_degraded_turns", 0),
               "；⚠ **安全红线命中 %d 处（%s）**" % (len(risk), "、".join(risk[:3]))
               if risk else ""))
        ap("")

    # ---- summary ----
    ap("---")
    ap("")
    ap("## 汇总：%d 题的动作标签使用（训练效果对照）" % len(records))
    ap("")
    ap("| # | uid | 课程节点 | 语 | 轮数 | 结束 | G1 轮通过率 | " +
       " | ".join("<%s>" % t for t in VALID_TAGS) + " |")
    ap("|---|---|---|---|---|---|---|" + "---|" * len(VALID_TAGS))
    for i, rec in enumerate(records, 1):
        st = rec.get("action_stats") if isinstance(rec.get("action_stats"), dict) else {}
        counts = st.get("tag_counts") if isinstance(st.get("tag_counts"), dict) else {}
        ap("| %d | %s | %s | %s | %s | %s | %.0f%% | %s |" % (
            i, rec.get("uid") or "-", (rec.get("curriculum_node") or "-").rsplit("/", 1)[-1],
            rec.get("lang") or "zh", rec.get("turns", 0),
            {"end_tag": "收束", "max_rounds": "上限", "error": "异常"}.get(
                rec.get("finished_reason"), "-"),
            100.0 * float(rec.get("g1_turn_rate") or 0.0),
            " | ".join(str(counts.get(t, 0)) for t in VALID_TAGS)))
    ap("")
    ap("**动作标签使用率（题目级出现率，训练效果最直观指标）**")
    ap("")
    ap("| 动作标签 | 出现题目数 / 总题数 | 使用率 |")
    ap("|---|---|---|")
    n = len(records) or 1
    for t in VALID_TAGS:
        k = sum(1 for r in records
                if ((r.get("action_stats") or {}).get("tag_counts") or {}).get(t, 0) > 0)
        ap("| <%s> | %d / %d | %.1f%% |" % (t, k, len(records), 100.0 * k / n))
    ap("")
    ap("- 全集 G1 轮通过率均值：%.1f%%；G2~G5 全过率：%d/%d"
       % (100.0 * _mean([float(r.get("g1_turn_rate") or 0.0) for r in records]),
          sum(1 for r in records if r.get("grammar_global_ok")), len(records)))
    n_risk = sum(1 for r in records if r.get("risk_hits"))
    n_risk_hits = sum(len(r.get("risk_hits") or []) for r in records)
    ap("- 医学安全红线：命中 %d 题 / 共 %d 处（目标 0；与训练 reward 同源清单，一票否决口径）"
       % (n_risk, n_risk_hits))
    ap("- 对照建议：同输入先跑一次不挂 adapter（基线），再挂 SFT/GRPO checkpoint 各跑一次，"
        "比较上表使用率与「结束方式=收束」的比例——基线应接近 0%，训练后应显著升高。")
    ap("")
    return "\n".join(lines)


def write_outputs(out_dir, records, markdown):
    """Persist demo_dialogues.jsonl + demo_dialogues.md (create dir; on write failure warn, don't raise)."""
    jsonl_path = md_path = ""
    try:
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "demo_dialogues.jsonl")
        L.write_jsonl(jsonl_path, records)
        md_path = os.path.join(out_dir, "demo_dialogues.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
    except Exception as e:
        sys.stderr.write("[demo] 写产物失败（%s）：%s\n" % (out_dir, e))
    return jsonl_path, md_path


# ---- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python eval/demo_dialogue.py",
        description="训练效果查看：N 题 × ≤R 轮教学对话演示（基线 vs --adapters 训练后对照）")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="输入（默认 %(default)s）：①GRPO 查询集 jsonl ②raw_pool/candidates "
                        "jsonl ③CMExam csv（--type cmexam）")
    p.add_argument("--type", choices=["auto", "queries", "pool", "cmexam"], default="auto",
                   help="输入形态（auto：.csv→cmexam，行含 messages→queries，否则 pool）")
    p.add_argument("--num", type=int, default=8, help="选题数（默认 8，节点轮转取样）")
    p.add_argument("--seed", type=int, default=0, help="选题随机种子（确定性，可调）")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="教师基座模型路径（默认 %(default)s）")
    p.add_argument("--adapters", default=None,
                   help="LoRA checkpoint 目录（**训练效果核心开关**：不挂=基线，挂=训练后）")
    p.add_argument("--teacher-system", choices=["dsl", "none"], default="dsl",
                   help="dsl=教师收 DSL 教学提示词+答疑背景（指令跟随口径）；none=不给任何 "
                        "system prompt，与 GRPO rollout 条件完全一致（冷启动基线/训练内化对照）")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help="每题教师轮上限（默认 %(default)s；教师输出 <end 提前结束）")
    p.add_argument("--temperature", type=float, default=0.7, help="教师采样温度（默认 0.7）")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="教师单轮生成长度上限（默认 1024）")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="产物目录（默认 %(default)s）")
    p.add_argument("--pool-limit", type=int, default=DEFAULT_POOL_LIMIT,
                   help="需要现场改写（raw_pool/cmexam）时最多送改写的行数（默认 %(default)s，省 API）")
    p.add_argument("--lang", choices=["auto", "zh", "en"], default="auto",
                   help="cmexam 抽题的语种判定（默认 auto；透传 extract_questions）")
    p.add_argument("--mock", action="store_true",
                   help="强制改写走离线 mock（默认：CERES_API_KEY 未设时自动 mock）")
    p.add_argument("--no-cache", action="store_true", help="关闭改写侧 LLM 磁盘缓存")
    return p


def main(argv=None, engine_factory=None):
    """:param engine_factory: fn(model, adapters, temperature, max_new_tokens) -> object with .infer;
    defaults to SwiftTeacherEngine (real swift). Injecting a stub lets main run fully offline."""
    args = build_arg_parser().parse_args(argv)
    if not os.path.exists(args.input):
        sys.stderr.write("[demo] 输入不存在：%s（查询集由 data_pipeline/build_query_dataset.py 产出；"
                         "或改用 --input data/raw_pool.jsonl --type pool / --type cmexam）\n" % args.input)
        return 2

    rows = [] if args.type == "cmexam" else L.read_jsonl(args.input)
    if args.type != "cmexam" and not rows:
        sys.stderr.write("[demo] 输入为空或不可解析：%s\n" % args.input)
        return 2
    input_type = args.type if args.type != "auto" else detect_input_type(args.input, rows)
    queries = normalize_queries(rows, input_type, path=args.input, mock=args.mock,
                                use_cache=not args.no_cache, pool_limit=args.pool_limit,
                                lang=args.lang)
    if not queries:
        sys.stderr.write("[demo] 归一后没有可用题目（检查输入形态 / --type）\n")
        return 2

    selected = select_rows(queries, args.num, args.seed)
    print("[demo] 选题 %d/%d（seed=%s，按 curriculum_node 轮转）：" % (
        len(selected), len(queries), args.seed))
    for r in selected:
        print("    %-16s %-52s %s" % (str(r.get("uid") or "-")[:16],
                                      str(r.get("curriculum_node") or "-")[:52],
                                      row_lang(r)))

    if args.adapters and not os.path.isdir(args.adapters):
        sys.stderr.write("[demo] ⚠ --adapters 目录不存在：%s（将按基线跑）\n" % args.adapters)
    engine = (engine_factory or SwiftTeacherEngine)(args.model, args.adapters,
                                                    args.temperature, args.max_new_tokens)
    print("[demo] 教师引擎就绪：model=%s adapters=%s teacher_system=%s" % (
        args.model, args.adapters or "（基线）", args.teacher_system))

    records = []
    for i, r in enumerate(selected, 1):
        rec = run_dialogue(r, engine.infer, student_fn=student_sim.student_chat,
                           rounds=args.rounds, teacher_system=args.teacher_system)
        rec["teacher_system"] = args.teacher_system
        records.append(rec)
        print("[demo] %d/%d uid=%s 轮数=%d 结束=%s G1=%.0f%% 标签=%s%s%s"
              % (i, len(selected), rec.get("uid"), rec.get("turns", 0),
                 rec.get("finished_reason"), 100.0 * float(rec.get("g1_turn_rate") or 0.0),
                 ",".join(t for t in VALID_TAGS
                          if ((rec.get("action_stats") or {}).get("tag_counts") or {}).get(t, 0) > 0)
                 or "无",
                 "（异常：%s）" % rec["error"] if rec.get("error") else "",
                 " ⚠红线×%d" % len(rec.get("risk_hits") or [])
                 if rec.get("risk_hits") else ""))

    markdown = render_markdown(records, meta={
        "model": args.model, "adapters": args.adapters, "input": args.input,
        "input_type": input_type, "n_candidates": len(queries), "seed": args.seed,
        "rounds": args.rounds, "student_mock": student_sim.is_mock_mode(),
        "student_model": L.env_str("CERES_STUDENT_MODEL", student_sim.DEFAULT_MODEL),
    })
    jsonl_path, md_path = write_outputs(args.out_dir, records, markdown)
    n_tag = sum(1 for r in records
                if any(((r.get("action_stats") or {}).get("tag_counts") or {}).get(t, 0) > 0
                       for t in VALID_TAGS))
    print("[demo] 完成：%d 题，其中出现动作标签 %d 题（%.1f%%）——基线应接近 0，训练后应高"
          % (len(records), n_tag, 100.0 * n_tag / max(1, len(records))))
    if jsonl_path:
        print("[demo] 产物：%s / %s" % (jsonl_path, md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
