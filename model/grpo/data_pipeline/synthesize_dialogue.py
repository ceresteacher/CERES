# -*- coding: utf-8 -*-
"""Step 3: candidates -> data/sft_raw.jsonl (design §6.3 Step3).

Per row, synthesize a <=5-turn demo dialogue: messages[0] = student_question (identical to the GRPO
first turn); teacher via llm_client.chat (TEACHER_SYSTEM_PROMPT + briefing + history); student reuses
ceres_plugin.student_sim.student_chat with the same persona_seed (observation and seed formula match
plugin.py:_step_one). <end/> or 5 turns ends it; unwrapped turn 5 -> FORCE_CLOSE_INSTRUCTION retry
(replaces turn 5, instruction only in payload); still failed -> "unresolved": true (dropped by
quality_filter). Output: {"messages", "images", "uid"} + audit columns.
"""
import argparse
import concurrent.futures
import json
import os
import sys

# Ensure project root is on sys.path (same as prepare_datasets)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ceres_plugin import student_sim  # noqa: E402
from data_pipeline import llm_client as L  # noqa: E402
from data_pipeline.prompts import (  # noqa: E402
    FORCE_CLOSE_INSTRUCTION,
    TEACHER_SYSTEM_PROMPT,
    build_teacher_context,
    force_close_instruction,
    teacher_system_prompt,
)

__all__ = ["student_observation", "teacher_messages", "synthesize_one", "synthesize_rows", "main"]


# ---------------------------------------------------------------- Observation / payload construction
def _s_col(v):
    """Dataset column safe-str (mirrors ceres_plugin/plugin.py:_s: None->empty, str as-is, else str())."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _norm_lang(lang):
    """Normalize lang: only 'en' stays 'en'; everything else (None/empty/'zh') -> 'zh' (default)."""
    return "en" if isinstance(lang, str) and lang.strip().lower() == "en" else "zh"


def _row_lang(row):
    """Dataset row -> 'zh'|'en' (row lang column; missing/invalid -> zh — same as plugin._lang_of)."""
    row = row if isinstance(row, dict) else {}
    return _norm_lang(row.get("lang"))


def student_observation(row, state, teacher_text, lang=None):
    """Student-side observation — **field-for-field identical to the training env ceres_plugin/
    plugin.py:_student_messages (:252-266)** (design §7.3 verbatim: single user turn, text only).

    Locked content order = course context/findings -> learner profile -> current learner state
    (state["turn"] written 0-based before the call) -> teacher turn -> response instruction; state
    serialized with json.dumps(..., ensure_ascii=False, default=str), falls back to str(state).
    lang=None -> row lang column (English rows get English labels + instruction). ⚠ changing this or
    plugin._student_messages requires syncing the other side.
    """
    dd = row if isinstance(row, dict) else {}
    try:
        state_json = json.dumps(state, ensure_ascii=False, default=str)
    except Exception:
        state_json = str(state)
    if _norm_lang(dd.get("lang") if lang is None else lang) == "en":
        content = (
            "[Course context / findings] %s\n"
            "[Learner profile] %s\n"
            "[Current learner state] %s\n"
            "[Teacher's action this turn]\n%s\n"
            "Please respond as an ophthalmology resident-in-training and output JSON."
        ) % (_s_col(dd.get("courseware_context", "")), _s_col(dd.get("learner_profile", "")),
             state_json, str(teacher_text or ""))
        return [{"role": "user", "content": content}]
    content = (
        "【课程上下文/检查所见】%s\n"
        "【学习者画像】%s\n"
        "【当前学习者状态】%s\n"
        "【教师本轮动作】\n%s\n"
        "请以眼科规培生身份回应，并输出 JSON。"
    ) % (_s_col(dd.get("courseware_context", "")), _s_col(dd.get("learner_profile", "")),
         state_json, str(teacher_text or ""))
    return [{"role": "user", "content": content}]


def teacher_messages(row, dialogue, force_close=False):
    """Teacher API payload: system = TEACHER_SYSTEM_PROMPT(_EN) (+ force-close instruction in row lang)
    + case briefing (build_teacher_context, same lang).

    The briefing only anchors the conclusion, never into output messages (SFT first user turn matches
    the student question). A7: force_close=True drops the trailing assistant turn (list(dialogue)[:-1])
    — force close replaces the max_turns reply, the old turn shouldn't enter the payload; some endpoints
    also continue-writing unstable after an assistant-ending prompt, ending on the last user message is safer.
    Regular turns pass the full history through. Lang per row (English source -> English trajectory).
    """
    lang = _row_lang(row)
    system = teacher_system_prompt(lang)
    if force_close:
        system = system + "\n" + force_close_instruction(lang)
    ctx = build_teacher_context(row.get("courseware_context"), row.get("gt_label"),
                                row.get("gt_explanation"), row.get("misconception_seed"),
                                row.get("difficulty"), lang=lang)
    history = list(dialogue)
    if force_close and history and isinstance(history[-1], dict) \
            and history[-1].get("role") == "assistant":
        history = history[:-1]   # trailing assistant is the replaced turn (defensive: only when assistant)
    return [{"role": "system", "content": system + "\n\n" + ctx}] + history


# ---------------------------------------------------------------- Single-item synthesis
def synthesize_one(row, max_turns=5, teacher_fn=None, student_fn=None, seed=0):
    """Synthesize one dialogue -> output row dict (internal exceptions caught as error rows, never raised).

    :param teacher_fn: fn(api_messages, seed) -> str; default llm_client.chat (injectable stub)
    :param student_fn: fn(stu_messages, seed[, lang]) -> dict; default student_sim.student_chat
                       (English rows pass lang='en' kwarg — only when row lang='en', for compat with
                        (messages, seed)-shaped stubs)
    :param max_turns:  teacher turn cap (default 5; env CERES_MAX_TURNS read by CLI)

    lang='en' rows -> teacher prompt / force-close / student observation and student_chat all English
    (trajectory lang follows source); output row carries the lang column.
    """
    row = row if isinstance(row, dict) else {}
    lang = _row_lang(row)
    out = {
        "uid": str(row.get("qid") or row.get("uid") or ""),
        "messages": [],
        "turns": 0,
        "unresolved": False,
        "forced_close": False,
        "curriculum_node": row.get("curriculum_node") or "",
        "difficulty": row.get("difficulty") or "",
        "gt_label": str(row.get("gt_label") or ""),
        "gt_explanation": str(row.get("gt_explanation") or ""),
        "source": row.get("source") or "",
        "persona_seed": int(row.get("persona_seed") or 0),
        "rewrote": bool(row.get("rewrote")),
        "lang": lang,
    }
    images = row.get("images")
    if isinstance(images, str):
        images = [images]
    if isinstance(images, (list, tuple)) and images:
        out["images"] = [str(p) for p in images if p]
    elif row.get("image"):                       # candidate rows with only a single image value pass through
        out["images"] = [str(row["image"])]

    question = row.get("student_question")
    if not isinstance(question, str) or not question.strip():
        out["error"] = "student_question 缺失"
        return out
    question = question.strip()
    if (row.get("image") or out.get("images")) and not question.startswith("<image>"):
        question = "<image>" + question          # align with images column (defensive placeholder)

    try:
        max_turns = max(1, int(max_turns or 5))
    except (TypeError, ValueError):
        max_turns = 5
    t_fn = teacher_fn or (lambda msgs, s: L.chat(msgs, seed=s))
    s_fn = student_fn or student_sim.student_chat
    try:
        base_seed = int(row.get("persona_seed") or 0)
    except (TypeError, ValueError):
        base_seed = 0

    messages = [{"role": "user", "content": question}]
    state = student_sim.init_state(row)   # same args as plugin._step_one's init_state(_input)
    n_degraded = 0

    try:
        for turn in range(1, max_turns + 1):
            t_text = str(t_fn(teacher_messages(row, messages), seed + base_seed + turn) or "")
            messages.append({"role": "assistant", "content": t_text})
            if "<end" in t_text:                 # natural wrap-up (incl. <end/> or malformed; grammar judges)
                break
            if turn == max_turns:
                # turn max_turns not wrapped up -> force close (replace this reply; instruction only in payload)
                f_text = str(t_fn(teacher_messages(row, messages, force_close=True),
                                  seed + base_seed + turn + 1) or "")
                if "<end" in f_text:
                    messages[-1] = {"role": "assistant", "content": f_text}
                    out["forced_close"] = True
                else:
                    out["unresolved"] = True      # force close still failed: keep original turn, mark for filter
                break
            # ---- student side: exactly same args as ceres_plugin/plugin.py:_step_one (:392-395) ----
            # 0-based turn0 = this turn's assistant count - 1; state["turn"] written before the student
            # call; seed = persona_seed + turn0 (**no CLI --seed offset** — the student distribution must
            # match training; --seed only affects the teacher llm_client). Keep both sides in sync.
            # (⚠ sync with plugin.py:392-395)
            n_assistant = sum(1 for m in messages
                              if isinstance(m, dict) and m.get("role") == "assistant")
            turn0 = max(0, n_assistant - 1)
            state["turn"] = turn0
            obs = student_observation(row, state, t_text)
            if lang == "en":
                # pass lang kwarg only for English rows: compatible with (messages, seed)-shaped stubs
                # (student_sim.student_chat's lang is optional; zh uses the default).
                stu = s_fn(obs, seed=base_seed + turn0, lang="en")
            else:
                stu = s_fn(obs, seed=base_seed + turn0)
            stu = stu if isinstance(stu, dict) else {}
            if stu.get("degraded"):
                n_degraded += 1
            raw_delta = stu.get("state_delta")
            delta = dict(raw_delta) if isinstance(raw_delta, dict) else {}
            # intel A (same as plugin.py:401): misconception_shown top-level key folded into delta (always set, empty is falsy)
            delta.setdefault("misconception_shown", stu.get("misconception_shown", ""))
            state = student_sim.apply_delta(state, delta)
            reply = stu.get("reply")
            messages.append({"role": "user",
                             "content": reply if isinstance(reply, str) and reply.strip() else "……"})
    except Exception as e:                        # never let a single row sink the batch
        out["error"] = "合成异常：%s" % e
        out["unresolved"] = True

    out["messages"] = messages
    out["turns"] = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
    out["student_degraded_turns"] = n_degraded
    return out


def synthesize_rows(rows, max_turns=5, teacher_fn=None, student_fn=None, seed=0,
                    concurrency=1):
    """Batch synthesis (row-level thread pool, order-preserving; single-row error -> error row)."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return []
    workers = max(1, min(int(concurrency or 1), len(rows)))
    if workers == 1:
        return [synthesize_one(r, max_turns, teacher_fn, student_fn, seed) for r in rows]
    out = [None] * len(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(synthesize_one, r, max_turns, teacher_fn, student_fn, seed): i
                for i, r in enumerate(rows)}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                out[i] = {"uid": str(rows[i].get("qid") or ""), "messages": [],
                          "error": "线程异常：%s" % e, "unresolved": True}
    return [o if isinstance(o, dict) else {"messages": [], "error": "空结果"} for o in out]


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.synthesize_dialogue",
        description="Step 3：candidates → sft_raw（教师=llm_client 强模型，学生=ceres_plugin 学生模拟器，≤5 轮）")
    p.add_argument("--input", default="data/queries_candidates.jsonl", help="candidates 路径")
    p.add_argument("--output", default="data/sft_raw.jsonl", help="sft_raw 输出路径")
    p.add_argument("--max-turns", type=int, default=0,
                   help="教师轮上限（默认取 env CERES_MAX_TURNS，再默认 5）")
    p.add_argument("--limit", type=int, default=0, help="最多处理条数（0=不限）")
    p.add_argument("--concurrency", type=int, default=2, help="行级并发（默认 2）")
    p.add_argument("--mock", action="store_true",
                   help="教师+学生全离线 mock（默认：key 未设时教师自动 mock；--mock 会同时强制 CERES_STUDENT_MOCK=1）")
    p.add_argument("--no-cache", action="store_true", help="关闭教师侧 LLM 缓存")
    p.add_argument("--seed", type=int, default=0, help="教师采样基准 seed（学生 seed=persona_seed+轮次，固定）")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    max_turns = args.max_turns or L.env_int("CERES_MAX_TURNS", 5)
    mock = bool(args.mock)
    if mock:
        os.environ["CERES_STUDENT_MOCK"] = "1"     # --mock forces both sides offline
    use_cache = not args.no_cache
    rows = L.read_jsonl(args.input)
    if not rows:
        sys.stderr.write("[synthesize] 输入为空：%s（先运行 rewrite_question）\n" % args.input)
        return 2
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    teacher_fn = None
    if not mock and L.is_mock_mode():
        mock = True                                # key unset: teacher side also goes offline mock
    if mock or not use_cache:
        teacher_fn = (lambda msgs, s: L.chat(msgs, seed=s, mock=mock, use_cache=use_cache))

    out = synthesize_rows(rows, max_turns=max_turns, teacher_fn=teacher_fn,
                          seed=args.seed, concurrency=args.concurrency)
    L.write_jsonl(args.output, out)
    n_err = sum(1 for r in out if r.get("error"))
    n_unres = sum(1 for r in out if r.get("unresolved") and not r.get("error"))
    n_forced = sum(1 for r in out if r.get("forced_close"))
    n_en = sum(1 for r in out if r.get("lang") == "en")
    turns = [r.get("turns") or 0 for r in out]
    avg = (sum(turns) / float(len(turns))) if turns else 0.0
    print("[synthesize] sft_raw=%s total=%d forced_close=%d unresolved=%d error=%d en=%d avg_turns=%.2f"
          % (args.output, len(out), n_forced, n_unres, n_err, n_en, avg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
