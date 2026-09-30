# -*- coding: utf-8 -*-
# ceres_plugin/plugin.py -- CERES ophthalmology teaching GRPO plugin: multi-turn virtual
# classroom environment + two rewards (ceres_sequence, ceres_anchor).
#
# swift 3.4.1 integration (functional plugin mechanism; verified against
# swift/trainers/rlhf_trainer/grpo_trainer.py:643-745):
#
#   swift sft \
#     --rlhf_type grpo \
#     --model /hy-tmp/model/Qwen2.5-VL-32B-Instruct \
#     --external_plugins ceres_plugin/plugin.py \
#     --multi_turn_func teacher_env \
#     --reward_funcs ceres_sequence ceres_anchor \
#     --reward_weights 1.0 0.5 \
#     --num_generations 8 ...
#
# swift 3.4.1 has NO --max_turns / --vllm_mode / --steps_per_generation. The per-trajectory turn
# cap comes from CERES_MAX_TURNS (default 5); G samples come from --num_generations
# (CERES_NUM_GENERATIONS is diagnostic-only for anchor grouping, not used in computation).
#
# Calling contract (from verified swift source):
#   1) multi-turn: each round the trainer builds inputs: list[dict] and calls
#      multi_turn_func(inputs); the plugin must return the SAME dicts (no add/remove). End ->
#      _input['finished']=True; continue -> append {'role':'user',...} to messages, finished=False.
#      Custom keys (_ceres_state/_ceres_traj) survive the trainer's deepcopy across rounds.
#   2) reward: completions[i] = messages[-1]['content']; kwargs = dataset extra columns (each a
#      list aligned to completions, so uid is kwargs['uid'][i]); the messages column is also
#      passed through and used to disambiguate traj_key collisions (I1/A2, see
#      _lookup_rec/_pick_by_messages).
#   3) when reward names hit orms, the trainer inspects __init__ to inject args; the two ORM
#      classes here define no __init__, so CeresSequenceORM()/CeresAnchorORM() instantiate arg-free.
#
# swift import isolation: the swift import sits in a module-level try. Without swift (or with
# CERES_PLUGIN_STUB_SWIFT=1) it degrades to "logic-only" mode: teacher_env and both rewards still
# work, orms/multi_turns register to a local empty table with a warning; real training must load
# this file via --external_plugins under ms-swift 3.4.1.
#
# Deps: stdlib + three sibling modules (grammar/student_sim/store, flat-imported via sys.path).
# No numpy (means/vars use pure Python). All entry points are defensive against arbitrary trainer
# structure / dataset columns / LLM output / store content -- never raise uncaught.
import hashlib
import json
import logging
import math
import os
import sys

# Flat-import the three sibling modules: swift's import_external_file inserts this directory into
# sys.path; the line below covers the package (ceres_plugin.plugin) import path too.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from grammar import check_grammar_global, check_grammar_turn, parse_actions  # noqa: E402
from store import (TrajectoryStore, anchor_key, final_eval, make_traj_key,  # noqa: E402
                   step_reward)
from student_sim import apply_delta, init_state, student_chat  # noqa: E402

__all__ = [
    "teacher_env", "ceres_sequence_reward", "ceres_anchor_reward",
    "CeresSequenceORM", "CeresAnchorORM", "W", "RISK_PATTERNS", "STORE",
    "SWIFT_AVAILABLE", "orms", "multi_turns", "DEFAULT_END_PENALTY",
]

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------- Env vars
DEFAULT_MAX_TURNS = 5          # CERES_MAX_TURNS: teacher turn cap per trajectory (swift has no --max_turns)
DEFAULT_NUM_GENERATIONS = 8    # CERES_NUM_GENERATIONS: G samples per prompt (true value from --num_generations)
#: CERES_END_PENALTY: fixed sequence-reward penalty for non-<end/> endings (length truncation /
#: max_turns / error sealing). Default 0.2 (introduced after the 2026-09-02 train_oph_lora.log
#: collapse; set 0 to ablate back to old behavior).
DEFAULT_END_PENALTY = 0.2


def _env_int(name, default):
    """Read an int env var at call time (monkeypatch-friendly); fall back to default on invalid."""
    try:
        return int(float(str(os.environ.get(name, default)).strip()))
    except Exception:
        return default


def _env_float(name, default):
    """Read a float env var at call time; fall back to default on invalid/NaN."""
    try:
        v = float(str(os.environ.get(name, default)).strip())
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _end_penalty():
    """Non-<end/> termination penalty: CERES_END_PENALTY (default DEFAULT_END_PENALTY),
    clamped to >= 0 (a negative penalty would inflate reward, so it is rejected)."""
    return max(0.0, _env_float("CERES_END_PENALTY", DEFAULT_END_PENALTY))


def _max_turns():
    return max(1, _env_int("CERES_MAX_TURNS", DEFAULT_MAX_TURNS))


def _num_generations():
    return max(1, _env_int("CERES_NUM_GENERATIONS", DEFAULT_NUM_GENERATIONS))


#: Global trajectory store: written by teacher_env (put maintains the uid index), read by the rewards.
STORE = TrajectoryStore()


# ---------------------------------------------------------------- swift import (isolated in try)
def _truthy_env(name):
    v = str(os.environ.get(name, "")).strip().lower()
    return v in ("1", "true", "yes", "on")


try:
    if _truthy_env("CERES_PLUGIN_STUB_SWIFT"):
        # pytest isolation switch: don't import swift even if installed (torch/transformers too heavy)
        raise ImportError("CERES_PLUGIN_STUB_SWIFT=1（测试隔离要求，跳过真实 swift 导入）")
    from swift.plugin import ORM, orms, multi_turns  # noqa: E402
    SWIFT_AVAILABLE = True
except Exception as _e:  # ImportError or any cascading swift import failure
    SWIFT_AVAILABLE = False

    class _StubORM:  # mirrors swift.plugin.orm.ORM: constrains only the __call__ signature
        def __call__(self, **kwargs):  # pragma: no cover - placeholder only
            raise NotImplementedError

    ORM = _StubORM
    orms, multi_turns = {}, {}
    _LOG.warning(
        "[ceres plugin] swift 不可用（%s）。已降级为纯逻辑模式：teacher_env 与奖励函数仍可"
        "独立运行/测试，但 orms/multi_turns 只注册到本地空表；真实训练必须在装有 ms-swift "
        "3.4.1 的环境用 --external_plugins ceres_plugin/plugin.py 加载本文件。", _e)


# ---------------------------------------------------------------- Small helpers (defensive)
def _msg_text(content):
    """Safely coerce message content to str: None / str / list (multimodal chunks) / dict."""
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


def _s(v):
    """Coerce a dataset column to a prompt-safe str (None -> "", structured -> JSON)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple, dict)):
        try:
            return json.dumps(v, ensure_ascii=False, default=str)
        except Exception:
            return str(v)
    return str(v)


def _uid_of(_input):
    """Get uid normalized to str (column may be int / None / missing)."""
    try:
        v = _input.get("uid", "")
        return "" if v is None else str(v)
    except Exception:
        return ""


def _count_assistant(messages):
    """Number of assistant (teacher) turns in messages -- the multi-turn cap counter."""
    n = 0
    for m in messages if isinstance(messages, (list, tuple)) else []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            n += 1
    return n


def _last_teacher_text(messages):
    """Last teacher turn text; the last message must be assistant (contract), but scan backwards defensively."""
    if not isinstance(messages, (list, tuple)):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return _msg_text(m.get("content"))
    return ""


def _persona_seed_int(dd):
    """persona_seed -> int (must be converted to int before calling student_chat).

    Missing/None/blank/non-numeric -> 0, so student seed = turn (see _step_one). Deliberately
    persona_seed + turn (not design's 2*turn), kept in sync with synthesize_dialogue. Tries
    int() then int(float()); non-numeric strings hash to a stable sha1 prefix so distinct
    personas stay distinguishable. Never raises.
    """
    v = dd.get("persona_seed", 0)
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
    return int(hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8], 16)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _std0(xs):
    """Population std (ddof=0, matching np.std default in design §7.5)."""
    xs = list(xs)
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _to_float(v, default=0.0):
    """Defensive float conversion: invalid / NaN / inf -> default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ---------------------------------------------------------------- Multi-turn environment plugin (functional)
def _end_reason(teacher_text, n_assistant, finish_reason):
    """Three end conditions (checked in order; returns finished_reason or None):

    1) teacher text contains <end (incl. <end/> and malformed <end>, wide match to block bypass) -> "end_tag"
    2) assistant turns >= CERES_MAX_TURNS (default 5) -> "max_turns"
    3) finish_reason == 'length' (trainer rewrites it each round; it stops the sample itself,
       but if the plugin doesn't seal, reward can't find the trajectory) -> "length"
    """
    if "<end" in (teacher_text or ""):
        return "end_tag"
    if n_assistant >= _max_turns():
        return "max_turns"
    if finish_reason == "length":
        return "length"
    return None


def _lang_of(dd):
    """Dataset row -> 'zh'|'en' (lang column; fallback 'zh' for backward compat). Must match
    data_pipeline.synthesize_dialogue._row_lang (pinned by cross-check tests)."""
    dd = dd if isinstance(dd, dict) else {}
    lang = dd.get("lang")
    return "en" if isinstance(lang, str) and lang.strip().lower() == "en" else "zh"


def _student_messages(dd, state, teacher_text):
    """Build the student simulator observation (single user message, text-only).

    dd['lang']=='en' -> English labels and response instruction. Must stay field-for-field in
    sync with data_pipeline.synthesize_dialogue.student_observation (pinned by tests).
    """
    try:
        state_json = json.dumps(state, ensure_ascii=False, default=str)
    except Exception:
        state_json = str(state)
    if _lang_of(dd) == "en":
        return [{
            "role": "user",
            "content": (
                f"[Course context / findings] {_s(dd.get('courseware_context', ''))}\n"
                f"[Learner profile] {_s(dd.get('learner_profile', ''))}\n"
                f"[Current learner state] {state_json}\n"
                f"[Teacher's action this turn]\n{teacher_text}\n"
                f"Please respond as an ophthalmology resident-in-training and output JSON."
            )}]
    return [{
        "role": "user",
        "content": (
            f"【课程上下文/检查所见】{_s(dd.get('courseware_context', ''))}\n"
            f"【学习者画像】{_s(dd.get('learner_profile', ''))}\n"
            f"【当前学习者状态】{state_json}\n"
            f"【教师本轮动作】\n{teacher_text}\n"
            f"请以眼科规培生身份回应，并输出 JSON。"
        )}]


def _append_step(rec, turn, teacher_text, student_reply, state_before, state_after,
                 delta, degraded, dd, fixed_step_reward=None):
    """Record one step (full contract schema; misconception_resolved/degraded are extra
    observation keys for G4 state overlay and offline analysis). If fixed_step_reward is not
    None, use it directly (end turn has no student interaction -> always 0).
    """
    rec["steps"].append({
        "turn": turn,
        "teacher": teacher_text,
        "student_reply": student_reply,
        "state_before": state_before,
        "state_after": state_after,
        "step_reward": (fixed_step_reward if fixed_step_reward is not None
                        else step_reward(state_after, delta, teacher_text)),
        "anchor": anchor_key(dd, state_after, teacher_text),
        "turn_ok": check_grammar_turn(teacher_text),
        "misconception_resolved": bool(delta.get("misconception_resolved")),
        "degraded": bool(degraded),
    })


def _resolved_seen(steps):
    """Whether any "misconception resolved" evidence exists: the step flag, or a severity drop
    >= 0.3 in one step (resolved is -0.5, shown is +0.1, so >=0.3 implies resolved)."""
    for st in steps:
        if not isinstance(st, dict):
            continue
        if st.get("misconception_resolved"):
            return True
        sb = st.get("state_before") if isinstance(st.get("state_before"), dict) else {}
        sa = st.get("state_after") if isinstance(st.get("state_after"), dict) else {}
        if _to_float(sb.get("misconception_severity"), None) is not None:
            drop = (_to_float(sb.get("misconception_severity"))
                    - _to_float(sa.get("misconception_severity"), 0.0))
            if drop >= 0.3:
                return True
    return False


def _g4_state_violation(steps):
    """G4 state-condition overlay (checked by the plugin on top of check_grammar_global):

    trajectory ends with <end/> (design §5: must have corrected or student answered correctly)
    but never had <correct> and no round resolved the misconception -> violation.
    """
    if not steps:
        return False
    last = steps[-1] if isinstance(steps[-1], dict) else {}
    last_acts = [t for t, _a, _b in parse_actions(last.get("teacher", ""))]
    if "end" not in last_acts:
        return False
    for st in steps:
        if not isinstance(st, dict):
            continue
        for t, _a, _b in parse_actions(st.get("teacher", "")):
            if t == "correct":
                return False  # explicit correction seen -> not a violation
    return not _resolved_seen(steps)


def _seal(_input, rec, state, teacher_text, reason, turn):
    """Seal a finished trajectory: append the last step + final eval + G2~G5 (with G4 overlay)
    + make_traj_key into the store and disk.

    Idempotent: the step is appended once per turn order (see comment below).
    reason in {"end_tag","max_turns","length","error"}; "error" is a defensive marker outside
    the contract enum, for offline debugging only (not consumed by reward).
    """
    steps = rec.setdefault("steps", [])
    if not isinstance(steps, list):
        steps = rec["steps"] = []
    last_step = steps[-1] if steps else None
    # Idempotence keyed on turn order, not teacher text (review C1): max_turns-truncated
    # trajectories often repeat the previous text verbatim; text-equality would wrongly skip
    # the real last step (losing one turn from f_fmt denominator, suffix return, anchor
    # buckets, G4 overlay). Normal seal: steps[-1].turn == turn-1 != turn -> append; abnormal
    # re-entry of the same turn -> skip, idempotence preserved.
    if not (isinstance(last_step, dict) and last_step.get("turn") == turn):
        # The final turn (with <end/>/cap/truncation) must stay in steps: G4/G5, reward's
        # f_fmt/f_rule/f_risk, and traj_key recomputation all depend on it. The end turn has
        # no student interaction or state transition, so its immediate reward is always 0.
        _append_step(rec, turn, teacher_text, "", dict(state), dict(state), {}, False,
                     _input, fixed_step_reward=0.0)

    uid = str(rec.get("uid", "") or _uid_of(_input))
    rec["uid"] = uid
    rec["lang"] = _lang_of(_input)   # language observation key (not consumed by reward; eval/debug only)
    rec["final_state"] = final_eval(rec)
    global_ok = bool(check_grammar_global(rec["steps"], _input))
    if global_ok and _g4_state_violation(rec["steps"]):
        global_ok = False  # G4 state condition overlaid by the plugin
    rec["grammar_global_ok"] = global_ok
    key = make_traj_key(uid, teacher_text)  # retrieval key: both sides share this derivation, never hand-roll the hash
    rec["traj_key"] = key
    rec["finished_reason"] = reason
    STORE.put(key, rec)      # put maintains the uid index (reward fallback lookup)
    STORE.flush(rec)         # append to disk (IO failure only warns)
    _input["_ceres_traj"] = rec
    _input["_ceres_state"] = state


def _step_one(_input):
    """Process one input: if ending -> seal; otherwise step the student simulator and append a user message."""
    if _input.get("finished"):  # trainer already filtered finished samples; defensive idempotence
        return

    messages = _input.get("messages")
    if not isinstance(messages, list):
        messages = _input["messages"] = []
    teacher_text = _last_teacher_text(messages)
    n_assistant = _count_assistant(messages)

    state = _input.get("_ceres_state")
    if not (isinstance(state, dict) and state):
        state = init_state(_input)
    rec = _input.get("_ceres_traj")
    if not (isinstance(rec, dict) and isinstance(rec.get("steps"), list)):
        rec = {"uid": _uid_of(_input), "steps": []}
    _input["_ceres_state"] = state
    _input["_ceres_traj"] = rec

    reason = _end_reason(teacher_text, n_assistant, _input.get("finish_reason"))
    if reason:
        _seal(_input, rec, state, teacher_text, reason, max(0, n_assistant - 1))
        _input["finished"] = True
        return

    turn = max(0, n_assistant - 1)            # 0-based teacher turn index (used by steps)
    state["turn"] = turn
    seed = _persona_seed_int(_input) + turn   # must convert to int first, then add turn
    obs = _student_messages(_input, state, teacher_text)
    lang = _lang_of(_input)
    if lang == "en":
        # Pass lang kwarg only for English rows: student_chat's lang is optional (default zh);
        # not passing it on the zh path keeps the old call shape (compatible with injected
        # (messages, seed) stubs). Teacher output language follows conversation history.
        stu = student_chat(obs, seed=seed, lang="en")
    else:
        stu = student_chat(obs, seed=seed)
    if not isinstance(stu, dict):
        stu = {}
    # misconception_shown is top-level in the returned dict; apply_delta only reads it inside state_delta
    raw_delta = stu.get("state_delta")
    delta = dict(raw_delta) if isinstance(raw_delta, dict) else {}
    delta.setdefault("misconception_shown", stu.get("misconception_shown", ""))

    state_before = dict(state)
    state_after = apply_delta(state, delta)   # in-place update; mastery/engagement/fatigue clamped to [0,1]
    # Empty-reply placeholder (I3/A4), identical to synthesize_dialogue.py: if reply is not a
    # non-blank str, use "……" and never append an empty-content user turn. _sanitize only
    # fills placeholders for missing keys, so a real API reply:"" still needs this re-check.
    reply = stu.get("reply")
    reply = reply if isinstance(reply, str) and reply.strip() else "……"
    _append_step(rec, turn, teacher_text, reply, state_before, dict(state_after),
                 delta, stu.get("degraded", False), _input)

    _input["_ceres_state"] = state_after
    _input["_ceres_traj"] = rec
    # student reply is the next round's observation (text-only user msg; images only in the first round)
    messages.append({"role": "user", "content": reply})
    _input["finished"] = False


def _maybe_clear_store_for_new_batch(inputs):
    """Clear the store at batch start (review I2-2 minimal mitigation; safe under pt infer).

    If every dict input lacks '_ceres_state' (present only on continued-round samples, surviving
    the trainer deepcopy), it's a fresh generation batch -> clear the store. Under pt infer,
    rollout and scoring run serially in the same batch, so the previous batch is cleared only
    after its rewards are computed. Benefit: no unbounded uid-index growth; a smaller uid
    fallback candidate set (just this batch's G entries for the uid).

    Uses store.clear() (locks and clears both indexes), no private-structure access.
    """
    try:
        dicts = [x for x in inputs if isinstance(x, dict)]
        if not dicts or any("_ceres_state" in x for x in dicts):
            return False  # empty batch or contains continued-round samples -> not a new batch, don't clear
        STORE.clear()
        return True
    except Exception:
        return False  # clear failure doesn't affect the main flow (worst case: unbounded growth)


def teacher_env(inputs):
    """swift multi-turn plugin (multi_turns['teacher_env']): step the virtual classroom once.

    Inputs/outputs: the trainer's inputs list[dict] -- return the SAME dicts, never add/remove
    (the trainer reassembles by 'index'; _ceres_state/_ceres_traj survive deepcopy). Per item:
    end (<end / turns >= CERES_MAX_TURNS / finish_reason=='length') -> seal + finished=True;
    continue -> append the student reply as a user message, finished=False. Per-item exceptions
    are caught and treated as "end + conservative seal". Batch-start clear runs first.
    """
    if not isinstance(inputs, list):
        _LOG.warning("[ceres plugin] teacher_env 收到非 list 输入（%s），原样返回",
                     type(inputs).__name__)
        return inputs
    _maybe_clear_store_for_new_batch(inputs)
    for _input in inputs:
        if not isinstance(_input, dict):
            _LOG.warning("[ceres plugin] teacher_env 跳过非 dict 元素: %r", type(_input))
            continue
        try:
            _step_one(_input)
        except Exception as e:  # per-item failure must not abort the whole batch
            _LOG.warning("[ceres plugin] teacher_env 处理单条输入异常（保守结束）: %r", e)
            try:
                msgs = _input.get("messages")
                state = _input.get("_ceres_state")
                if not (isinstance(state, dict) and state):
                    state = init_state(_input)
                rec = _input.get("_ceres_traj")
                if not (isinstance(rec, dict) and isinstance(rec.get("steps"), list)):
                    rec = {"uid": _uid_of(_input), "steps": []}
                _seal(_input, rec, state, _last_teacher_text(msgs), "error",
                      max(0, _count_assistant(msgs) - 1))
            except Exception as e2:
                _LOG.warning("[ceres plugin] 保守封存也失败（放弃该条轨迹）: %r", e2)
            _input["finished"] = True
    return inputs


# ---------------------------------------------------------------- Rewards: weights and red lines
# Four paper weights. learner-outcome is currently unused (out=0; see the reserved block below);
# fmt/rule scaled from 0.2:0.3 to 0.4:0.6 so positive terms sum to 1; risk remains a penalty.
W = dict(fmt=0.4, rule=0.6, out=0.0, risk=0.1)      # restore dict(fmt=0.2, rule=0.3, out=0.4, risk=0.1) for deployment
# Ophthalmology safety red lines (examples; production should add a medical LLM reviewer + checklist)
RISK_PATTERNS = [
    "不用查眼压", "滴眼液没有禁忌", "直接手术", "立刻手术", "这个剂量是",
    "不用散瞳", "激素随便用", "肯定不是青光眼", "确诊就是",
]


def _completion_text(c):
    """Coerce a completion (str or list[dict] multi-turn messages) to plain text."""
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)):
        return "\n".join(_msg_text(m.get("content", "")) for m in c if isinstance(m, dict))
    return "" if c is None else str(c)


def _uid_at(kwargs, i, n):
    """Get the i-th uid from kwargs' uid column (a list aligned to completions); tolerate missing/scalar/out-of-range."""
    col = kwargs.get("uid")
    if isinstance(col, (list, tuple)):
        v = col[i] if 0 <= i < len(col) else None
        return "" if v is None else str(v)
    if col is None:
        return ""
    return str(col) if n == 1 else ""  # a scalar is only trusted when batch=1, otherwise give up


#: Min length for the uid-fallback "suffix" branch (review I2-1): a short text is almost always
#: a suffix of some long text from the same uid, so below this threshold only exact equality is
#: allowed; exact equality (the primary key is derived from full text) is always allowed.
_SUFFIX_MATCH_MIN_LEN = 32


def _pick_by_messages(cands, kwargs, i):
    """Disambiguate collided traj_key candidates (I1/A2): compare kwargs['messages'][i] (the
    sample's full multi-turn messages) against each candidate's steps[*]['teacher'] sequence.

    The messages column is passed through by the trainer (verified), so comparison is free. If
    missing / length mismatch / no match -> deterministically take cands[-1] and log debug.
    Never raises, no fuzzy matching (conservative when undecidable).
    """
    msgs = None
    col = kwargs.get("messages")
    if isinstance(col, (list, tuple)) and 0 <= i < len(col):
        msgs = col[i]
    texts = []
    if isinstance(msgs, (list, tuple)):
        texts = [_msg_text(m.get("content", "")) for m in msgs
                 if isinstance(m, dict) and m.get("role") == "assistant"]
    if texts:
        for cand in cands:
            if not isinstance(cand, dict):
                continue
            steps = cand.get("steps")
            steps = [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []
            teachers = [s.get("teacher") if isinstance(s.get("teacher"), str) else ""
                        for s in steps]
            if teachers == texts:
                return cand
    _LOG.debug("[ceres plugin] traj_key 碰撞（%d 候选）且 kwargs['messages'] 不可判别 → "
               "取最后封存者", len(cands))
    return cands[-1] if cands else None


def _lookup_rec(kwargs, i, completions):
    """Look up a trajectory: primary make_traj_key lookup (disambiguate collisions via
    kwargs['messages']), then uid-index fallback (same last teacher text, or mutual suffix when
    length >= 32), else warn.

    The primary key = hash(uid, last completion text) is not injective: G rollouts of one uid can
    repeat the same last text verbatim (common under max_turns truncation) -> multi-valued main
    index (store.put appends). One candidate: direct; >1: _pick_by_messages. The fallback covers
    truncated/rewritten completions but enforces a 32-char minimum overlap to avoid mismatching
    short completions to other trajectories of the same uid.
    """
    n = len(completions) if isinstance(completions, (list, tuple)) else 1
    uid = _uid_at(kwargs, i, n)
    text = _completion_text(completions[i]).strip()
    cands = STORE.get_all(make_traj_key(uid, text))
    rec = None
    if len(cands) == 1:
        rec = cands[0]                        # single candidate: same path as legacy get()
    elif len(cands) > 1:
        rec = _pick_by_messages(cands, kwargs, i)   # traj_key collision -> full-text disambiguation
    if rec is None and uid:
        allow_suffix = len(text) >= _SUFFIX_MATCH_MIN_LEN
        for cand in STORE.get_by_uid(uid):
            last_t = ""
            if isinstance(cand, dict):
                steps = cand.get("steps")
                if isinstance(steps, list) and steps and isinstance(steps[-1], dict):
                    last_t = _msg_text(steps[-1].get("teacher", "")).strip()
            if not (last_t and text):
                continue
            if last_t == text or (allow_suffix and (last_t.endswith(text) or text.endswith(last_t))):
                rec = cand
                break
    if rec is None:
        # only log the first 8 chars of uid, never the full completion (avoid log spam / sensitive info)
        _LOG.warning("[ceres plugin] 轨迹检索失败：uid=%s… i=%d/%d → 奖励记 0.0",
                     (uid[:8] or "<空>"), i, n)
    return rec, uid


# ---------------------------------------------------------------- 1) Sequence-level reward
def ceres_sequence_reward(completions, **kwargs):
    """ceres_sequence: f_format + f_rule - f_risk (learner-outcome reserved, see the block below).

    completions[i] = the trajectory's last assistant text; kwargs = extra dataset columns
    (each a list aligned to completions). Lookup miss -> 0.0 for that sample (never raises).
    """
    completions = list(completions or [])
    out = []
    for i, comp in enumerate(completions):
        rec, _uid = _lookup_rec(kwargs, i, completions)
        if not isinstance(rec, dict):
            out.append(0.0)
            continue
        turns = rec.get("steps")
        turns = [t for t in turns if isinstance(t, dict)] if isinstance(turns, list) else []
        if not turns:
            out.append(0.0)
            continue
        text_all = "\n".join(_s(t.get("teacher", "")) for t in turns)
        turn_oks = [_to_float(t.get("turn_ok"), 0.0) for t in turns]
        grammar_ok = bool(rec.get("grammar_global_ok"))
        f_fmt = _mean(turn_oks) * (1.0 if grammar_ok else 0.3)
        acts = [tag for t in turns for (tag, _a, _b) in parse_actions(_s(t.get("teacher", "")))]
        scaffold = ("recall" in acts) + ("hint" in acts)          # 0~2
        student_part = sum(1 for t in turns if "<check" in _s(t.get("teacher", "")))
        order_ok = 1.0 if grammar_ok else 0.0
        f_rule = _clip(0.25 * scaffold + 0.25 * min(1, student_part) + 0.5 * order_ok, 0, 1)
        # ════════════ Reserved: learner-outcome term (w3·f_outcome) — unused, enable for deployment ════════════
        # Reason: final mastery/engagement/misconception come from the LLM student's self-report
        # (final_eval) — noisy, unreliable in absolute value, and easy to inflate via scripted
        # praise (classic reward hacking). So during experiments this term is removed entirely;
        # only hard-checkable format/rule/risk remain.
        # Once a trustworthy outcome signal exists (pre/post tests, report scores, expert-
        # calibrated evaluator), restore the three lines below and reset
        # W = dict(fmt=0.2, rule=0.3, out=0.4, risk=0.1):
        #     fs    = rec["final_state"]
        #     f_out = float(np.clip(0.6 * fs["mastery"] + 0.2 * fs["engagement"]
        #                           + 0.2 * (1 - fs["misconception_severity"]), 0, 1))
        #     score += W["out"] * f_out
        # ═══════════════════════════════════════════════════════════════════════════════════
        f_risk = 0.0
        if any(p in text_all for p in RISK_PATTERNS):
            f_risk += 1.0
        if "<correct" in text_all and student_part == 0:
            f_risk += 1.0   # correcting without letting the student answer (diagnosis-first principle, design §1.2)
        score = W["fmt"] * f_fmt + W["rule"] * f_rule - W["risk"] * f_risk
        # [2026-09-02] End penalty: non-<end/> endings (truncation / turn cap / error seal) are
        # penalized by _end_penalty(). Background: in the first full training run, the <end/> rate
        # collapsed 55% -> 0% and the second half was 100% max-length truncated; since G1~G5 don't
        # require <end/> (G4 only checks position), truncated trajectories stayed ~98% grammar-ok
        # and reward only slid 0.96 -> 0.75, so the collapse was nearly invisible and GRPO kept
        # training the reward downward. This prices "correct termination" explicitly: only end_tag
        # is exempt; a missing finished_reason key is also penalized (_seal always writes it).
        if rec.get("finished_reason") != "end_tag":
            score -= _end_penalty()
        out.append(float(score))
    return out


# ---------------------------------------------------------------- 2) Anchor-level group-relative credit
def ceres_anchor_reward(completions, **kwargs):
    """ceres_anchor: scalarized learner-state-aware group-relative credit assignment.

    For the G trajectories of one uid, aggregate per-step suffix returns by anchor and normalize
    within each bucket ((v-μ)/(σ+1e-6)); each trajectory's credit = mean advantage over the
    buckets it actually participates in (mean over participating buckets, not all buckets, to
    avoid diluting credit to 0 via single-member buckets; single-member buckets are skipped),
    clipped to ±3. Missing/empty uid column or lookup failure -> credit 0.
    """
    completions = list(completions or [])
    n = len(completions)
    recs, uids = [], []
    for i in range(n):
        rec, uid = _lookup_rec(kwargs, i, completions)
        recs.append(rec if isinstance(rec, dict) else None)
        uids.append(uid)

    credit = [0.0] * n
    if not any(uids):  # uid column missing/empty: no relative comparison possible, all 0 (no warning; seq reward covers it)
        return credit

    by_uid = {}
    for i, uid in enumerate(uids):
        if uid:
            by_uid.setdefault(uid, []).append(i)

    g_expected = _num_generations()  # diagnostic only: group size != G implies lookup failures
    for uid, idxs in by_uid.items():
        buckets = {}   # anchor -> [(i, suffix_return)]
        for i in idxs:
            rec = recs[i]
            if rec is None:
                continue
            steps = rec.get("steps")
            steps = [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []
            suffix = [0.0] * len(steps)                      # suffix return: downstream utility
            acc = 0.0
            for t in range(len(steps) - 1, -1, -1):
                acc += _to_float(steps[t].get("step_reward"), 0.0)
                suffix[t] = acc
            for t, st in enumerate(steps):
                buckets.setdefault(_s(st.get("anchor", "")), []).append((i, suffix[t]))
        if g_expected > 1 and len(idxs) != g_expected:
            _LOG.debug("[ceres plugin] uid=%s… 组内 %d 条（预期 G=%d，可能有轨迹检索失败）",
                       uid[:8], len(idxs), g_expected)
        used = [0] * n  # number of buckets each trajectory got credit from (mean denominator)
        for _anchor, items in buckets.items():
            if len(items) < 2:   # skip single-member buckets: no within-group comparability
                continue
            vals = [v for _i, v in items]
            mu, sd = _mean(vals), _std0(vals) + 1e-6
            for (i, v) in items:
                credit[i] += (v - mu) / sd
                used[i] += 1
        for i in idxs:
            if used[i] > 0:
                credit[i] /= used[i]   # trajectory mean over participating buckets
    return [float(_clip(c, -3.0, 3.0)) for c in credit]


# ---------------------------------------------------------------- 3) Registration
class CeresSequenceORM(ORM):
    """swift ORM wrapper: no __init__ (the trainer inspects __init__ to inject training args)."""

    def __call__(self, completions=None, **kwargs):
        return ceres_sequence_reward(completions or [], **kwargs)


class CeresAnchorORM(ORM):

    def __call__(self, completions=None, **kwargs):
        return ceres_anchor_reward(completions or [], **kwargs)


orms["ceres_sequence"] = CeresSequenceORM
orms["ceres_anchor"] = CeresAnchorORM
multi_turns["teacher_env"] = teacher_env


# ---------------------------------------------------------------- [2026-08-31] Crash-protection callback registration (work package C)
# SaveOnExit: before any uncaught exception / SIGTERM / SIGINT exits the process, save the LoRA
# deltas to <output_dir>/emergency-adapter (see save_on_exit.py docstring).
# Registration mechanism (verified in swift 3.4.1): swift/plugin/callback.py:30 defines a
# module-level shared extra_callbacks list; sft.py:231 does `callbacks += extra_callbacks`
# (shared by SwiftSft/SwiftRLHF/SwiftPt) -- appending takes effect with no CLI flag. This file
# is loaded via --external_plugins during arg parsing (before the callback table is collected).
# Only registered when real swift is available (non-stub); registration failure only warns.
if SWIFT_AVAILABLE:
    try:
        from save_on_exit import register_save_on_exit  # noqa: E402  flat import (_HERE is already on sys.path)
        _SAVE_ON_EXIT_REGISTERED = register_save_on_exit()
    except Exception as _save_exit_e:  # noqa: BLE001
        _SAVE_ON_EXIT_REGISTERED = False
        _LOG.warning("[ceres plugin] save_on_exit 注册失败（训练继续，仅失去崩溃保护）：%r",
                     _save_exit_e)
