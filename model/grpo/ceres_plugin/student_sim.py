# ceres_plugin/student_sim.py -- virtual ophthalmology resident simulator (design §7.2 + mock mode).
#
# Contract signatures (do not change; lang is an optional new kwarg, default 'zh', backward compatible):
#   student_chat(messages, seed=0, temperature=0.9, lang='zh') -> dict
#       {"reply": str, "state_delta": {...}, "misconception_shown": str}; adds "degraded": True on fallback
#       lang='en': real path appends an English output instruction; mock path emits English templates
#       (determinism unchanged: same input + seed -> same output)
#   init_state(dd: dict) -> dict
#   apply_delta(state: dict, delta: dict) -> dict     # in-place; mastery/engagement/fatigue clamped to [0,1]
#
# Env vars: CERES_STUDENT_API_BASE / CERES_STUDENT_API_KEY / CERES_STUDENT_MODEL /
#           CERES_STUDENT_CONCURRENCY / CERES_STUDENT_TIMEOUT / CERES_STUDENT_MOCK
# Mock mode: CERES_STUDENT_MOCK=1 or no API key -> fully offline, deterministic seed-based replies.
# Defensive: any external input (LLM/env/args) is guarded; training never crashes.
import hashlib
import json
import math
import os
import random
import threading
import time

__all__ = ["student_chat", "init_state", "apply_delta", "is_mock_mode", "SYSTEM_PROMPT",
           "SYSTEM_PROMPT_EN_ADDON", "_norm_lang"]

# ---------------- Env vars (read at call time so tests can monkeypatch) ----------------
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # DashScope compatible-mode
DEFAULT_MODEL = "qwen-plus"      # switch to qwen-vl-plus when images are needed
DEFAULT_CONCURRENCY = 64
DEFAULT_TIMEOUT = 30.0
RETRY_TIMES = 3                  # 3 backoff retries
_BACKOFF_BASE = 1.5              # backoff base: 1.5s / 3.0s
_CACHE_MAX = 200_000             # cache soft cap, clear all when exceeded (mock can deterministically regenerate)

# System prompt kept verbatim from design §7.2
SYSTEM_PROMPT = """你是一个高保真的「眼科规培生模拟器」。根据给定的学习者状态和带教老师刚发出的教学动作，
以该规培生的口吻生成真实、自然的课堂回应（2~5 句），并客观评估这次教学动作引起的状态变化。
严格只输出 JSON：
{"reply": "学生的口头回应",
 "state_delta": {"mastery": 0.0~0.2 之间的浮点(可为负), "engagement": -0.2~0.2, "fatigue": 0~0.1,
                  "misconception_resolved": true/false},
 "misconception_shown": "本轮暴露出的错误概念关键词（如：把出血点数量当分期唯一标准），没有则空串"}
规则：mastery 只有在学生真正理解时才为正；教师直接给完整答案/诊断时 mastery 增益记 0；
教师只是提问/check 时若学生答错，mastery 可为 0 并暴露误解；学生水平与其画像一致。"""

#: Appended after SYSTEM_PROMPT when lang='en' (JSON schema unchanged; only constrains language)
SYSTEM_PROMPT_EN_ADDON = (
    "IMPORTANT: Always respond in English. The reply field must be natural classroom English; "
    "keep the JSON schema unchanged."
)

# Degraded default reply: neutral, no state change (design §7.2), adds "degraded": True
_DEGRADED = {
    "reply": "老师，我还有点没想明白，能再讲讲吗？",
    "state_delta": {"mastery": 0.0, "engagement": -0.05, "fatigue": 0.05,
                    "misconception_resolved": False},
    "misconception_shown": "",
    "degraded": True,
}
_DEGRADED_EN = {
    "reply": "Dr., I'm still a bit confused here — could you go over it again?",
    "state_delta": {"mastery": 0.0, "engagement": -0.05, "fatigue": 0.05,
                    "misconception_resolved": False},
    "misconception_shown": "",
    "degraded": True,
}


def _norm_lang(lang):
    """Normalize lang: 'en' (case/whitespace tolerant) -> 'en'; anything else -> 'zh'."""
    return "en" if isinstance(lang, str) and lang.strip().lower() == "en" else "zh"


def _env(name, default=""):
    """Read an env var, always returning str (env is untrusted)."""
    try:
        v = os.environ.get(name, default)
        return v if isinstance(v, str) else str(v)
    except Exception:
        return default


def _env_int(name, default):
    try:
        return int(float(_env(name, str(default)).strip()))
    except Exception:
        return default


def _env_float(name, default):
    try:
        return float(_env(name, str(default)).strip())
    except Exception:
        return default


def _to_float(v, default=0.0):
    """Defensive float conversion: invalid / NaN / inf -> default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def is_mock_mode():
    """Mock mode if CERES_STUDENT_MOCK is truthy or no API key is set (offline required)."""
    flag = _env("CERES_STUDENT_MOCK", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    return _env("CERES_STUDENT_API_KEY", "").strip() == ""


# ---------------- Cache / rate-limit / client (all lazy; importing never touches the network) ----------------
_CACHE, _CACHE_LOCK = {}, threading.Lock()
_SEM, _SEM_LOCK = None, threading.Lock()      # threading.Semaphore(CERES_STUDENT_CONCURRENCY)
_CLIENT, _CLIENT_LOCK = None, threading.Lock()  # OpenAI-compatible client, created on first real call


def _get_sem():
    """Lazily create the rate-limit semaphore (concurrency from env, default 64)."""
    global _SEM
    if _SEM is None:
        with _SEM_LOCK:
            if _SEM is None:
                _SEM = threading.Semaphore(max(1, _env_int("CERES_STUDENT_CONCURRENCY", DEFAULT_CONCURRENCY)))
    return _SEM


def _get_client():
    """Lazily create the OpenAI-compatible client; raises if openai is missing/creation fails (caller degrades)."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        from openai import OpenAI  # lazy import: module stays usable in mock mode / without openai
        _CLIENT = OpenAI(base_url=_env("CERES_STUDENT_API_BASE", DEFAULT_API_BASE) or DEFAULT_API_BASE,
                         api_key=_env("CERES_STUDENT_API_KEY", ""),
                         timeout=_env_float("CERES_STUDENT_TIMEOUT", DEFAULT_TIMEOUT))
        return _CLIENT


def _cache_key(messages, seed):
    """sha1(messages + seed): reuse results for same context + seed (design §7.2)."""
    try:
        blob = json.dumps(list(messages), ensure_ascii=False)
    except Exception:
        blob = repr(messages)
    return hashlib.sha1((blob + f"#{seed}").encode("utf-8", "ignore")).hexdigest()


def _cache_put(key, out):
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()  # soft cap: just clear (mock is deterministic; real API just re-calls)
        _CACHE[key] = out


def _sanitize(out):
    """Defensively normalize LLM JSON: ensure contract keys exist and types are correct."""
    if not isinstance(out, dict):
        raise ValueError("student_chat 返回非 dict JSON")
    reply = out.get("reply", "……")
    out["reply"] = reply if isinstance(reply, str) else str(reply)
    delta = out.get("state_delta")
    out["state_delta"] = delta if isinstance(delta, dict) else {}
    shown = out.get("misconception_shown", "")
    out["misconception_shown"] = shown if isinstance(shown, str) else str(shown)
    return out


# ---------------- Mock mode: fully offline, deterministic by seed ----------------
_MOCK_REPLIES = (
    "老师，我觉得这可能是视网膜静脉阻塞，眼底有片状出血，但我不是很确定，能给我一点提示吗？",
    "老师，我大概能说出来：糖尿病视网膜病变分期主要看微血管瘤、出血和新生血管，但具体到这一期我有点犹豫。",
    "这道题我会一部分：眼压高不一定就是青光眼，还要结合视野和视盘改变，不过再往下的鉴别我就说不清了。",
    "老师，是不是只要出血点数量够多，就一定说明分期更严重？我总觉得数量是最重要的标准。",
    "我看到黄斑区有硬性渗出，是不是就可以诊断糖尿病黄斑水肿了？其他检查我还没想好怎么做。",
    "我理解了：要先看视盘杯盘比，再结合视野缺损来判断青光眼，老师请您再带我完整过一遍好吗？",
    "老师，这部分我掌握得不扎实，能不能先给我一个方向性的提示？我想自己再推理一次。",
    "我试着总结一下：这位患者的关键体征是视力下降伴眼压升高，鉴别要排除高眼压症和继发性青光眼，我总结得对吗？",
)
_MOCK_MISCONCEPTIONS = (
    "把出血点数量当分期唯一标准",
    "眼压高即诊断青光眼",
    "把棉绒斑当成硬性渗出",
    "认为视力好就可排除黄斑水肿",
)

#: English mock material (lang='en'): same RNG consumption order as zh, so "same seed -> same
#: output" determinism holds independently in each language.
_MOCK_REPLIES_EN = (
    "Dr., I think this might be retinal vein occlusion — there is a patch of hemorrhage in the "
    "fundus, but I'm not quite sure. Could you give me a hint?",
    "Dr., I can roughly say it: DR staging mainly depends on microaneurysms, hemorrhages, and "
    "neovascularization, but I hesitate at picking the exact stage.",
    "I know part of this one: high IOP alone doesn't equal glaucoma — you also need the visual "
    "field and optic disc changes — but beyond that I can't tell the differentials apart.",
    "Dr., is it true that once the number of hemorrhage points is large enough, the stage must be "
    "more severe? I keep feeling the count is the decisive criterion.",
)
_MOCK_MISCONCEPTIONS_EN = (
    "treats hemorrhage-point count as the sole staging criterion",
    "diagnoses glaucoma from high IOP alone",
    "mistakes cotton-wool spots for hard exudates",
    "assumes good vision rules out macular edema",
)


def _mock_reply(cache_key, seed, lang="zh"):
    """Deterministically generate a schema-valid mock reply (rng = seed + cache_key, reproducible).

    lang='en' -> English reply template and misconception_shown examples (schema and value
    distributions unchanged).
    """
    rng = random.Random(f"{seed}:{cache_key}")
    delta = {
        "mastery": round(rng.uniform(-0.05, 0.15), 4),    # small step: in [-0.05, 0.15]
        "engagement": round(rng.uniform(-0.10, 0.10), 4),  # ∈ [-0.10, 0.10]
        "fatigue": round(rng.uniform(0.00, 0.08), 4),      # ∈ [0, 0.08]
        "misconception_resolved": bool(rng.random() < 0.2),
    }
    if _norm_lang(lang) == "en":
        shown = rng.choice(_MOCK_MISCONCEPTIONS_EN) if rng.random() < 0.3 else ""
        return {"reply": rng.choice(_MOCK_REPLIES_EN), "state_delta": delta,
                "misconception_shown": shown}
    shown = rng.choice(_MOCK_MISCONCEPTIONS) if rng.random() < 0.3 else ""
    return {"reply": rng.choice(_MOCK_REPLIES), "state_delta": delta, "misconception_shown": shown}


# ---------------- Public API ----------------
def student_chat(messages, seed=0, temperature=0.9, lang="zh"):
    """One virtual-student call with cache / rate-limit / retry / degradation. Never aborts training.

    - mock mode (CERES_STUDENT_MOCK=1 or no key): offline deterministic generation;
    - real mode: OpenAI-compatible client + response_format=json_object + 3 backoff retries;
    - lang (optional, default 'zh'): 'en' -> English system-prompt addon / mock template /
      degraded reply; the JSON schema is identical in both languages.
    - returns {"reply", "state_delta", "misconception_shown"}, plus "degraded": True on fallback.
    """
    lang = _norm_lang(lang)
    try:
        msgs = [m for m in (messages or []) if isinstance(m, dict)]
    except Exception:
        msgs = []
    # cache key gets a language suffix (payload and mock output vary by lang); zh keeps the
    # original key for full compat with existing calls/cache
    key = _cache_key(msgs, seed) + ("#en" if lang == "en" else "")
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    if is_mock_mode():
        out = _mock_reply(key, seed, lang=lang)
        _cache_put(key, out)
        return out

    # ---- Real API path ----
    system = SYSTEM_PROMPT if lang != "en" else SYSTEM_PROMPT + "\n" + SYSTEM_PROMPT_EN_ADDON
    payload = [{"role": "system", "content": system}]
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if role and content is not None:
            payload.append({"role": str(role), "content": str(content)})
    seed_i = _to_float(seed, 0.0)
    seed_i = int(seed_i)
    temp = _to_float(temperature, 0.9)
    timeout = _env_float("CERES_STUDENT_TIMEOUT", DEFAULT_TIMEOUT)
    model = _env("CERES_STUDENT_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    # DeepSeek thinking mode (default disabled): student replies are short JSON, so thinking is
    # wasteful (~20s per call, dragging multi-round rollout); thinking tokens also crowd out
    # output causing empty content. CERES_STUDENT_THINKING=enabled re-enables it.
    thinking = _env("CERES_STUDENT_THINKING", "disabled").strip().lower()
    extra = ({"extra_body": {"thinking": {"type": thinking}}}
             if thinking in ("enabled", "disabled") else {})
    sem = _get_sem()
    for attempt in range(RETRY_TIMES):
        acquired = False
        try:
            sem.acquire()
            acquired = True
            resp = _get_client().chat.completions.create(
                model=model, messages=payload, temperature=temp, seed=seed_i, timeout=timeout,
                response_format={"type": "json_object"}, **extra)
            content = resp.choices[0].message.content
            out = _sanitize(json.loads(content))  # parse/sanitize failure -> this round's except -> retry/degrade
            _cache_put(key, out)
            return out
        except Exception:
            if attempt < RETRY_TIMES - 1:
                time.sleep(_BACKOFF_BASE * (attempt + 1))  # backoff: 1.5s / 3.0s
        finally:
            if acquired:
                sem.release()
    # degraded: neutral reply, no state change (not cached, so transient failures aren't frozen); English rows get the English variant
    deg = _DEGRADED_EN if lang == "en" else _DEGRADED
    return {"reply": deg["reply"],
            "state_delta": dict(deg["state_delta"]),
            "misconception_shown": deg["misconception_shown"],
            "degraded": True}


def init_state(dd: dict) -> dict:
    """Build the initial learner state from a dataset row (design §7.2 + None/empty fallback)."""
    dd = dd if isinstance(dd, dict) else {}
    mis = dd.get("misconception_seed", "none")
    if mis is None or (isinstance(mis, str) and not mis.strip()):
        mis = "none"
    return {"mastery": 0.30, "engagement": 0.70, "fatigue": 0.10,
            "misconception": str(mis),
            "misconception_severity": 0.80, "turn": 0}


def apply_delta(state: dict, delta: dict) -> dict:
    """Apply state_delta in place and return state; mastery/engagement/fatigue clamped to [0,1].

    Defensive: invalid numeric values in delta are ignored (never raises); misconception_resolved
    -> severity -0.5 (floor 0); misconception_shown -> severity +0.1 (cap 1).
    """
    state = state if isinstance(state, dict) else {}
    delta = delta if isinstance(delta, dict) else {}
    for k in ("mastery", "engagement", "fatigue"):
        if k in delta:
            v = _to_float(delta[k], None)
            if v is None:
                continue
            state[k] = float(min(1.0, max(0.0, _to_float(state.get(k), 0.0) + v)))
    if delta.get("misconception_resolved"):
        state["misconception_severity"] = float(
            max(0.0, _to_float(state.get("misconception_severity"), 1.0) - 0.5))
    if delta.get("misconception_shown"):
        state["misconception_severity"] = float(
            min(1.0, _to_float(state.get("misconception_severity"), 1.0) + 0.1))
    return state
