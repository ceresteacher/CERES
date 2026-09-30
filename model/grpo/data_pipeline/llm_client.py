# -*- coding: utf-8 -*-
"""Unified LLM client for the pipeline (OpenAI-compatible) + shared small utilities.

Provides: OpenAI-compatible client via env (CERES_API_BASE/KEY/TEACHER_MODEL/SYNTH_*); chat() with
cache + deterministic offline mock + 3-retry backoff (returns "" on total failure); chat_many() with
thread pool + semaphore; disk cache keyed by sha1(messages+seed+model[+json_mode]). Also hosts the
pipeline's shared utilities (env_*/read_jsonl/write_jsonl/load_json/dump_json/persona_seed_from_qid/
parse_json_dict/detect_lang) so sibling modules import only this module.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import threading
import time

__all__ = [
    "DEFAULT_API_BASE", "DEFAULT_MODEL", "DEFAULT_CONCURRENCY", "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_TOKENS",
    "is_mock_mode", "cache_key", "chat", "chat_many", "detect_lang",
    "env_str", "env_int", "env_float", "read_jsonl", "write_jsonl",
    "load_json", "dump_json", "persona_seed_from_qid", "parse_json_dict",
    "main",
]

# ---------------------------------------------------------------- Constants
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-max"
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 1024
RETRY_TIMES = 3
_BACKOFF_BASE = 1.5            # backoff: 1.5s / 3.0s (same as student_sim)
_MEM_CACHE_MAX = 100_000       # in-memory cache soft cap (disk cache still there; clear only loses speed)


# ---------------------------------------------------------------- Env helpers (read at call time, monkeypatch-friendly)
def env_str(name, default=""):
    try:
        v = os.environ.get(name, default)
        return v if isinstance(v, str) else str(v)
    except Exception:
        return default


def env_int(name, default):
    try:
        return int(float(env_str(name, str(default)).strip()))
    except Exception:
        return default


def env_float(name, default):
    try:
        return float(env_str(name, str(default)).strip())
    except Exception:
        return default


def is_mock_mode(mock=False):
    """mock decision: explicit mock=True, or CERES_API_KEY unset (can't reach network, must go offline)."""
    if mock:
        return True
    return env_str("CERES_API_KEY", "").strip() == ""


# ---------------------------------------------------------------- jsonl / json IO (pipeline shared utilities)
def read_jsonl(path):
    """Read jsonl line by line -> list[dict]. Missing file / bad lines: skip with warning, never raise."""
    rows = []
    if not path or not isinstance(path, str):
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    sys.stderr.write("[data_pipeline] 跳过无法解析的行：%s\n" % ln[:80])
    except Exception as e:
        sys.stderr.write("[data_pipeline] 读 jsonl 失败 %s：%s\n" % (path, e))
    return rows


def write_jsonl(path, rows):
    """Write jsonl (overwrite + auto-mkdir). Per-line serialize failure falls back to placeholder, never raises."""
    if not path or not isinstance(path, str):
        sys.stderr.write("[data_pipeline] write_jsonl 收到非法路径：%r\n" % (path,))
        return
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                try:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                except Exception:
                    f.write(json.dumps({"_unserializable_row": True}, ensure_ascii=False) + "\n")
    except Exception as e:
        sys.stderr.write("[data_pipeline] 写 jsonl 失败 %s：%s\n" % (path, e))


def load_json(path, default=None):
    """Read a single json file, return default on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def dump_json(path, obj):
    """Write a single json file (indent=2, ensure_ascii=False); warn on failure, no raise."""
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        sys.stderr.write("[data_pipeline] 写 json 失败 %s：%s\n" % (path, e))
        return False


def persona_seed_from_qid(qid):
    """Deterministic persona_seed: sha1(qid) first 8 hex digits -> int in [0, 100000).

    Compatible with design §7.3's int(persona_seed) usage (real student_chat path truncates the seed
    via int(); this function already yields an int). qid missing -> 0.
    """
    try:
        blob = str(qid if qid is not None else "").encode("utf-8", "ignore")
        return int(hashlib.sha1(blob).hexdigest()[:8], 16) % 100000
    except Exception:
        return 0


def parse_json_dict(text):
    """Lenient JSON object parse: strip ```json fences, slice first/last braces; None on failure."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")  # strip fences (including ```json)
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------- Language detection (trajectory lang follows source data)
#: CJK ideograph ranges (incl. Extension A): enough for Chinese question banks, not exhaustive
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # Extension A
    (0xF900, 0xFAFF),    # Compatibility Ideographs
    (0x3000, 0x303F),    # CJK punctuation (，。、「」 etc.)
)
_CJK_LANG_THRESHOLD = 0.25   # CJK ratio >= 0.25 -> zh (English stems with occasional CJK drug names won't flip)


def _is_cjk_char(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def detect_lang(text):
    """Text language detection -> 'zh' | 'en' (unified entry for "trajectory lang follows source lang").

    Rule: CJK chars (incl. CJK punctuation) over non-whitespace chars >= 0.25 -> 'zh', else 'en'.
    Empty / None / non-string -> **'zh'** (project default; callers explicitly override text-less
    image sources instead of relying on this).
    """
    if not isinstance(text, str):
        return "zh"
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return "zh"
    n_cjk = sum(1 for c in chars if _is_cjk_char(c))
    return "zh" if (n_cjk / float(len(chars))) >= _CJK_LANG_THRESHOLD else "en"


# ---------------------------------------------------------------- Message normalization / cache
def _norm_messages(messages):
    """Normalize any input into [{"role","content"}]: tolerate dict/None/multimodal content, drop invalid."""
    out = []
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, (list, tuple)):
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        c = m.get("content")
        if isinstance(c, str):
            content = c
        elif isinstance(c, (list, tuple)):
            parts = []
            for p in c:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            content = "\n".join([p for p in parts if p])
        elif isinstance(c, dict):
            content = str(c.get("text") or c.get("content") or "")
        elif c is None:
            content = ""
        else:
            content = str(c)
        out.append({"role": role, "content": content})
    return out


def cache_key(messages, seed, model, json_mode=False):
    """Cache key = sha1(messages + seed + model [+ json_mode]) (brief contract + mode isolation)."""
    try:
        blob = json.dumps(list(messages), ensure_ascii=False, sort_keys=True)
    except Exception:
        blob = repr(messages)
    raw = "%s#%s#%s#%s" % (blob, seed, model, "json" if json_mode else "text")
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


_MEM_CACHE, _MEM_LOCK = {}, threading.Lock()


def _cache_dir():
    return env_str("CERES_SYNTH_CACHE_DIR", "./.llm_cache") or "./.llm_cache"


def _cache_file(key):
    return os.path.join(_cache_dir(), key + ".json")


def _cache_get(key):
    """Memory -> disk two-level cache read; any failure treated as a miss."""
    with _MEM_LOCK:
        if key in _MEM_CACHE:
            return _MEM_CACHE[key]
    try:
        with open(_cache_file(key), "r", encoding="utf-8") as f:
            obj = json.load(f)
        text = obj.get("text") if isinstance(obj, dict) else None
        if isinstance(text, str) and text:
            with _MEM_LOCK:
                _MEM_CACHE[key] = text
            return text
    except Exception:
        pass
    return None


def _cache_put(key, text, meta=None):
    """Write memory + disk cache; disk failure only warns (cache outage never affects correctness)."""
    with _MEM_LOCK:
        if len(_MEM_CACHE) >= _MEM_CACHE_MAX:
            _MEM_CACHE.clear()
        _MEM_CACHE[key] = text
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        with open(_cache_file(key), "w", encoding="utf-8") as f:
            json.dump({"key": key, "text": text, "meta": meta or {}},
                      f, ensure_ascii=False, default=str)
    except Exception as e:
        sys.stderr.write("[llm_client] 写缓存失败：%s\n" % e)


# ---------------------------------------------------------------- Real API path
_CLIENT, _CLIENT_LOCK = None, threading.Lock()
_SEM, _SEM_LOCK = None, threading.Lock()


def _get_sem():
    global _SEM
    if _SEM is None:
        with _SEM_LOCK:
            if _SEM is None:
                _SEM = threading.Semaphore(
                    max(1, env_int("CERES_SYNTH_CONCURRENCY", DEFAULT_CONCURRENCY)))
    return _SEM


def _get_client():
    """Lazy OpenAI-compatible client; openai missing / create failure raises (caller degrades)."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        from openai import OpenAI  # lazy import: module still importable under mock / without openai
        _CLIENT = OpenAI(
            base_url=env_str("CERES_API_BASE", DEFAULT_API_BASE) or DEFAULT_API_BASE,
            api_key=env_str("CERES_API_KEY", ""),
            timeout=env_float("CERES_SYNTH_TIMEOUT", DEFAULT_TIMEOUT))
        return _CLIENT


def _remote_chat(msgs, seed, json_mode, temperature, model):
    """Real API call: semaphore + 3 backoff retries; all failed -> "".

    max_tokens defaults to 1024 (CERES_SYNTH_MAX_TOKENS overrides; <=0 means omit, server default).
    DeepSeek thinking (optional): CERES_API_THINKING=enabled|disabled -> extra_body
    {"thinking":{"type":...}}; CERES_API_REASONING_EFFORT=high|medium|low -> reasoning_effort.
    Both default unset (zero impact on DashScope/BigModel etc.).
    """
    sem = _get_sem()
    timeout = env_float("CERES_SYNTH_TIMEOUT", DEFAULT_TIMEOUT)
    max_tokens = env_int("CERES_SYNTH_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    thinking = env_str("CERES_API_THINKING", "").strip().lower()
    effort = env_str("CERES_API_REASONING_EFFORT", "").strip().lower()
    last_err = None
    for attempt in range(RETRY_TIMES):
        acquired = False
        try:
            sem.acquire()
            acquired = True
            kwargs = {"model": model, "messages": msgs,
                      "temperature": float(temperature if isinstance(temperature, (int, float)) else 0.7),
                      "seed": int(seed), "timeout": float(timeout)}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if max_tokens and max_tokens > 0:
                kwargs["max_tokens"] = int(max_tokens)
            if thinking in ("enabled", "disabled"):
                kwargs["extra_body"] = {"thinking": {"type": thinking}}
            if effort in ("high", "medium", "low"):
                kwargs["reasoning_effort"] = effort
            resp = _get_client().chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            content = content.strip()
            return content if content else ""  # empty reply retried as failure
        except Exception as e:
            last_err = e
            if attempt < RETRY_TIMES - 1:
                time.sleep(_BACKOFF_BASE * (attempt + 1))
        finally:
            if acquired:
                sem.release()
    sys.stderr.write("[llm_client] API 调用失败（已重试 %d 次）：%s\n" % (RETRY_TIMES, last_err))
    return ""


# ---------------------------------------------------------------- mock generator (offline deterministic, style aligned with student_sim)
_MOCK_GENERIC = (
    "（mock 离线输出）这是一段确定性合成文本，用于在无 API key 的环境打通管线。",
    "（mock 离线输出）教师应先激活已有知识，再分级提示，最后让学生先说。",
    "（mock 离线输出）请结合课件要点与本页上下文继续完成任务。",
    "（mock 离线输出）统一测试阶段请配置 CERES_API_KEY 以获得真实合成内容。",
)

_MOCK_RECALL = (
    "还记得我们抓的两条主线吗？一条看有没有新生血管，一条看出血渗出的范围。",
    "先回顾一下这个部位的正常结构，再对照你看到的异常。",
)
_MOCK_HINT = (
    "先别急着下结论，在病灶所在区域找一找关键征象。",
    "注意病灶的位置、形态和数量级，按定性到定量的顺序想。",
)
_MOCK_EXPLAIN = (
    "这一步的关键征象要结合解剖位置来理解，注意与相近体征区分。",
    "判读依据要落到具体结构上，不能只凭单一征象下结论。",
)
_MOCK_CHECK = (
    "把你目前的判断依据说给我听听？",
    "请先描述一下你看到的最突出所见，再说你的倾向。",
)
_MOCK_CORRECT = "对照课件要点复述一遍判读依据，把「依据→结论」的链条说完整。"
_MOCK_ENCOURAGE = "这一轮你的思路已经完整了，继续按「定性→定位→定量→分期」的顺序练习。"

# ---- English versions (trajectory lang follows English source: same structured branches, lang only) ----
_MOCK_GENERIC_EN = (
    "(mock offline output) This is deterministic synthetic text to exercise the pipeline without an API key.",
    "(mock offline output) The teacher should first activate prior knowledge, then give leveled hints, "
    "and let the student speak first.",
    "(mock offline output) Please continue the task using the courseware key points and this page's context.",
    "(mock offline output) Set CERES_API_KEY during unified testing to obtain real synthesis.",
)
_MOCK_RECALL_EN = (
    "Remember the two threads we follow: one asks whether there is neovascularization, "
    "the other tracks the extent of hemorrhage and exudation.",
    "Let's first review the normal anatomy of this region, then compare it against what you see.",
)
_MOCK_HINT_EN = (
    "Hold off on the conclusion for a moment; look for the key sign in the region where the lesions sit.",
    "Note the location, morphology, and order of magnitude of the lesions, moving from qualitative to quantitative.",
)
_MOCK_EXPLAIN_EN = (
    "The key sign at this step must be read against its anatomical location; keep it distinct from similar findings.",
    "The reading rationale has to be anchored to concrete structures; never conclude from a single sign.",
)
_MOCK_CHECK_EN = (
    "Walk me through the rationale behind your current judgement?",
    "Please first describe the most striking finding you see, then state your impression.",
)
_MOCK_CORRECT_EN = ("Recite the reading rationale against the courseware key points, "
                    "and complete the chain from evidence to conclusion.")
_MOCK_ENCOURAGE_EN = ("Your reasoning is already complete this round; keep practicing in the order "
                      "qualitative, localization, quantification, staging.")


def _mock_lang(messages):
    """mock output lang: detect_lang on the first system message (else whole text). The teacher/rewrite
    system prompt is already selected per row lang, so detecting the prompt lang yields the target
    output lang (empty text -> default zh)."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    text = ""
    for m in msgs:
        if m.get("role") == "system":
            text = m.get("content") or ""
            break
    if not text:
        text = "\n".join(str(m.get("content") or "") for m in msgs)
    return detect_lang(text)


def _mock_chat(messages, seed, json_mode):
    """Deterministic mock generation (same style as student_sim._mock_reply: seeded by sha1(msg+seed)).

    Special branches:
      * teacher-synthesis mode (system contains TEACHER_SYSTEM_PROMPT(_EN)'s <recall>...</check>):
        produces a grammar-valid teacher turn; hint level = min(3, 1+prior assistant turns) so G3 stays
        non-decreasing; regular turns never contain <end/> -> reliably triggers turn-5 force-close;
      * force-close branch (text contains both 收束/wrap up and <end/>, i.e. the FORCE_CLOSE
        instruction): returns a <correct> (embedding gt after 正确要点：/Reference answer:) +
        <encourage> + <end/> closing turn;
      * lang follows source: detect_lang on the system prompt switches zh/en templates, so English-row
        mock trajectories are fully English (determinism unchanged: same input+seed -> same output);
      * json_mode: returns valid JSON (no business keys; upper layers take the degrade path — offline smoke).
    """
    msgs = list(messages or [])
    try:
        blob = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
    except Exception:
        blob = repr(msgs)
    key = hashlib.sha1((blob + "#%s" % seed).encode("utf-8", "ignore")).hexdigest()
    rng = random.Random("%s:%s" % (seed, key))

    all_text = "\n".join([m.get("content", "") for m in msgs if isinstance(m, dict)])
    en = _mock_lang(msgs) == "en"
    teacher_mode = ("<recall>" in all_text) and ("</check>" in all_text)
    if teacher_mode:
        n_asst = sum(1 for m in msgs
                     if isinstance(m, dict) and m.get("role") == "assistant")
        forced = ("<end/>" in all_text) and (
            ("收束" in all_text) or ("wrap up" in all_text.lower()))
        if forced:
            gt = ""
            for line in all_text.splitlines():
                for marker in ("正确要点：", "Reference answer:"):
                    if marker in line:
                        gt = line.split(marker, 1)[1].strip()[:80]
                        break
                if gt:
                    break
            if en:
                conclusion = gt if gt else "give the standard conclusion per the courseware key points"
                return ("<correct>Reference conclusion: %s. Please recite the chain from evidence "
                        "to conclusion.</correct>\n<encourage>%s</encourage>\n<end/>"
                        % (conclusion, _MOCK_ENCOURAGE_EN))
            conclusion = gt if gt else "按课件要点给出规范结论"
            return ("<correct>规范结论：%s。请对照「依据→结论」的链条复述一遍。</correct>\n"
                    "<encourage>%s</encourage>\n<end/>" % (conclusion, _MOCK_ENCOURAGE))
        if en:
            if n_asst <= 0:
                return ('<recall>%s</recall>\n<hint level="1">%s</hint>\n<check>%s</check>'
                        % (rng.choice(_MOCK_RECALL_EN), rng.choice(_MOCK_HINT_EN),
                           rng.choice(_MOCK_CHECK_EN)))
            level = min(3, 1 + n_asst)
            return ('<hint level="%d">%s</hint>\n<explain>%s</explain>\n<check>%s</check>'
                    % (level, rng.choice(_MOCK_HINT_EN), rng.choice(_MOCK_EXPLAIN_EN),
                       rng.choice(_MOCK_CHECK_EN)))
        if n_asst <= 0:
            return ('<recall>%s</recall>\n<hint level="1">%s</hint>\n<check>%s</check>'
                    % (rng.choice(_MOCK_RECALL), rng.choice(_MOCK_HINT), rng.choice(_MOCK_CHECK)))
        level = min(3, 1 + n_asst)
        return ('<hint level="%d">%s</hint>\n<explain>%s</explain>\n<check>%s</check>'
                % (level, rng.choice(_MOCK_HINT), rng.choice(_MOCK_EXPLAIN), rng.choice(_MOCK_CHECK)))

    text = rng.choice(_MOCK_GENERIC_EN if en else _MOCK_GENERIC)
    if json_mode:
        return json.dumps({"mock": True, "seed": seed, "text": text}, ensure_ascii=False)
    return text


# ---------------------------------------------------------------- Public interface
def chat(messages, seed=0, json_mode=False, mock=False, use_cache=True,
         temperature=0.7, model=None):
    """Single completion -> str (degrades to "" on failure, never raises).

    :param messages:  [{"role": "system"|"user"|"assistant", "content": str}, ...]
    :param seed:      sampling seed (int; passed to API; also feeds mock determinism)
    :param json_mode: True -> response_format=json_object (mock returns valid JSON)
    :param mock:      True forces offline; default auto-mocks when key is unset
    :param use_cache: False disables memory+disk cache (CLI --no-cache)
    """
    msgs = _norm_messages(messages)
    seed_i = _to_int(seed)
    model = (model or env_str("CERES_TEACHER_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL)
    key = cache_key(msgs, seed_i, model, json_mode)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached
    if mock or is_mock_mode():
        text = _mock_chat(msgs, seed_i, json_mode)
        if use_cache:
            _cache_put(key, text, {"mock": True, "model": model})
        return text
    text = _remote_chat(msgs, seed_i, json_mode, temperature, model)
    if text and use_cache:
        _cache_put(key, text, {"model": model})
    return text


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def chat_many(batch, seeds=None, json_mode=False, mock=False, use_cache=True,
              concurrency=None, temperature=0.7):
    """Batch concurrent calls -> list[str] in input order (single failure -> "").

    :param batch: list[messages] (each element is one chat's messages)
    :param seeds: seed list aligned with batch; None -> all 0; short lists padded
    :param concurrency: thread-pool cap; None -> env CERES_SYNTH_CONCURRENCY (default 8)
    """
    if not isinstance(batch, (list, tuple)):
        return []
    items = list(batch)
    if not items:
        return []
    seed_list = list(seeds) if isinstance(seeds, (list, tuple)) else []
    workers = max(1, int(concurrency or env_int("CERES_SYNTH_CONCURRENCY", DEFAULT_CONCURRENCY)))
    workers = min(workers, max(1, len(items)))
    results = [""] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for i, msgs in enumerate(items):
            futs[ex.submit(chat, msgs, seed_list[i] if i < len(seed_list) else 0,
                           json_mode, mock, use_cache, temperature)] = i
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # one failed item must not break the whole batch
                sys.stderr.write("[llm_client] chat_many 第 %d 项失败：%s\n" % (i, e))
                results[i] = ""
    return results


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.llm_client",
        description="离线数据构造管线的统一 LLM 客户端（mock/缓存/并发限流/重试）；"
                    "本 CLI 仅作单次调用自检，管线各步骤在各自模块内。")
    p.add_argument("--prompt", required=True, help="user 消息正文（必填）")
    p.add_argument("--system", default="", help="可选 system 消息正文")
    p.add_argument("--seed", type=int, default=0, help="采样种子（默认 0）")
    p.add_argument("--json-mode", action="store_true", help="要求 JSON 输出")
    p.add_argument("--mock", action="store_true",
                   help="强制离线 mock（默认：CERES_API_KEY 未设置时自动）")
    p.add_argument("--no-cache", action="store_true", help="关闭磁盘+内存缓存")
    p.add_argument("--model", default="", help="覆写 CERES_TEACHER_MODEL")
    p.add_argument("--concurrency", type=int, default=None, help="覆写并发上限（仅 chat_many 用）")
    p.add_argument("--timeout", type=float, default=None, help="覆写超时秒数")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.timeout is not None:
        os.environ["CERES_SYNTH_TIMEOUT"] = str(args.timeout)
    if args.concurrency is not None:
        os.environ["CERES_SYNTH_CONCURRENCY"] = str(args.concurrency)
    if args.model:
        os.environ["CERES_TEACHER_MODEL"] = args.model
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    out = chat(messages, seed=args.seed, json_mode=args.json_mode,
               mock=args.mock, use_cache=not args.no_cache)
    sys.stdout.write(out + "\n")
    sys.stderr.write("[llm_client] mock=%s cache=%s model=%s\n"
                     % (is_mock_mode(args.mock), "off" if args.no_cache else "on",
                        env_str("CERES_TEACHER_MODEL", DEFAULT_MODEL)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
