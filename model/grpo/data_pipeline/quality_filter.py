# -*- coding: utf-8 -*-
"""Step 4: sft_raw -> data/ceres_oph_sft_warmup.jsonl + filter_report.json.

Filter rules (non-compliant rows dropped whole, never rewritten — avoid hidden medical errors):
grammar all-pass via ceres_plugin.grammar.validate_sft_trajectory (G1~G5, with per-row
curriculum_node/difficulty); teacher turns 2~5; last-turn teacher text must contain <end/>; gt_label
keyword must appear in the last <correct>/<explain>; per-turn char cap (zh 1200 / en 2400 by default).
Outputs the strict contract schema plus a report (total/kept/dropped/drop_reasons/dropped_rows detail/
label_unverified/per-difficulty-node-lang stats).
"""
import argparse
import os
import re
import sys

# Ensure project root is on sys.path (same as prepare_datasets)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ceres_plugin.grammar import parse_actions, validate_sft_trajectory  # noqa: E402
from data_pipeline import llm_client as L  # noqa: E402

__all__ = ["LABEL_KEYWORD_MAP", "label_keywords", "check_medical_point",
           "check_row", "run_filter", "main"]

# ---------------------------------------------------------------- zh/en label keyword map
#: English dataset label (lowercase, substring) -> Chinese check keywords (any hit passes). Covers
#: IDRiD/ODIR/OCTID/Kermany/PALM/REFUGE/GAMMA common labels; unmatched Chinese labels check as raw substring.
LABEL_KEYWORD_MAP = {
    # DR grades (IDRiD)
    "no dr": ["无明显", "无糖尿病视网膜病变", "正常", "未见"],
    "mild": ["轻度", "非增殖", "微血管瘤"],
    "moderate": ["中度", "非增殖"],
    "severe": ["重度", "非增殖"],
    "proliferative": ["增殖", "PDR", "新生血管"],
    "npdr": ["非增殖", "NPDR"],
    "pdr": ["增殖", "PDR", "新生血管"],
    "dr": ["糖尿病视网膜病变", "糖网", "视网膜病变"],
    # ODIR eight classes
    "normal": ["正常", "未见明显异常", "无异常"],
    "diabetes": ["糖尿病", "糖网"],
    "glaucoma": ["青光眼", "杯盘比", "视杯"],
    "cataract": ["白内障", "晶状体混浊", "晶体混浊"],
    "amd": ["年龄相关", "黄斑变性", "AMD", "玻璃膜疣", "黄斑新生血管"],
    "armd": ["年龄相关", "黄斑变性", "AMD"],
    "hypertension": ["高血压", "动脉硬化"],
    "myopia": ["近视"],
    "pathological myopia": ["病理性近视", "高度近视", "近视"],
    # OCT five classes (OCTID / Kermany)
    "drusen": ["玻璃膜疣", "drusen", "脉络膜小疣"],
    "cnv": ["新生血管", "CNV"],
    "dme": ["黄斑水肿", "水肿"],
    # others
    "disc": ["视盘", "视乳头"],
    "edema": ["水肿"],
}

#: long-label threshold: above it, character-overlap heuristic kicks in (long answer text rarely
#: reproduced verbatim)
_LONG_LABEL = 8
_OVERLAP_RATIO = 0.6

_G_RE = re.compile(r"^G([1-5])[:：]")


# ---------------------------------------------------------------- Medical point check
def label_keywords(gt_label):
    """gt_label -> check keyword list (raw text + the **longest-matched** mapping key's entries).

    Only the longest English key is taken, so short keys like "dr" don't pollute (e.g. drusen wrongly
    matched to the DR table).
    """
    label = str(gt_label or "").strip()
    if not label:
        return []
    low = label.lower()
    best = None
    for key in LABEL_KEYWORD_MAP:
        if key in low and (best is None or len(key) > len(best)):
            best = key
    kws = [label]
    if best:
        kws.extend(list(LABEL_KEYWORD_MAP[best]))
    return kws


def _as_messages_text(messages):
    """Safely get the message list (non-list -> [])."""
    if isinstance(messages, (list, tuple)):
        return list(messages)
    return []


def last_assistant_text(messages):
    """Last-turn teacher text ("" when not found)."""
    for m in reversed(_as_messages_text(messages)):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content")
            return c if isinstance(c, str) else ""
    return ""


def check_medical_point(gt_label, messages):
    """Whether the last <correct>/<explain> body covers the gt_label key point.

    :return: (ok, unverified) — unverified=True means label missing, cannot check (passed, counted in report).
    """
    label = str(gt_label or "").strip()
    if not label:
        return True, True
    # PubMedQA three-way answers (Yes/No/Maybe): the explanatory prose rarely contains these words
    # verbatim — substring check would only systematically false-drop (gt='No' would pass via 'not'),
    # so pass as "cannot verify" (counted in report's label_unverified, not dropped)
    if label.lower() in ("yes", "no", "maybe"):
        return True, True
    text = last_assistant_text(messages)
    if not text.strip():
        return False, False
    bodies = [body for tag, _attrs, body in parse_actions(text)
              if tag in ("correct", "explain")]
    target = "\n".join(bodies) if bodies else text
    if not target.strip():
        target = text
    low_target = target.lower()
    hit = False
    for kw in label_keywords(label):
        k = kw.lower().strip()
        if k and k in low_target:
            hit = True
            break
    if not hit and len(label) > _LONG_LABEL:
        chars = set(ch for ch in label
                    if not ch.isspace() and ch not in "()（）/、,，.。-—_")
        if chars:
            ratio = sum(1 for ch in chars if ch in target) / float(len(chars))
            hit = ratio >= _OVERLAP_RATIO
    return hit, False


# ---------------------------------------------------------------- Per-row check
def _group_reason(reason):
    """Group a grammar violation phrase into a report group key (original text kept verbatim in details)."""
    m = _G_RE.match(reason or "")
    if m:
        return "G" + m.group(1)
    if ("消息" in (reason or "")) or ("轨迹为空" in (reason or "")):
        return "消息结构"
    return "其他"


#: per-turn length cap (per trajectory lang: English ~2x Chinese char count; explicit int -> no split)
DEFAULT_MAX_TURN_CHARS = {"zh": 1200, "en": 2400}


def _turn_char_limit(row, max_turn_chars):
    """max_turn_chars None -> per-row-lang default; explicit int -> used as-is (CLI override)."""
    if isinstance(max_turn_chars, int):
        return max_turn_chars
    lang = row.get("lang") if isinstance(row, dict) else None
    return DEFAULT_MAX_TURN_CHARS.get(lang, DEFAULT_MAX_TURN_CHARS["zh"])


def check_row(row, min_turns=2, max_turns=5, max_turn_chars=None):
    """Single-row check -> (ok, groups:list[str], details:list[str], unverified:bool)."""
    row = row if isinstance(row, dict) else {}
    turn_limit = _turn_char_limit(row, max_turn_chars)
    groups, details = [], []
    messages = row.get("messages")
    msgs = _as_messages_text(messages)
    if not msgs:
        groups.append("结构非法")
        details.append("messages 为空或非列表")
        return False, groups, details, False

    n_asst = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")
    if n_asst < min_turns or n_asst > max_turns:
        groups.append("轮数越界")
        details.append("教师轮数 %d 不在 [%d, %d]" % (n_asst, min_turns, max_turns))

    last_text = last_assistant_text(msgs)
    if "<end/>" not in last_text:
        groups.append("未收束")
        details.append("末轮教师文本不含 <end/>（%s）"
                       % ("unresolved 标记" if row.get("unresolved") else "缺收束标签"))

    over = [i + 1 for i, m in enumerate(msgs)
            if isinstance(m, dict) and m.get("role") == "assistant"
            and isinstance(m.get("content"), str) and len(m["content"]) > turn_limit]
    if over:
        groups.append("单轮超长")
        details.append("第 %s 轮超过 %d 字" % ("、".join(str(x) for x in over), turn_limit))

    dd = {"curriculum_node": row.get("curriculum_node"), "difficulty": row.get("difficulty")}
    ok_g, reasons = validate_sft_trajectory(msgs, dd)
    if not ok_g:
        for r in reasons:
            groups.append(_group_reason(r))
            details.append(r)

    med_ok, unverified = check_medical_point(row.get("gt_label"), msgs)
    if not med_ok:
        groups.append("医学要点不符")
        details.append("末轮 <correct>/<explain> 未覆盖 gt_label=%r" % str(row.get("gt_label"))[:60])

    # dedup preserving order
    groups = list(dict.fromkeys(groups))
    return (not groups), groups, details, unverified


# ---------------------------------------------------------------- Batch filter
def run_filter(rows, min_turns=2, max_turns=5, max_turn_chars=None, max_examples=50):
    """sft_raw rows -> (kept contract rows, report dict).

    lang passthrough: report groups by lang (missing lang -> project default 'zh'); kept rows still
    output the strict contract schema (messages/images/uid, no lang — contract unchanged).
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    kept, dropped, drop_reasons, dropped_rows, unverified_uids = [], [], {}, [], []
    kept_by_difficulty, kept_by_node, kept_by_lang = {}, {}, {}
    for row in rows:
        uid = str(row.get("uid") or row.get("qid") or "")
        ok, groups, details, unverified = check_row(row, min_turns, max_turns, max_turn_chars)
        if unverified:
            unverified_uids.append(uid)
        if not ok:
            dropped.append(row)
            for g in groups:
                drop_reasons[g] = drop_reasons.get(g, 0) + 1
            if len(dropped_rows) < max_examples:
                dropped_rows.append({"uid": uid, "groups": groups, "reasons": details})
            continue
        msgs = _as_messages_text(row.get("messages"))
        clean = {"messages": msgs, "uid": uid}
        images = row.get("images")
        if isinstance(images, str):
            images = [images]
        if isinstance(images, (list, tuple)):
            imgs = [str(p) for p in images if p]
            if imgs:
                clean["images"] = imgs
        kept.append(clean)
        d = str(row.get("difficulty") or "unknown")
        n = str(row.get("curriculum_node") or "unknown")
        lg = str(row.get("lang") or "").strip().lower()
        lg = lg if lg in ("zh", "en") else "zh"
        kept_by_difficulty[d] = kept_by_difficulty.get(d, 0) + 1
        kept_by_node[n] = kept_by_node.get(n, 0) + 1
        kept_by_lang[lg] = kept_by_lang.get(lg, 0) + 1

    total = len(kept) + len(dropped)
    report = {
        "total": total,
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_rate": round(len(dropped) / float(total), 4) if total else 0.0,
        "drop_reasons": dict(sorted(drop_reasons.items(), key=lambda kv: -kv[1])),
        "dropped_rows": dropped_rows,
        "label_unverified": unverified_uids,
        "kept_by_difficulty": dict(sorted(kept_by_difficulty.items())),
        "kept_by_curriculum_node": dict(sorted(kept_by_node.items())),
        "kept_by_lang": dict(sorted(kept_by_lang.items())),
        "rule_semantics": "G1~G5 文案来自 ceres_plugin.grammar.validate_sft_trajectory"
                          "（每轮至少 1 个动作标签；<end/> 自闭合且只能出现在最后一轮；"
                          "六类内容标签不得自闭合；G5=阅片类先 <check> 后 explain/correct）",
        "note": "医学要点核对为关键词启发式（长标签走 %.0f%% 字符重叠），设计要求的人工/医师"
                "抽检 5~10%% 仍需执行" % (_OVERLAP_RATIO * 100),
    }
    return kept, report


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.quality_filter",
        description="Step 4：sft_raw → ceres_oph_sft_warmup.jsonl + filter_report.json（grammar 真实校验 + 医学要点核对）")
    p.add_argument("--input", default="data/sft_raw.jsonl", help="sft_raw 路径")
    p.add_argument("--output", default="data/ceres_oph_sft_warmup.jsonl", help="SFT 预热集输出路径")
    p.add_argument("--report", default="data/filter_report.json", help="过滤报告输出路径")
    p.add_argument("--min-turns", type=int, default=2, help="教师轮数下限（默认 2）")
    p.add_argument("--max-turns", type=int, default=5, help="教师轮数上限（默认 5）")
    p.add_argument("--max-turn-chars", type=int, default=None,
                   help="单轮教师文本长度上限（默认按语言分档：zh 1200 / en 2400；显式传值则不分档）")
    p.add_argument("--max-examples", type=int, default=50, help="report 中 dropped_rows 明细上限")
    p.add_argument("--limit", type=int, default=0, help="最多处理条数（0=不限）")
    p.add_argument("--mock", action="store_true", help="保留位（本步骤不调 LLM）")
    p.add_argument("--seed", type=int, default=0, help="保留位（本步骤无随机性）")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    rows = L.read_jsonl(args.input)
    if not rows:
        sys.stderr.write("[filter] 输入为空：%s（先运行 synthesize_dialogue）\n" % args.input)
        return 2
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    kept, report = run_filter(rows, args.min_turns, args.max_turns,
                              args.max_turn_chars, args.max_examples)
    L.write_jsonl(args.output, kept)
    L.dump_json(args.report, report)
    print("[filter] warmup=%s kept=%d/%d drop_rate=%.1f%% lang=%s" % (
        args.output, report["kept"], report["total"], report["drop_rate"] * 100,
        " ".join("%s:%d" % kv for kv in sorted(report["kept_by_lang"].items()))))
    for g, n in report["drop_reasons"].items():
        print("    drop_reason %-14s %d" % (g, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
