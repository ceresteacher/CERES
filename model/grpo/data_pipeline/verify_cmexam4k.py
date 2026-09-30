# -*- coding: utf-8 -*-
"""Stats and sanity checks for the dual-format dataset (quality gate).

Generic over CMExam 4k (zh) and PubMedQA 1k (en) dual-format sets (--lang switches the gauge). Hard
checks (any failure -> exit 1, per-item [verify] FAIL): both files' row count == --expect and unique
uids; queries rows have single A-E gt_label, non-empty gt_answer/gt_explanation, no option trace in
messages[0], valid curriculum_node and lang; qa rows have the three gt columns, >=2 parseable option
letters with gt_label among them. Soft metrics: difficulty/node/lang distributions, eye/general ratio,
rewrote/degraded rates, qa gt_answer text match ratio. Report written to --report as JSON.
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

from data_pipeline import llm_client as L  # noqa: E402
from data_pipeline import extract_questions as EX  # noqa: E402
from data_pipeline import build_qa_dataset as QA  # noqa: E402

__all__ = ["verify_files", "main"]

_OPT_LINE_TEXT_RE = re.compile(r"(?m)^\s*([A-Ja-j])\s*[.、:：)）]\s*(\S.*)$")

#: option-trace regex (per lang): zh keeps original behavior; en line-start anchored ("vitamin D." is not a trace)
_OPT_TRACE_RE_ZH = re.compile(r"(?:^|\n|\s)[A-E][.、:：)）]|下列哪|以下哪")
_OPT_TRACE_RE_EN = re.compile(
    r"(?m)^\s*[A-E]\s*[.、:：)）]\s*\S|which\s+of\s+the\s+following", re.IGNORECASE)


def _content(row):
    """row -> messages[0].content ('' when key missing / bad structure)."""
    msgs = row.get("messages") if isinstance(row, dict) else None
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        return str(msgs[0].get("content") or "")
    return ""


def _check_gt(row, tag, fails, expect_letters=True):
    """gt three-column hard check (collected into fails)."""
    gt = str(row.get("gt_label") or "")
    if expect_letters and not re.fullmatch(r"[A-E]", gt):
        fails.append("%s uid=%s gt_label 非单字母 A–E：%r" % (tag, row.get("uid"), gt[:40]))
    if not str(row.get("gt_answer") or "").strip():
        fails.append("%s uid=%s gt_answer 为空" % (tag, row.get("uid")))
    if not str(row.get("gt_explanation") or "").strip():
        fails.append("%s uid=%s gt_explanation 为空" % (tag, row.get("uid")))


def _dist(rows, key):
    out = {}
    for r in rows or []:
        k = str(r.get(key) or "unknown")
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def verify_files(queries, qa, cand_open, cand_mcq, expect=0, lang="zh"):
    """Four row lists -> (fails: list[str], report: dict). lang='en' switches the English gauge (option-
    trace regex and lang hard check use the expected lang; rows missing lang default zh, so in en mode
    they correctly fail)."""
    fails = []
    report = {}
    expect_lang = "en" if str(lang or "").strip().lower() == "en" else "zh"
    opt_re = _OPT_TRACE_RE_EN if expect_lang == "en" else _OPT_TRACE_RE_ZH

    for tag, rows in (("queries", queries), ("qa", qa)):
        if expect and len(rows) != expect:
            fails.append("%s 行数 %d != expect %d" % (tag, len(rows), expect))
        uids = [str(r.get("uid") or "") for r in rows]
        if len(set(uids)) != len(uids):
            fails.append("%s uid 不唯一：%d/%d" % (tag, len(set(uids)), len(uids)))
        report["%s_total" % tag] = len(rows)

    # ---- queries file: gt complete + no option trace + node/lang valid
    n_trace = 0
    for r in queries:
        _check_gt(r, "queries", fails)
        if opt_re.search(_content(r)):
            n_trace += 1
            if n_trace <= 3:
                fails.append("queries uid=%s 提问含选项痕迹：%r" % (r.get("uid"), _content(r)[:60]))
        node = str(r.get("curriculum_node") or "")
        if not (node.startswith("ophthalmology/") or node == EX.GENERAL_NODE):
            fails.append("queries uid=%s curriculum_node 非法：%r" % (r.get("uid"), node))
        if str(r.get("lang") or "zh") != expect_lang:
            fails.append("queries uid=%s lang 非 %s：%r" % (r.get("uid"), expect_lang, r.get("lang")))
    report["queries_option_trace"] = n_trace
    report["queries_by_difficulty"] = _dist(queries, "difficulty")
    report["queries_by_node"] = _dist(queries, "curriculum_node")
    report["queries_by_lang"] = _dist(queries, "lang")
    n_eye = sum(1 for r in queries if str(r.get("curriculum_node") or "").startswith("ophthalmology/"))
    report["queries_eye_vs_general"] = {"eye": n_eye, "general": len(queries) - n_eye}

    # ---- qa file: gt complete + self-contained options and answer letter among them
    n_match = 0
    for r in qa:
        _check_gt(r, "qa", fails)
        letters = QA._parse_option_letters(_content(r))
        if len(letters) < 2:
            fails.append("qa uid=%s 正文选项字母不足 2 个：%r" % (r.get("uid"), _content(r)[:60]))
        elif str(r.get("gt_label") or "")[:1] not in letters:
            fails.append("qa uid=%s gt_label %r 不在正文选项 %s 中"
                         % (r.get("uid"), r.get("gt_label"), sorted(letters)))
        else:
            n_match += 1
    report["qa_gt_in_options"] = n_match

    # ---- soft metric: gt option-line text vs gt_answer match ratio (LLM copy fidelity)
    n_text_match = 0
    for r in qa:
        gt_letter = str(r.get("gt_label") or "")[:1]
        gt_answer = str(r.get("gt_answer") or "").strip()
        for m in _OPT_LINE_TEXT_RE.finditer(_content(r)):
            if m.group(1).upper() == gt_letter and m.group(2).strip() == gt_answer:
                n_text_match += 1
                break
    report["qa_gt_answer_text_match_ratio"] = round(
        n_text_match / float(len(qa)), 4) if qa else 0.0

    # ---- soft metric: two-round rewrite rewrote/degraded
    for tag, rows in (("open", cand_open), ("mcq", cand_mcq)):
        n_ok = sum(1 for r in rows if r.get("rewrote"))
        report["rewrite_%s" % tag] = {"total": len(rows), "rewrote": n_ok,
                                      "degraded": len(rows) - n_ok}
    return fails, report


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.verify_cmexam4k",
        description="4k 双格式 GRPO 数据集统计与健全性校验（硬校验失败 exit 1）")
    p.add_argument("--queries", default="data/ceres_cmexam4k_queries.jsonl", help="契约查询集路径")
    p.add_argument("--qa", default="data/cmexam4k_qa_grpo.jsonl", help="QA GRPO 集路径")
    p.add_argument("--candidates", default="data/queries_candidates_cmexam4k.jsonl",
                   help="open 模式 candidates（rewrote 率统计）")
    p.add_argument("--candidates-mcq", default="data/queries_candidates_cmexam4k_mcq.jsonl",
                   help="mcq 模式 candidates（rewrote 率统计）")
    p.add_argument("--expect", type=int, default=4000, help="期望行数（0=不检查）")
    p.add_argument("--lang", choices=["zh", "en"], default="zh",
                   help="行语种硬校验与选项痕迹正则口径（默认 zh=CMExam；PubMedQA 英文集用 en）")
    p.add_argument("--report", default="data/cmexam4k_report.json", help="报告 JSON 输出路径")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    queries = L.read_jsonl(args.queries)
    qa = L.read_jsonl(args.qa)
    cand_open = L.read_jsonl(args.candidates)
    cand_mcq = L.read_jsonl(args.candidates_mcq)
    fails, report = verify_files(queries, qa, cand_open, cand_mcq, expect=args.expect,
                                 lang=args.lang)
    if args.report:
        L.dump_json(args.report, report)

    eye_gen = report.get("queries_eye_vs_general", {})
    print("[verify] queries=%d qa=%d eye=%d general=%d lang_trace=%d"
          % (report.get("queries_total", 0), report.get("qa_total", 0),
             eye_gen.get("eye", 0), eye_gen.get("general", 0),
             report.get("queries_option_trace", 0)))
    print("    难度分布（queries）：%s" % report.get("queries_by_difficulty", {}))
    print("    课程节点分布（queries，top10）：%s"
          % dict(list(report.get("queries_by_node", {}).items())[:10]))
    for tag in ("open", "mcq"):
        rw = report.get("rewrite_%s" % tag, {})
        print("    rewrite[%s]: total=%d rewrote=%d degraded=%d"
              % (tag, rw.get("total", 0), rw.get("rewrote", 0), rw.get("degraded", 0)))
    print("    qa gt_answer 文本吻合率：%.1f%%"
          % (report.get("qa_gt_answer_text_match_ratio", 0.0) * 100))
    for f in fails:
        sys.stderr.write("[verify] FAIL %s\n" % f)
    if fails:
        sys.stderr.write("[verify] 共 %d 项硬校验失败，报告：%s\n" % (len(fails), args.report))
        return 1
    print("[verify] 全部硬校验通过，报告：%s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
