# -*- coding: utf-8 -*-
"""Deterministic CMExam sampling: train.csv -> data/cmexam_4k_sample.csv.

Ophthalmology rows are too few for 4k (train hits DEFAULT_OPH_KEYWORDS only ~1,180, ~1,069 after
filtering), so "ophthalmology first + general fill" reaches the target: filter (empty stem/explanation,
bad options, multi-letter answers, dedup), keep all eye rows, fill from general rows via
Random(seed).sample, output in original CSV order. Deterministic for fixed (input, total, seed); only
train is used (val/test kept as benchmark).
"""
import argparse
import csv
import os
import random
import sys

# Ensure project root is on sys.path (same as prepare_datasets)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_pipeline import llm_client as L  # noqa: E402
from data_pipeline import prepare_datasets as P  # noqa: E402
from data_pipeline import extract_questions as EX  # noqa: E402

__all__ = ["filter_cmexam_rows", "sample_rows", "write_cmexam_csv", "main"]


def filter_cmexam_rows(rows, keywords=None):
    """csv dict rows -> (candidate list, rule-count dict).

    Candidate: {"idx": source row, "stem", "options": dict, "ans_letter", "explanation", "is_eye":
    stem hits ophthalmology keyword}. Filter order (per-rule counts): empty stem -> empty explanation ->
    options < 2 or answer letter absent -> non-single-letter answer -> stem dedup (first kept).
    """
    keywords = keywords if keywords else [k.strip() for k in EX.DEFAULT_OPH_KEYWORDS.split(",")
                                          if k.strip()]
    counts = {"total": 0, "empty_stem": 0, "empty_explanation": 0, "bad_options": 0,
              "multi_letter": 0, "dup_stem": 0, "kept": 0}
    cands, seen_stems = [], set()
    for idx, row in enumerate(rows or []):
        counts["total"] += 1
        stem = str(row.get("Question") or "").strip()
        explanation = str(row.get("Explanation") or "").strip()
        if not stem:
            counts["empty_stem"] += 1
            continue
        if not explanation:
            counts["empty_explanation"] += 1
            continue
        options = P.parse_options(row.get("Options"))
        ans_letter = P.norm_answer_letters(row.get("Answer"))
        if len(options) < 2 or ans_letter[:1] not in options:
            counts["bad_options"] += 1
            continue
        if len(ans_letter) != 1:
            counts["multi_letter"] += 1     # drop multi-answer items (grading unified to single letter)
            continue
        key = stem.lower()
        if key in seen_stems:
            counts["dup_stem"] += 1
            continue
        seen_stems.add(key)
        cands.append({"idx": idx, "stem": stem, "options": options,
                      "ans_letter": ans_letter, "explanation": explanation,
                      "is_eye": EX._keyword_hit(stem, keywords)})
        counts["kept"] += 1
    return cands, counts


def sample_rows(candidates, total, seed):
    """Keep all eye rows + fill to total from general rows via Random(seed).sample; sorted by source idx.

    If eye rows exceed total, deterministically sample from eye rows (general rows unused); total<=0
    returns [].
    """
    total = int(total or 0)
    candidates = list(candidates or [])
    if total <= 0 or not candidates:
        return []
    eye = [c for c in candidates if c.get("is_eye")]
    general = [c for c in candidates if not c.get("is_eye")]
    rng = random.Random(seed)
    if len(eye) >= total:
        chosen = rng.sample(eye, total)
    else:
        fill_n = min(total - len(eye), len(general))
        chosen = eye + rng.sample(general, fill_n)
    return sorted(chosen, key=lambda c: c.get("idx", 0))


def write_cmexam_csv(path, chosen):
    """selected rows -> CMExam 4-col csv (Options re-serialized as 'A text\nB text', Answer=single letter)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["Question", "Options", "Answer", "Explanation"])
        for c in chosen or []:
            opts_text = "\n".join("%s %s" % (letter, c["options"][letter])
                                  for letter in sorted(c["options"].keys()))
            w.writerow([c["stem"], opts_text, c["ans_letter"], c["explanation"]])


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.sample_cmexam",
        description="CMExam 确定性采样（眼科优先 + 全科补足）→ CMExam 格式 csv，"
                    "供 prepare_datasets 登记后走标准 extract→rewrite→build 管线")
    p.add_argument("--input", required=True, help="CMExam csv 路径（建议 train.csv，val/test 留作评测）")
    p.add_argument("--output", default="data/cmexam_4k_sample.csv", help="采样输出 csv 路径")
    p.add_argument("--total", type=int, default=4000, help="目标条数（默认 4000）")
    p.add_argument("--seed", type=int, default=42, help="全科补足采样种子（默认 42，确定性）")
    p.add_argument("--keyword", default=EX.DEFAULT_OPH_KEYWORDS,
                   help="眼科子集关键词（逗号分隔，覆写默认表；决定「眼科优先」的范围）")
    p.add_argument("--mock", action="store_true", help="保留位（本步骤不调 LLM）")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    rows = P._read_csv_rows(args.input)
    if not rows:
        sys.stderr.write("[sample] 输入为空：%s\n" % args.input)
        return 2
    keywords = [k.strip() for k in str(args.keyword).split(",") if k.strip()]
    cands, counts = filter_cmexam_rows(rows, keywords=keywords)
    for k in ("empty_stem", "empty_explanation", "bad_options", "multi_letter", "dup_stem"):
        print("    %-18s %6d" % (k + ":", counts[k]))
    n_eye = sum(1 for c in cands if c["is_eye"])
    n_general = len(cands) - n_eye
    chosen = sample_rows(cands, args.total, args.seed)
    write_cmexam_csv(args.output, chosen)
    n_chosen_eye = sum(1 for c in chosen if c["is_eye"])
    print("[sample] input=%s total=%d kept=%d (eye=%d general_pool=%d) "
          "seed=%d selected=%d (eye=%d fill=%d) output=%s"
          % (args.input, counts["total"], counts["kept"], n_eye, n_general,
             args.seed, len(chosen), n_chosen_eye, len(chosen) - n_chosen_eye, args.output))
    if len(chosen) < args.total:
        sys.stderr.write("[sample] 警告：候选不足，仅选得 %d/%d 条\n" % (len(chosen), args.total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
