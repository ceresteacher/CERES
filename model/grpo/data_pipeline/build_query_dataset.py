# -*- coding: utf-8 -*-
"""Build the GRPO query set (contracts.md §6.1 schema + lang + three gt columns) from queries_candidates.

Rows are deduplicated by uid (later wins, with a warning); images are resolved to local relative paths
(key omitted when absent); difficulty/persona_seed are defensively normalized. Prints per-difficulty/
node/lang stats; --balance optionally downsamples per (node, difficulty) group (no forced resampling).
"""
import argparse
import os
import random
import sys

# Ensure project root is on sys.path (same as prepare_datasets)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_pipeline import llm_client as L  # noqa: E402
from data_pipeline.prompts import DIFFICULTY_LEVELS  # noqa: E402

__all__ = ["build_query_row", "build_query_rows", "build_stats", "balance_rows", "main"]

_QUERY_KEYS = ("uid", "messages", "courseware_context", "curriculum_node",
               "learner_profile", "misconception_seed", "difficulty", "persona_seed", "lang",
               "gt_label", "gt_answer", "gt_explanation")


def _norm_lang(lang):
    """Normalize lang: only 'en' stays 'en'; everything else (None/empty/'zh') -> 'zh'."""
    return "en" if isinstance(lang, str) and lang.strip().lower() == "en" else "zh"


def _to_local_relpath(p):
    """Absolute path -> relative to CWD; relative paths pass through unchanged."""
    s = str(p or "").strip()
    if not s:
        return ""
    if os.path.isabs(s):
        try:
            return os.path.relpath(s, os.getcwd())
        except Exception:
            return s
    return s


def build_query_row(row):
    """Map one candidate to a schema row (None when student_question is missing)."""
    row = row if isinstance(row, dict) else {}
    question = row.get("student_question")
    if not isinstance(question, str) or not question.strip():
        return None
    question = question.strip()
    images = row.get("images")
    if isinstance(images, str):
        images = [images]
    img_list = []
    if isinstance(images, (list, tuple)):
        img_list = [str(p) for p in images if p]
    if not img_list and row.get("image"):
        img_list = [str(row["image"])]
    if img_list and not question.startswith("<image>"):
        question = "<image>" + question          # align with images column (ms-swift VL placeholder)

    difficulty = row.get("difficulty")
    if difficulty not in DIFFICULTY_LEVELS:
        difficulty = "routine_clarification"
    try:
        persona_seed = int(row.get("persona_seed"))
    except (TypeError, ValueError):
        persona_seed = 0

    out = {
        "uid": str(row.get("qid") or row.get("uid") or ""),
        "messages": [{"role": "user", "content": question}],
        "courseware_context": str(row.get("courseware_context") or ""),
        "curriculum_node": str(row.get("curriculum_node") or "ophthalmology/general/ophthalmology_qa"),
        "learner_profile": str(row.get("learner_profile") or ""),
        "misconception_seed": str(row.get("misconception_seed") or ""),
        "difficulty": difficulty,
        "persona_seed": persona_seed,
        "lang": _norm_lang(row.get("lang")),
    }
    # gt columns (constant output, passthrough for swift extra columns; current rewards don't consume
    # them, kept for evaluation / later outcome reward): with gt_letter (MCQ source) -> gt_label=letter,
    # gt_answer=option text; image/legacy rows (no gt_letter) -> gt_label keeps raw label, gt_answer empty.
    gt_letter = str(row.get("gt_letter") or "").strip().upper()
    gt_text = str(row.get("gt_label") or "").strip()
    out["gt_label"] = gt_letter or gt_text
    out["gt_answer"] = gt_text if gt_letter else ""
    out["gt_explanation"] = str(row.get("gt_explanation") or "").strip()
    rel = [_to_local_relpath(p) for p in img_list]
    rel = [p for p in rel if p]
    if rel:
        out["images"] = rel                      # omit images key when no image (contract)
    return out


def build_query_rows(rows):
    """Build rows in bulk with uid dedup (later rows overwrite, with a warning)."""
    out, seen = [], {}
    for r in rows or []:
        q = build_query_row(r)
        if q is None:
            sys.stderr.write("[build] 跳过缺 student_question 的行：%r\n"
                             % str(r.get("qid") if isinstance(r, dict) else r)[:60])
            continue
        uid = q["uid"]
        if not uid:
            q["uid"] = uid = "q-%06d" % (len(out) + 1)
        if uid in seen:
            sys.stderr.write("[build] uid 重复，后到覆盖：%s\n" % uid)
            out[seen[uid]] = q
        else:
            seen[uid] = len(out)
            out.append(q)
    return out


def build_stats(rows):
    """Per-difficulty/node/lang/image/gt-answer stats -> print-friendly dict."""
    rows = rows or []
    by_difficulty, by_node, by_lang = {}, {}, {}
    n_img = 0
    n_gt_answer = 0
    for r in rows:
        d = str(r.get("difficulty") or "unknown")
        n = str(r.get("curriculum_node") or "unknown")
        by_difficulty[d] = by_difficulty.get(d, 0) + 1
        by_node[n] = by_node.get(n, 0) + 1
        lg = _norm_lang(r.get("lang"))
        by_lang[lg] = by_lang.get(lg, 0) + 1
        if r.get("images"):
            n_img += 1
        if str(r.get("gt_answer") or "").strip():
            n_gt_answer += 1
    total = len(rows)
    return {"total": total,
            "by_difficulty": dict(sorted(by_difficulty.items())),
            "by_curriculum_node": dict(sorted(by_node.items(), key=lambda kv: -kv[1])),
            "by_lang": dict(sorted(by_lang.items())),
            "image_ratio": round(n_img / float(total), 4) if total else 0.0,
            "gt_answer_coverage": round(n_gt_answer / float(total), 4) if total else 0.0}


def balance_rows(rows, cap=800, seed=0):
    """Optionally downsample each (curriculum_node, difficulty) group to cap rows (deterministic).

    No upsampling/synthesis (design only suggests a ratio); cap<=0 returns rows unchanged.
    """
    rows = list(rows or [])
    if cap <= 0 or not rows:
        return rows
    rng = random.Random(seed)
    groups = {}
    for i, r in enumerate(rows):
        key = (str(r.get("curriculum_node") or ""), str(r.get("difficulty") or ""))
        groups.setdefault(key, []).append(i)
    keep_idx = set()
    for key, idxs in groups.items():
        if len(idxs) <= cap:
            keep_idx.update(idxs)
        else:
            keep_idx.update(sorted(rng.sample(idxs, cap)))
    return [rows[i] for i in sorted(keep_idx)]


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.build_query_dataset",
        description="构建 GRPO 查询集 ceres_oph_queries.jsonl（契约 schema）+ 难度/节点配比统计")
    p.add_argument("--input", default="data/queries_candidates.jsonl", help="candidates 路径")
    p.add_argument("--output", default="data/ceres_oph_queries.jsonl", help="查询集输出路径")
    p.add_argument("--balance", action="store_true",
                   help="开启组内降采样（默认只打印统计，不做强制重采样）")
    p.add_argument("--balance-cap", type=int, default=800,
                   help="每个 (curriculum_node, difficulty) 组的上限（默认 800）")
    p.add_argument("--limit", type=int, default=0, help="最多处理条数（0=不限）")
    p.add_argument("--mock", action="store_true", help="保留位（本步骤不调 LLM）")
    p.add_argument("--seed", type=int, default=0, help="balance 降采样的随机种子")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    rows = L.read_jsonl(args.input)
    if not rows:
        sys.stderr.write("[build] 输入为空：%s（先运行 rewrite_question）\n" % args.input)
        return 2
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    queries = build_query_rows(rows)
    stats_before = build_stats(queries)
    if args.balance:
        queries = balance_rows(queries, args.balance_cap, args.seed)
    stats = build_stats(queries)
    L.write_jsonl(args.output, queries)

    print("[build] queries=%s total=%d image_ratio=%.1f%% lang=%s gt_answer_coverage=%.1f%%"
          % (args.output, stats["total"], stats["image_ratio"] * 100,
             " ".join("%s:%d" % kv for kv in sorted(stats["by_lang"].items())),
             stats["gt_answer_coverage"] * 100))
    if args.balance:
        print("    balance: %d -> %d (cap=%d per (node,difficulty) group)"
              % (stats_before["total"], stats["total"], args.balance_cap))
    print("    难度分布：")
    for d, n in stats["by_difficulty"].items():
        print("        %-26s %6d (%.1f%%)" % (d, n, 100.0 * n / max(1, stats["total"])))
    print("    课程节点分布（top 20）：")
    for i, (node, n) in enumerate(stats["by_curriculum_node"].items()):
        if i >= 20:
            print("        ... 共 %d 个节点" % len(stats["by_curriculum_node"]))
            break
        print("        %-58s %6d" % (node, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
