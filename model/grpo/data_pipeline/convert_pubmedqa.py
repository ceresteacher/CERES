# -*- coding: utf-8 -*-
"""Convert PubMedQA pqa_labeled.parquet to a unified jsonl (one-shot).

huggingface.co is unreachable locally, so the parquet is downloaded via hf-mirror and converted with
the venv python (the only interpreter with pyarrow); downstream steps (system python3) never touch
parquet. Hence pyarrow is lazily imported (safe to import/--help/test under system python3). Output is
1:1, no filtering/truncation (those happen in prepare_datasets.load_pubmedqa); each line:
{"pubid": int, "question": str, "context_text": str, "long_answer": str, "final_decision": yes|no|maybe}.
context_text = "\n\n".join("<LABEL>: <paragraph>"), empty paragraphs skipped, "CONTEXT" fallback.
"""
import argparse
import os
import sys

# Ensure project root is on sys.path (same as prepare_datasets)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_pipeline import llm_client as L  # noqa: E402

__all__ = ["flatten_record", "read_parquet_rows", "main"]


def _s(v):
    """Safe str: non-str -> ""; replaces U+2028/U+2029 (line separators json.dumps won't escape,
    present in PubMedQA data) with plain spaces."""
    if not isinstance(v, str):
        return ""
    return v.replace("\u2028", " ").replace("\u2029", " ").strip()


def flatten_record(item):
    """parquet row dict -> unified internal record (1:1, no filtering/truncation; all keys defaulted).

    Observed parquet shape: context is parallel lists inside a single dict (HF card says list[struct],
    to_pylist gives {"contexts": [...], "labels": [...], "meshes": [...]}); paragraphs pair with
    section names by index. Also tolerates list[dict] (per-segment) and plain string (forward-compat).
    """
    item = item if isinstance(item, dict) else {}
    pubid = item.get("pubid")
    try:
        pubid = int(pubid)
    except (TypeError, ValueError):
        pubid = -1
    question = _s(item.get("question"))

    # ---- context -> join "LABEL: paragraph"
    parts = []
    ctx = item.get("context")
    if isinstance(ctx, dict):
        # parallel-list shape (observed): contexts[i] pairs with labels[i]
        texts = ctx.get("contexts")
        labels = ctx.get("labels")
        texts = [t for t in texts if isinstance(t, str)] if isinstance(texts, (list, tuple)) else []
        labels = [l for l in labels if isinstance(l, str)] if isinstance(labels, (list, tuple)) else []
        for i, text in enumerate(texts):
            if text.strip():
                label = labels[i].strip() if i < len(labels) and labels[i].strip() else "CONTEXT"
                parts.append("%s: %s" % (label, text.strip()))
    elif isinstance(ctx, (list, tuple)):
        # per-segment shape (card spec, forward-compat): list[dict{contexts, labels}] / list[str]
        for seg in ctx:
            text = label = ""
            if isinstance(seg, dict):
                text, label = _s(seg.get("contexts")), _s(seg.get("labels"))
            elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                text, label = _s(seg[0]), _s(seg[1])
            elif isinstance(seg, str):
                text = seg.strip()
            if text:
                parts.append("%s: %s" % (label or "CONTEXT", text))
    context_text = "\n\n".join(parts)

    return {"pubid": pubid, "question": question, "context_text": context_text,
            "long_answer": _s(item.get("long_answer")),
            "final_decision": _s(item.get("final_decision"))}


def read_parquet_rows(path):
    """parquet -> list[raw row dict] (pyarrow lazily imported; missing package / bad file -> SystemExit)."""
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        sys.stderr.write("[pubmedqa-convert] parquet 不存在：%r\n" % (path,))
        raise SystemExit(2)
    try:
        import pyarrow.parquet as pq  # lazy import: only venv python has it
    except ImportError:
        sys.stderr.write("[pubmedqa-convert] 缺 pyarrow——请用 venv 解释器跑本命令：\n"
                         "  /hy-tmp/my-env/swift/bin/python -m data_pipeline.convert_pubmedqa ...\n")
        raise SystemExit(1)
    try:
        table = pq.read_table(path)
        rows = table.to_pylist()
    except Exception as e:                       # bad file / truncated download
        sys.stderr.write("[pubmedqa-convert] parquet 解析失败 %s：%s\n" % (path, e))
        raise SystemExit(1)
    return rows


# ---------------------------------------------------------------- CLI
def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_pipeline.convert_pubmedqa",
        description="PubMedQA pqa_labeled parquet → 统一 jsonl（1:1 无过滤；用 venv python 跑）")
    p.add_argument("--input", required=True, help="pqa_labeled parquet 路径")
    p.add_argument("--output", required=True, help="输出 jsonl 路径")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    rows = read_parquet_rows(args.input)
    records = [flatten_record(r) for r in rows if isinstance(r, dict)]
    L.write_jsonl(args.output, records)
    print("[pubmedqa-convert] rows=%d → %s" % (len(records), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
