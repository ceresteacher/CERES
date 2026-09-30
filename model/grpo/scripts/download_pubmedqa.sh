#!/bin/bash
# scripts/download_pubmedqa.sh — one-shot download + convert PubMedQA pqa_labeled (idempotent).
#   Output: $PUBMEDQA_DIR/pqa_labeled.jsonl (1,000 rows, unified internal record format).
#   Idempotent: skips if the jsonl already exists and is non-empty (no network needed, incl. MOCK runs).
#   Downloads via hf-mirror (huggingface.co unreachable locally); curl -fL follows the 302 to CDN.
#   parquet->jsonl conversion needs pyarrow — only the venv python has it (VENV_PYTHON overridable).
# Alternative download (other machines / curl failure):
#   HF_ENDPOINT=https://hf-mirror.com $VENV_PYTHON -c \
#     "from huggingface_hub import hf_hub_download; \
#      print(hf_hub_download('qiaojin/PubMedQA','pqa_labeled/train-00000-of-00001.parquet',repo_type='dataset'))"
# Env vars: PUBMEDQA_DIR(/hy-tmp/dataset/PubMedQA) / VENV_PYTHON / HF_MIRROR
set -euo pipefail
cd "$(dirname "$0")/.."

PUBMEDQA_DIR=${PUBMEDQA_DIR:-/hy-tmp/dataset/PubMedQA}
VENV_PYTHON=${VENV_PYTHON:-/hy-tmp/my-env/swift/bin/python}   # only this script needs pyarrow
HF_MIRROR=${HF_MIRROR:-https://hf-mirror.com}
PARQUET_REL="pqa_labeled/train-00000-of-00001.parquet"
JSONL="$PUBMEDQA_DIR/pqa_labeled.jsonl"

if [ -s "$JSONL" ]; then
    echo "[pubmedqa-dl] 已存在，跳过：$JSONL（$(wc -l < "$JSONL") 行）"
    exit 0
fi

mkdir -p "$PUBMEDQA_DIR/pqa_labeled"
echo "[pubmedqa-dl] 下载 $HF_MIRROR/datasets/qiaojin/PubMedQA/resolve/main/$PARQUET_REL"
curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 \
     -o "$PUBMEDQA_DIR/$PARQUET_REL" \
     "$HF_MIRROR/datasets/qiaojin/PubMedQA/resolve/main/$PARQUET_REL"
echo "[pubmedqa-dl] parquet $(stat -c%s "$PUBMEDQA_DIR/$PARQUET_REL") 字节（上游基准 1075513，漂移仅告警级）"

"$VENV_PYTHON" -m data_pipeline.convert_pubmedqa \
    --input "$PUBMEDQA_DIR/$PARQUET_REL" --output "$JSONL"
echo "[pubmedqa-dl] 完成：$(wc -l < "$JSONL") 行（预期 1000）"
