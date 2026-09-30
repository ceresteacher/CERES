#!/bin/bash
# ============================================================================
# run_demo_dialogue.sh — training-effect demo: 8 questions x 4 teaching rounds (wraps eval/demo_dialogue.py).
#
# Purpose: same questions + same student simulator, only the teacher changes (base vs LoRA adapter),
#   to compare whether the teacher learned to emit the 7 action-tag DSL. Key metric: action-tag
#   usage rate (baseline ~0%, should rise sharply after SFT/GRPO; output: output/demo/demo_dialogues.md).
#
# Usage:
#   bash scripts/run_demo_dialogue.sh                       # default: data/ceres_oph_queries.jsonl + 7B baseline
#   bash scripts/run_demo_dialogue.sh --out-dir output/demo/baseline
#   bash scripts/run_demo_dialogue.sh --adapters output/ceres-oph-sft-warmup --out-dir output/demo/sft
#   bash scripts/run_demo_dialogue.sh --adapters output/ceres-oph-grpo-lora-v1/checkpoint-150 --out-dir output/demo/grpo
#   # Compare "action-tag usage rate" and "convergence ratio" across the three demo_dialogues.md.
#   bash scripts/run_demo_dialogue.sh --input data/raw_pool.jsonl --type pool
#   bash scripts/run_demo_dialogue.sh --input /hy-tmp/dataset/CMExam/data/val.csv --type cmexam
#   # Common args: --num 8 --rounds 4 --seed 0 --model /hy-tmp/model/Qwen2.5-VL-32B-Instruct
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  GPU for the demo (default 0, single GPU).
#   Other args pass through to eval/demo_dialogue.py (--help for the full list).
#
# Deps: local venv (swift 3.4.1 + torch) runs the teacher PtEngine; student simulator uses the
#   BigModel glm API (loaded by env_glm.sh; auto-degrades to mock when key is missing).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."        # project root as cwd (data/ relative paths and image paths depend on it)

# Student-simulator API env (BigModel glm; failure only warns — student_sim auto-mocks when key is unset)
if ! source ./scripts/env_glm.sh; then
    echo "[run_demo] ⚠ BigModel 环境装载失败：学生模拟器将走 mock 离线回复（演示仍可跑）。" >&2
fi

# Demo defaults to a single GPU (7B bf16 PtEngine needs ~18GB); override via env var
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Python interpreter: must import swift 3.4.1 + torch (local venv example:
# PY_BIN=/hy-tmp/my-env/swift/bin/python; defaults to python on PATH)
PY_BIN=${PY_BIN:-$(command -v python)}

# Args pass through (see usage above / python eval/demo_dialogue.py --help)
"$PY_BIN" eval/demo_dialogue.py "$@"
