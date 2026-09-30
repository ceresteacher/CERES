#!/bin/bash
# ============================================================================
# run_sft_warmup.sh — SFT-format warmup for the CERES ophthalmology-teaching GRPO (design §8.3 revision).
#
# Purpose: first run 1 epoch of LoRA SFT on 800~1.5k demonstration trajectories that follow the
#   teaching-action DSL (design §5), lifting the action-tag success rate from ~40% to ~95% before GRPO.
# Data: data/ceres_oph_sft_warmup.jsonl (produced by data_pipeline/quality_filter.py; schema in
#   data/README.md and data_pipeline/README.md).
#
# Revisions vs design §8.3 (appendix A; all multi-GPU pitfalls observed):
#   1) + --split_dataset_ratio 0        default 0.01 carves a degenerate 1-row val set -> multi-GPU
#                                       eval crashes (KeyError: eval_reward).
#   2) + --gradient_checkpointing true + '{"use_reentrant": false}'
#                                       reentrant grad checkpoint conflicts with multi-GPU (DDP/ZeRO-3),
#                                       "lora_B...weight has been marked as ready twice"; single-GPU
#                                       safe, multi-GPU always hits it.
#   3) + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   32B memory-fragmentation guard.
#   4) --loss_scale changed from the draft's all to default (fix-round-2 decision; see launch note):
#                                       'all' also counts student user turns in the loss; SFT should
#                                       teach only the teacher-side actions.
#
# Every flag was grep-verified against installed ms-swift 3.4.1 source (list in .sdd/task-5-report.md).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."        # project root as cwd (relative paths data/ ceres_plugin/ depend on it)

# ---------- Model / output (env-overridable) ----------
CERES_MODEL=${CERES_MODEL:-/hy-tmp/model/Qwen2.5-VL-32B-Instruct}
CERES_OUTPUT_DIR=${CERES_OUTPUT_DIR:-output/ceres-oph-sft-warmup}
CERES_SFT_DATASET=${CERES_SFT_DATASET:-data/ceres_oph_sft_warmup.jsonl}
CERES_LR=${CERES_LR:-1e-5}            # LoRA SFT lr (raise to 1e-4 and rerun if format injection lags)
CERES_EPOCHS=${CERES_EPOCHS:-1}          # epochs (default 1; 2~3 if action tags are unstable)
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # appendix A: 32B fragmentation OOM guard

# ---------- Optional dependency probe & context budget (same as run_grpo_oph_lora.sh; auto-degrade to plain DDP) ----------
PY_BIN=${PY_BIN:-$(command -v python3)}
ENGINE_FLAGS=""
CERES_ENGINE="ddp"
if "$PY_BIN" -c "import deepspeed" 2>/dev/null; then
    ENGINE_FLAGS="$ENGINE_FLAGS --deepspeed zero3"
    CERES_ENGINE="zero3"
fi
if "$PY_BIN" -c "import flash_attn" 2>/dev/null; then
    ENGINE_FLAGS="$ENGINE_FLAGS --attn_impl flash_attn"
fi
if [ "$CERES_ENGINE" = "zero3" ]; then
    CERES_MAX_LENGTH=${CERES_MAX_LENGTH:-4096}
else
    CERES_MAX_LENGTH=${CERES_MAX_LENGTH:-1024}
    echo "[ceres-sft] ⚠ 无 deepspeed → 纯 DDP + 收缩上下文（max_length=${CERES_MAX_LENGTH}）；32B 建议 pip install deepspeed"
fi
echo "[ceres-sft] 引擎：${CERES_ENGINE} | ENGINE_FLAGS='${ENGINE_FLAGS}' | max_length=${CERES_MAX_LENGTH}"
# Gradient checkpointing impl switches with engine (both pitfalls observed):
#   plain DDP + reentrant -> "lora_B weight marked as ready twice" (appendix A) -> use false
#   ZeRO-3 + non-reentrant -> tensor metadata mismatch on recompute (CheckpointError) -> use true
if [ "$CERES_ENGINE" = "zero3" ]; then
    GC_KWARGS='{"use_reentrant": true}'
else
    GC_KWARGS='{"use_reentrant": false}'
fi

# ---------- Dataset existence guard ----------
if [ ! -f "$CERES_SFT_DATASET" ]; then
    echo "[ceres-sft] 错误：SFT 数据集不存在：$CERES_SFT_DATASET" >&2
    echo "[ceres-sft] 请先运行数据管线生成（四步命令见 data_pipeline/README.md，" >&2
    echo "[ceres-sft] 最终产物由 data_pipeline/quality_filter.py 写出），或用 CERES_SFT_DATASET 指向已有文件。" >&2
    exit 1
fi

# ---------- Launch ----------
# loss_scale decision (fix round 2): use default, NOT the design §8.3 draft's all — the draft
# mistakenly carried the §8.1 GRPO semantics into SFT: swift's 'all' = TrainAllLossScale
# (swift/plugin/loss_scale/loss_scale.py:125-128,139), which counts loss on every context token
# regardless of user/assistant — SFT warmup would also learn the student (user) turns; SFT should
# teach only teacher-side actions. 'default' has exactly the needed semantics: all assistant
# (teacher) turns count, user (student) turns don't (loss_scale.py:35-55, key check :52-55).
# Memory budget (per card, 4xA800-80G): ZeRO-3 sharded weights+grads+optimizer ~33GB +
# activations (bs=1, max_length=4096, flash-attn + grad checkpoint) ~6~10GB ~ 40~45GB.
# OOM degrade: --max_length 4096->3072 -> per_device_train_batch_size already 1 -> close other
#   GPU processes; SFT does not use vLLM, so no vllm_util to tune.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
swift sft \
    --model "$CERES_MODEL" \
    --train_type lora \
    --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
    --freeze_vit true --freeze_aligner true \
    --torch_dtype bfloat16 \
    --dataset "$CERES_SFT_DATASET" \
    --max_length "$CERES_MAX_LENGTH" \
    --split_dataset_ratio 0 \
    --loss_scale default \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate "$CERES_LR" \
    --num_train_epochs "$CERES_EPOCHS" \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs "$GC_KWARGS" \
    $ENGINE_FLAGS \
    --logging_steps 5 \
    --save_steps 50 --save_total_limit 2 \
    --report_to tensorboard \
    --add_version false \
    --output_dir "$CERES_OUTPUT_DIR" 2>&1 | tee sft_warmup.log

# ---------- Handoff notes ----------
# After the run, the LoRA adapter is at $CERES_OUTPUT_DIR (--add_version false, no version subdir).
# Next GRPO warmup handoff: export CERES_ADAPTERS=<this script's output_dir>, then run
#   scripts/run_grpo_oph_lora.sh (it turns CERES_ADAPTERS into --adapters to mount and resume this LoRA).
# ⚠ Do not feed the SFT output to CERES_MODEL: 3.4.1's --model only accepts a base model dir, not an
#   adapter dir (adapters are detected by the adapters/ckpt_dir branch,
#   swift/llm/argument/base_args/base_args.py:32-43/74).
echo "[ceres-sft] SFT 预热完成。GRPO 衔接：CERES_ADAPTERS=$CERES_OUTPUT_DIR bash scripts/run_grpo_oph_lora.sh"
