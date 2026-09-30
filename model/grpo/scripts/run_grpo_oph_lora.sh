#!/bin/bash
# ============================================================================
# run_grpo_oph_lora.sh — CERES ophthalmology-teaching GRPO main training script (LoRA r=64 + ZeRO-3 + vLLM colocate).
# Based on design §8.2, revised per the pre-flight decision (see .sdd/contracts.md and .sdd/task-5-report.md).
#
# Key revisions vs the design draft (all verified against installed ms-swift 3.4.1 source):
#   1) --max_turns / --vllm_mode / --steps_per_generation do not exist in 3.4.1 -> removed.
#      The turn cap is enforced by the plugin via env CERES_MAX_TURNS (default 5);
#      generation_batch = per_device_bs x world_size x grad_accum.
#   2) --multi_turns teacher_env -> --multi_turn_func teacher_env (swift/trainers/arguments.py:186).
#   3) --vllm_tensor_parallel_size does not exist -> equivalent tensor_parallel_size (default 1, omitted).
#   4) [fix round 1] default to pt infer (no --use_vllm): vLLM colocate has a structural
#      "rollout vs scoring cross-rank" risk (section 3 below); append vLLM args only when
#      CERES_USE_VLLM=1 (incl. --num_infer_workers 4, rlhf_args.py:226-247).
#   5) + --split_dataset_ratio 0 (multi-GPU eval_reward KeyError guard, appendix A).
#   6) + --gradient_checkpointing true + use_reentrant:false (multi-GPU LoRA pitfall, appendix A).
#
# Every flag was grep-verified against installed ms-swift 3.4.1 source; the flag->file:line table
# is in .sdd/task-5-report.md.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."        # project root as cwd (--external_plugins relative path depends on it)

# ============================================================================
# 1. Top env vars: virtual student simulator (cloud OpenAI-compatible API, not on training GPUs)
# ============================================================================
export CERES_STUDENT_API_BASE=${CERES_STUDENT_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}
export CERES_STUDENT_MODEL=${CERES_STUDENT_MODEL:-qwen-plus}     # switch to qwen-vl-plus when the student needs vision
export CERES_STUDENT_CONCURRENCY=${CERES_STUDENT_CONCURRENCY:-64}
export CERES_STUDENT_TIMEOUT=${CERES_STUDENT_TIMEOUT:-30}
# export CERES_STUDENT_API_KEY=sk-xxxx          # set before real training (never commit a real key)
# API key must be set (otherwise every student falls back to degraded replies and the GPU is wasted):
if [ -z "${CERES_STUDENT_API_KEY:-}" ] && [ "${CERES_STUDENT_MOCK:-0}" != "1" ]; then
    echo "[ceres-grpo] 错误：未设置 CERES_STUDENT_API_KEY。" >&2
    echo "[ceres-grpo] 正式训练：export CERES_STUDENT_API_KEY=sk-xxx 后重跑；" >&2
    echo "[ceres-grpo] 离线自测：export CERES_STUDENT_MOCK=1（student_sim 走确定性 mock）。" >&2
    exit 1
fi
export CERES_MAX_TURNS=${CERES_MAX_TURNS:-5}    # per-trajectory dialogue cap (read by plugin; swift has no --max_turns; override to speed up)
CERES_NUM_GENERATIONS=${CERES_NUM_GENERATIONS:-8}
export CERES_NUM_GENERATIONS                # plugin-side group size G must match --num_generations
export CERES_TRAJ_DIR=${CERES_TRAJ_DIR:-./traj_dump}   # trajectory dump dir (consumed by eval/analyze_traj.py)
export CERES_END_PENALTY=${CERES_END_PENALTY:-0.2}     # fix round 2: penalty for non-<end/> termination (0 = ablate to old behavior)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # appendix A: 32B memory-fragmentation OOM guard

# ============================================================================
# 1.5 Fix-round-2 hyperparams (2026-09-02, train_oph_lora.log incident review; all env-overridable)
#   Incident: <end/> termination rate 55%->0% over 250 steps; latter half 100% truncated at 768 tokens;
#   sequence reward 0.986->0.73 (RL trained the reward down). Three root causes:
#   ① reward did not penalize truncation (now plugged by CERES_END_PENALTY);
#   ② anchor z-score noise dominated within-group advantage variance (student self-assessment unreliable);
#   ③ lr 1e-5 too high: KL hit 5.6 after warmup (steps 15-25), policy pushed off the SFT manifold.
#   Fixes: anchor weight 0.5->0.1; lr 1e-5->3e-6; beta 0.04->0.08; warmup 3%->10%;
#         max_completion 768->1024 (45% hit the cap at step 1).
# ============================================================================
CERES_REWARD_WEIGHTS=${CERES_REWARD_WEIGHTS:-"1.0 0.1"}   # anchor down-weighted; set "1.0 0" to ablate
CERES_LR=${CERES_LR:-3e-6}
CERES_BETA=${CERES_BETA:-0.08}
CERES_WARMUP_RATIO=${CERES_WARMUP_RATIO:-0.1}

# ============================================================================
# 2. Model / batch / output (env-overridable)
# ============================================================================
CERES_MODEL=${CERES_MODEL:-/hy-tmp/model/Qwen2.5-VL-32B-Instruct}   # local model path (verified in contracts)
# Warmup->RL handoff: run scripts/run_sft_warmup.sh first, then
#   export CERES_ADAPTERS=output/ceres-oph-sft-warmup
# (the exact 3.4.1 equivalent of the design draft's "point --model at the SFT checkpoint" is
#   --adapters: swift detects the adapter dir's adapter_config.json, mounts and resumes that LoRA;
#   see swift/llm/argument/base_args/base_args.py:74 and :32-43)
CERES_ADAPTERS=${CERES_ADAPTERS:-}
CERES_OUTPUT_DIR=${CERES_OUTPUT_DIR:-output/ceres-oph-grpo-lora-v1}
CERES_DATASET=${CERES_DATASET:-data/ceres_oph_queries.jsonl}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# ---------- 2.5 Optional dependency probe & context budget (auto-degrade without deepspeed/flash-attn) ----------
# deepspeed present -> design main path ZeRO-3 (sharded weights, 8192 context budget 61~65GB);
# absent -> plain DDP (32B replicates 66GB weights per card, 4xA800 ~80.4/80GB @max_length 768),
#          must shrink context; long-dialogue teaching quality suffers — install deepspeed.
PY_BIN=${PY_BIN:-$(command -v python3)}
ENGINE_FLAGS=""
CERES_ENGINE="ddp"
if "$PY_BIN" -c "import deepspeed" 2>/dev/null; then
    # CERES_DEEPSPEED override (default zero3): 32B GRPO pt-infer rollout gathers full weights
    # per card in unwrap_model_for_generation (64GB full + 16GB local shard > 80GB, step0 OOM on
    # 2026-09-01). zero3_offload moves the local shard to CPU, leaving full gathered weights + KV cache.
    ENGINE_FLAGS="$ENGINE_FLAGS --deepspeed ${CERES_DEEPSPEED:-zero3}"
    CERES_ENGINE="zero3"
fi
if "$PY_BIN" -c "import flash_attn" 2>/dev/null; then
    ENGINE_FLAGS="$ENGINE_FLAGS --attn_impl flash_attn"
fi
if [ "$CERES_ENGINE" = "zero3" ]; then
    CERES_MAX_LENGTH=${CERES_MAX_LENGTH:-8192}
    CERES_MAX_COMPLETION=${CERES_MAX_COMPLETION:-1024}   # fix round 2: 768 budget was tight (45% hit cap at step 1)
else
    CERES_MAX_LENGTH=${CERES_MAX_LENGTH:-1024}
    CERES_MAX_COMPLETION=${CERES_MAX_COMPLETION:-256}
    echo "[ceres-grpo] ⚠ 无 deepspeed → 纯 DDP + 收缩上下文（max_length=${CERES_MAX_LENGTH}，completion=${CERES_MAX_COMPLETION}）"
    echo "[ceres-grpo]   32B 多轮教学建议先：pip install deepspeed（启用 ZeRO-3 后自动恢复 8192 上下文）"
fi
echo "[ceres-grpo] 引擎：${CERES_ENGINE} | ENGINE_FLAGS='${ENGINE_FLAGS}' | max_length=${CERES_MAX_LENGTH}"
# Gradient checkpointing impl switches with engine (both pitfalls observed):
#   plain DDP + reentrant -> "lora_B weight marked as ready twice" (appendix A) -> use false
#   ZeRO-3 + non-reentrant -> tensor metadata mismatch on recompute (CheckpointError) -> use true
if [ "$CERES_ENGINE" = "zero3" ]; then
    GC_KWARGS='{"use_reentrant": true}'
else
    GC_KWARGS='{"use_reentrant": false}'
fi
PER_DEV_BS=${PER_DEV_BS:-1}                 # micro-batch per card
GRAD_ACCUM=${GRAD_ACCUM:-16}                # gradient accumulation

# ---------- Dataset existence guard ----------
if [ ! -f "$CERES_DATASET" ]; then
    echo "[ceres-grpo] 错误：GRPO 查询集不存在：$CERES_DATASET" >&2
    echo "[ceres-grpo] 请先运行数据管线（data_pipeline/build_query_dataset.py 产出，" >&2
    echo "[ceres-grpo] 命令块见 data_pipeline/README.md 与 data/README.md），或用 CERES_DATASET 指向已有文件。" >&2
    exit 1
fi

# ---------- Batch divisibility check (3.4.1 semantics) ----------
# swift/trainers/rlhf_trainer/grpo_trainer.py:232-247:
#   effective_train_batch_size = per_device_train_batch_size x num_processes x gradient_accumulation_steps
#   (3.4.1 has no --steps_per_generation; grad_accum is the replay factor)
#   Default: 1 x 4 x 16 = 64 = generation_batch; 64 / num_generations(8) = 8 independent prompts.
#   Non-divisible -> trainer raises ValueError (grpo_trainer.py:243); compute here for a friendlier error.
WORLD_SIZE=$NPROC_PER_NODE
GENERATION_BATCH=$((PER_DEV_BS * WORLD_SIZE * GRAD_ACCUM))
if [ $((GENERATION_BATCH % CERES_NUM_GENERATIONS)) -ne 0 ]; then
    echo "[ceres-grpo] 错误：generation_batch(${GENERATION_BATCH}) 必须被 num_generations(${CERES_NUM_GENERATIONS}) 整除。" >&2
    echo "[ceres-grpo] 当前：PER_DEV_BS=${PER_DEV_BS} × WORLD_SIZE=${WORLD_SIZE} × GRAD_ACCUM=${GRAD_ACCUM} = ${GENERATION_BATCH}。" >&2
    exit 1
fi
echo "[ceres-grpo] 批次校验通过：generation_batch=${GENERATION_BATCH}，${GENERATION_BATCH}/${CERES_NUM_GENERATIONS}=$((GENERATION_BATCH / CERES_NUM_GENERATIONS)) 组独立 prompt/step"

# ---------- LoRA adapter handoff (optional) ----------
# swift 3.4.1 pitfall: training-side --adapters alone creates a zero-init LoRA (weights not
# loaded; behavior = base; tuner.py:355 only loads weights in the resume_from_checkpoint branch).
# --adapters only mounts on the inference side (the ref model uses it via rlhf.py:71).
# Correct way to resume an SFT adapter: --resume_from_checkpoint + --resume_only_model true
# (loads weights only, not SFT optimizer/scheduler/trainer state; train_args.py:147 syncs it
# into args.adapters, so the ref model mounts correctly too).
ADAPTER_ARGS=()
if [ -n "$CERES_ADAPTERS" ]; then
    ADAPTER_ARGS=(--resume_from_checkpoint "$CERES_ADAPTERS" --resume_only_model true)
    echo "[ceres-grpo] 挂载 SFT 预热 adapter（resume_only_model）：$CERES_ADAPTERS"
fi

# ---------- Step cap (optional: CERES_MAX_STEPS=N to smoke-test throughput/reward first) ----------
MAX_STEPS_ARGS=()
if [ -n "${CERES_MAX_STEPS:-}" ]; then
    MAX_STEPS_ARGS=(--max_steps "$CERES_MAX_STEPS")
    echo "[ceres-grpo] 步数上限：$CERES_MAX_STEPS（验证段；吞吐达标后去掉重跑全量）"
fi

# ============================================================================
# 3. Inference backend selection + training launch (4xA800-80G)
# ============================================================================
# ---------- Inference backend: default pt infer (fix-round-1 decision) ----------
# ① Why vLLM is off by default: with colocate, multi_turn_func runs only in infer-rank processes —
#    rollout side effects (student API calls, TrajectoryStore writes) land in that process's memory;
#    while _score_completions scores each rank's local batch (grpo_trainer.py:887-890 -> :907).
#    _fast_infer gathers all inputs then round-robins slices to infer ranks (grpo_trainer.py:760-772;
#    infer rank determined by :510-521) — the rollout batch and the batch a rank scores are not
#    guaranteed to align -> STORE misses (uid, last-turn text) -> reward silently becomes 0.
#    pt infer (no --use_vllm) rolls out and scores in the same process/batch (grpo_trainer.py:866-871),
#    so STORE is consistent by construction; same path as the verified run_4card_32b.sh.
# ② Enable precondition (CERES_USE_VLLM=1): unified testing must first verify "rollout rank ==
#    scoring rank" — check completions.jsonl rewards are not all-zero and no uid-miss warning flood.
# ③ Memory/throughput: pt infer matches the verified script; ZeRO-3 budget per design §8.2
#    (shard ~33GB + activations ~8~12GB < 80GB), saves vLLM's ~20GB resident vs colocate but
#    autoregressive sampling is slower; OOM degrade order unchanged (below).
VLLM_ARGS=()
if [ "${CERES_USE_VLLM:-0}" = "1" ]; then
    echo "[ceres-grpo] CERES_USE_VLLM=1：附加 vLLM colocate 参数。"
    echo "[ceres-grpo] ⚠ --num_infer_workers 必须等于卡数才进 colocate 分支（rlhf_args.py:226-247，"
    echo "[ceres-grpo]   否则 async 分支断言 4 == 4+1 直接报错）；且奖励跨 rank 归零风险未验证，"
    echo "[ceres-grpo]   统一测试需按脚本头注释 ② 核对 completions.jsonl 奖励是否全 0。"
    VLLM_ARGS=(
        --use_vllm true
        --num_infer_workers "$WORLD_SIZE"
        --vllm_gpu_memory_utilization 0.25
        --vllm_max_model_len 8192
    )
fi

# ---------- Memory budget & OOM degrade (appendix A; change one item at a time, rerun) ----------
# Budget per card (design §4.1/§8.2; pt-infer default path):
#   ZeRO-3 sharded weights+grads+optimizer ~33GB + activations (bs=1, len=8192, flash-attn +
#   grad checkpoint) ~8~12GB ~ 41~45GB < 80GB. If CERES_USE_VLLM=1, add vLLM 0.25x80GB ~20GB ~ 61~65GB.
# OOM degrade order:
#   a) --max_length 8192 -> 6144;
#   b) ensure --deepspeed zero3 (zero2 replicates 65GB weights per card — will OOM);
#   c) only when CERES_USE_VLLM=1: --vllm_gpu_memory_utilization 0.25 -> 0.22 (lower slows sampling),
#      --vllm_max_model_len 8192 -> 6144; under colocate swift also suggests --sleep_level 1
#      (vLLM sleeps during training to free memory, rlhf_args.py:240-243, last resort).
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
swift rlhf \
    --rlhf_type grpo \
    --model "$CERES_MODEL" \
    "${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"}" \
    --train_type lora \
    --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
    --freeze_vit true --freeze_aligner true \
    --torch_dtype bfloat16 \
    --dataset "$CERES_DATASET" \
    --max_length "$CERES_MAX_LENGTH" \
    --external_plugins ceres_plugin/plugin.py \
    --multi_turn_func teacher_env \
    --loss_scale all \
    --reward_funcs ceres_sequence ceres_anchor \
    --reward_weights $CERES_REWARD_WEIGHTS \
    --num_generations "$CERES_NUM_GENERATIONS" \
    --num_iterations 1 \
    --max_completion_length "$CERES_MAX_COMPLETION" \
    --temperature 1.0 --top_p 0.99 \
    --beta "$CERES_BETA" --epsilon 0.2 \
    --scale_rewards true \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$CERES_LR" \
    --warmup_ratio "$CERES_WARMUP_RATIO" \
    --num_train_epochs 2 \
    "${MAX_STEPS_ARGS[@]+"${MAX_STEPS_ARGS[@]}"}" \
    "${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"}" \
    $ENGINE_FLAGS \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs "$GC_KWARGS" \
    --split_dataset_ratio 0 \
    --log_completions true \
    --logging_steps 5 \
    --report_to tensorboard \
    --save_steps 50 --save_total_limit 5 \
    --add_version false \
    --output_dir "$CERES_OUTPUT_DIR" 2>&1 | tee train_oph_lora.log

# ============================================================================
# 4. Post-training
# ============================================================================
echo "[ceres-grpo] 训练结束。离线评估：python eval/analyze_traj.py --traj-dir $CERES_TRAJ_DIR --out report.md"
echo "[ceres-grpo] 提醒：mastery 增益/误解消解为「仅观测不进 reward」指标（design §7.5 预留位）。"
if [ "${CERES_USE_VLLM:-0}" != "1" ]; then
    echo "[ceres-grpo] 本次为 pt infer 路径；如需试 vLLM（先读脚本头注释 ①② 的跨 rank 风险）：CERES_USE_VLLM=1 bash $0"
fi
