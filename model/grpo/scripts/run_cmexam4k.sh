#!/bin/bash
# scripts/run_cmexam4k.sh — offline generation of the CMExam 4k dual-format GRPO datasets (8 steps).
#   Outputs:
#     data/ceres_cmexam4k_queries.jsonl  CERES multi-turn teaching-contract query set (+3 gt columns);
#                                        train directly via CERES_DATASET=data/ceres_cmexam4k_queries.jsonl
#                                        bash scripts/run_grpo_oph_lora.sh
#     data/cmexam4k_qa_grpo.jsonl        standard single-turn QA GRPO set (student-style MCQ + gt answer)
#     data/cmexam4k_report.json          distribution/quality report
#   Usage:
#     bash scripts/run_cmexam4k.sh        # real API (default: source scripts/env_deepseek.sh, deepseek-v4-pro
#                                         #   + thinking high; ENV_SCRIPT can switch back to env_glm.sh)
#     MOCK=1 bash scripts/run_cmexam4k.sh # offline smoke (no API; 100% rewrite degradation expected)
#   Env vars: CMEXAM_TRAIN / TOTAL(4000) / SEED(42) / CONCURRENCY(16) / PYTHON / ENV_SCRIPT
#     (real API needs an interpreter with the openai package:
#      PYTHON=/hy-tmp/my-env/swift/bin/python bash scripts/run_cmexam4k.sh)
#   Resume: just re-run (llm_client disk cache hits on (messages,seed,model); keep SEED and
#     CERES_TEACHER_MODEL unchanged).
set -euo pipefail
cd "$(dirname "$0")/.."

CMEXAM_TRAIN=${CMEXAM_TRAIN:-/hy-tmp/dataset/CMExam/data/train.csv}
TOTAL=${TOTAL:-4000}
SEED=${SEED:-42}
CONCURRENCY=${CONCURRENCY:-16}
MOCK=${MOCK:-0}
ENV_SCRIPT=${ENV_SCRIPT:-scripts/env_deepseek.sh}

SAMPLE_CSV=data/cmexam_4k_sample.csv
MANIFEST=data/manifest_cmexam4k.jsonl
POOL=data/raw_pool_cmexam4k.jsonl
CAND_OPEN=data/queries_candidates_cmexam4k.jsonl
CAND_MCQ=data/queries_candidates_cmexam4k_mcq.jsonl
QUERIES=data/ceres_cmexam4k_queries.jsonl
QA=data/cmexam4k_qa_grpo.jsonl
REPORT=data/cmexam4k_report.json

EXTRA_RW=""
if [ "$MOCK" = "1" ]; then
    EXTRA_RW="--mock --no-cache"
else
    source "$ENV_SCRIPT"
fi
export CERES_SYNTH_CONCURRENCY=$CONCURRENCY

step() { echo; echo "[cmexam4k] ===== $* ====="; }

step "1/8 采样（眼科优先 + seed=$SEED 确定性补足到 $TOTAL；只用 train，val/test 留作评测）"
${PYTHON:-python3} -m data_pipeline.sample_cmexam --input "$CMEXAM_TRAIN" --output $SAMPLE_CSV \
    --total $TOTAL --seed $SEED

step "2/8 登记到独立 manifest（--source-name cmexam4k，不触碰 data/manifest.jsonl）"
${PYTHON:-python3} -m data_pipeline.prepare_datasets --type cmexam --input $SAMPLE_CSV \
    --source-name cmexam4k --output $MANIFEST

step "3/8 抽题（--no-keyword 放行全科行；眼科/全科节点分流；lang=zh；纯规则无 LLM）"
${PYTHON:-python3} -m data_pipeline.extract_questions --input $MANIFEST --output $POOL \
    --no-keyword --lang zh

step "4/8 改写 A：开放式学生提问（无选项痕迹 → 契约 schema）"
${PYTHON:-python3} -m data_pipeline.rewrite_question --input $POOL --output $CAND_OPEN \
    --seed $SEED $EXTRA_RW

step "5/8 改写 B：选择题式学生提问（保留 A–E 选项 → QA GRPO）"
${PYTHON:-python3} -m data_pipeline.rewrite_question --input $POOL --output $CAND_MCQ \
    --mode mcq --seed $SEED $EXTRA_RW

step "6/8 构建 CERES 契约查询集（扩展 gt_label/gt_answer/gt_explanation 三列）"
${PYTHON:-python3} -m data_pipeline.build_query_dataset --input $CAND_OPEN --output $QUERIES

step "7/8 构建标准单轮 QA GRPO 集"
${PYTHON:-python3} -m data_pipeline.build_qa_dataset --input $CAND_MCQ --output $QA

step "8/8 统计与健全性校验（硬校验失败 exit 1）"
${PYTHON:-python3} -m data_pipeline.verify_cmexam4k --queries $QUERIES --qa $QA \
    --candidates $CAND_OPEN --candidates-mcq $CAND_MCQ \
    --expect $TOTAL --report $REPORT

echo
echo "[cmexam4k] 完成。训练接驳："
echo "  CERES_DATASET=$QUERIES bash scripts/run_grpo_oph_lora.sh"
