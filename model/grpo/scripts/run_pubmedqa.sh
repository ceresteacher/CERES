#!/bin/bash
# scripts/run_pubmedqa.sh — offline generation of the PubMedQA pqa_labeled 1k English dual-format GRPO datasets (8 steps).
#   Outputs (all rows lang='en'):
#     data/ceres_pubmedqa_queries.jsonl  CERES multi-turn teaching-contract query set (+3 gt columns);
#                                        train directly via CERES_DATASET=data/ceres_pubmedqa_queries.jsonl
#                                        bash scripts/run_grpo_oph_lora.sh
#     data/pubmedqa_qa_grpo.jsonl        standard single-turn QA GRPO set (student-style question keeping
#                                        A. Yes / B. No / C. Maybe options + gt answer)
#     data/pubmedqa_report.json          distribution/quality report
#   Mapping: yes/no/maybe -> A/B/C; gt_explanation=long_answer; English eye questions route to the
#          ophthalmology node, the rest to general_medicine/qa; all 1,000 pqa_labeled rows (no sampling).
#   Usage:
#     bash scripts/run_pubmedqa.sh        # real API (default: source scripts/env_deepseek.sh,
#                                         #   deepseek-v4-flash + thinking low; ENV_SCRIPT overridable)
#     MOCK=1 bash scripts/run_pubmedqa.sh # offline smoke (no API; 100% rewrite degradation expected;
#                                         #   step 1 download only needs network the first time)
#   Env vars: PUBMEDQA_DIR(/hy-tmp/dataset/PubMedQA) / TOTAL(1000) / SEED(42) /
#             CONCURRENCY(16) / PYTHON / VENV_PYTHON / ENV_SCRIPT / HF_MIRROR
#   (real API needs an interpreter with the openai package:
#     PYTHON=/hy-tmp/my-env/swift/bin/python bash scripts/run_pubmedqa.sh)
#   Resume: just re-run (llm_client disk cache hits on (messages,seed,model); keep SEED and
#     CERES_TEACHER_MODEL unchanged).
set -euo pipefail
cd "$(dirname "$0")/.."

PUBMEDQA_DIR=${PUBMEDQA_DIR:-/hy-tmp/dataset/PubMedQA}
TOTAL=${TOTAL:-1000}
SEED=${SEED:-42}
CONCURRENCY=${CONCURRENCY:-16}
MOCK=${MOCK:-0}
ENV_SCRIPT=${ENV_SCRIPT:-scripts/env_deepseek.sh}

MANIFEST=data/manifest_pubmedqa.jsonl
POOL=data/raw_pool_pubmedqa.jsonl
CAND_OPEN=data/queries_candidates_pubmedqa.jsonl
CAND_MCQ=data/queries_candidates_pubmedqa_mcq.jsonl
QUERIES=data/ceres_pubmedqa_queries.jsonl
QA=data/pubmedqa_qa_grpo.jsonl
REPORT=data/pubmedqa_report.json

EXTRA_RW=""
if [ "$MOCK" = "1" ]; then
    EXTRA_RW="--mock --no-cache"
else
    source "$ENV_SCRIPT"
fi
export CERES_SYNTH_CONCURRENCY=$CONCURRENCY

step() { echo; echo "[pubmedqa] ===== $* ====="; }

step "1/8 下载+转换（幂等；jsonl 已在则跳过，仅本步需要网络与 venv python）"
bash scripts/download_pubmedqa.sh

step "2/8 登记（--source-name pubmedqa，独立 manifest）"
${PYTHON:-python3} -m data_pipeline.prepare_datasets --type pubmedqa \
    --input "$PUBMEDQA_DIR/pqa_labeled.jsonl" --source-name pubmedqa --output $MANIFEST

step "3/8 抽题（--no-keyword 放行；眼科/全科节点分流；lang=en；纯规则无 LLM）"
${PYTHON:-python3} -m data_pipeline.extract_questions --input $MANIFEST --output $POOL \
    --no-keyword --lang en

EXPECT=$(wc -l < $POOL)
if [ "$EXPECT" -ne "$TOTAL" ]; then
    echo "[pubmedqa] 警告：pool 行数 $EXPECT != TOTAL $TOTAL（适配器有丢行，查上方 stderr；verify 按实际 $EXPECT 校验）"
fi

step "4/8 改写 A：开放式学生提问（无选项痕迹 → 契约 schema）"
${PYTHON:-python3} -m data_pipeline.rewrite_question --input $POOL --output $CAND_OPEN \
    --seed $SEED $EXTRA_RW

step "5/8 改写 B：选择题式学生提问（保留 A–C 选项 → QA GRPO）"
${PYTHON:-python3} -m data_pipeline.rewrite_question --input $POOL --output $CAND_MCQ \
    --mode mcq --seed $SEED $EXTRA_RW

step "6/8 构建 CERES 契约查询集（gt_label/gt_answer/gt_explanation 三列）"
${PYTHON:-python3} -m data_pipeline.build_query_dataset --input $CAND_OPEN --output $QUERIES

step "7/8 构建标准单轮 QA GRPO 集"
${PYTHON:-python3} -m data_pipeline.build_qa_dataset --input $CAND_MCQ --output $QA

step "8/8 统计与健全性校验（--lang en；硬校验失败 exit 1）"
${PYTHON:-python3} -m data_pipeline.verify_cmexam4k --queries $QUERIES --qa $QA \
    --candidates $CAND_OPEN --candidates-mcq $CAND_MCQ \
    --lang en --expect $EXPECT --report $REPORT

echo
echo "[pubmedqa] 完成。训练接驳："
echo "  CERES_DATASET=$QUERIES bash scripts/run_grpo_oph_lora.sh"
