#!/bin/bash
# scripts/env_deepseek.sh — DeepSeek OpenAI-compatible API env loader.
# Usage: source scripts/env_deepseek.sh   (then run the data pipeline / training scripts as usual).
#   Key is read from deepseek-api at the project root (gitignored; never commit or echo it).
#
# Sets two env groups:
#   Teacher side (data synthesis)    CERES_API_BASE / CERES_API_KEY / CERES_TEACHER_MODEL
#   Student side (student simulator) CERES_STUDENT_API_BASE / CERES_STUDENT_API_KEY / CERES_STUDENT_MODEL
# DeepSeek-specific (optional llm_client._remote_chat args; no effect on other providers):
#   CERES_API_THINKING=enabled       extra_body {"thinking":{"type":"enabled"}} (thinking mode)
#   CERES_API_REASONING_EFFORT=high  reasoning_effort (high|medium|low)
# Note: thinking-mode output comes back via message.content (reasoning stays in reasoning_content,
#   not in content), so json_mode still works; latency and token cost are much higher.
DS_KEY_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deepseek-api"
if [ ! -f "$DS_KEY_FILE" ]; then
    echo "[env_deepseek] 缺少 $DS_KEY_FILE（把 DeepSeek API key 放进该文件）" >&2
    return 1 2>/dev/null || exit 1
fi
DS_KEY="$(head -c 200 "$DS_KEY_FILE" | tr -d '[:space:]')"
if [ -z "$DS_KEY" ]; then
    echo "[env_deepseek] deepseek-api 文件为空" >&2
    return 1 2>/dev/null || exit 1
fi

export CERES_API_BASE="https://api.deepseek.com"
export CERES_API_KEY="$DS_KEY"
export CERES_TEACHER_MODEL="${CERES_TEACHER_MODEL:-deepseek-v4-flash}"

export CERES_STUDENT_API_BASE="https://api.deepseek.com"
export CERES_STUDENT_API_KEY="$DS_KEY"
# Student simulator uses the same model as the teacher (override CERES_STUDENT_MODEL=<name> for a cheaper tier)
export CERES_STUDENT_MODEL="${CERES_STUDENT_MODEL:-deepseek-v4-flash}"
# Disable mock, use the real API; cache stays on (same (messages,seed) called once — cheaper & reproducible)
unset CERES_STUDENT_MOCK

# DeepSeek thinking mode (user-specified call shape: reasoning_effort=high + thinking enabled)
export CERES_API_THINKING="${CERES_API_THINKING:-enabled}"
export CERES_API_REASONING_EFFORT="${CERES_API_REASONING_EFFORT:-low}"
# max_tokens quota includes thinking tokens: 1024 is fully consumed by reasoning, leaving an empty
# body (observed) -> raise to 8192
export CERES_SYNTH_MAX_TOKENS="${CERES_SYNTH_MAX_TOKENS:-2048}"

echo "[env_deepseek] DeepSeek 环境已装载：teacher=${CERES_TEACHER_MODEL} student=${CERES_STUDENT_MODEL} thinking=${CERES_API_THINKING}/${CERES_API_REASONING_EFFORT} base=${CERES_API_BASE}"
