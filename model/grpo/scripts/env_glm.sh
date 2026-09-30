#!/bin/bash
# scripts/env_glm.sh — BigModel (Zhipu) OpenAI-compatible API env loader.
# Usage: source scripts/env_glm.sh   (then run the data pipeline / training scripts as usual).
#   Key is read from glm-api at the project root (gitignored; never commit or echo it).
#
# Sets two env groups (docs.bigmodel.cn OpenAI-compatible mode, verified 2026-08-31):
#   Teacher side (data synthesis)    CERES_API_BASE / CERES_API_KEY / CERES_TEACHER_MODEL
#   Student side (student simulator) CERES_STUDENT_API_BASE / CERES_STUDENT_API_KEY / CERES_STUDENT_MODEL
# glm-5.3-flash supports response_format={"type":"json_object"}, so student_sim needs no change.
GLM_KEY_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/glm-api"
if [ ! -f "$GLM_KEY_FILE" ]; then
    echo "[env_glm] 缺少 $GLM_KEY_FILE（把 BigModel API key 放进该文件）" >&2
    return 1 2>/dev/null || exit 1
fi
GLM_KEY="$(head -c 200 "$GLM_KEY_FILE" | tr -d '[:space:]')"
if [ -z "$GLM_KEY" ]; then
    echo "[env_glm] glm-api 文件为空" >&2
    return 1 2>/dev/null || exit 1
fi

export CERES_API_BASE="https://open.bigmodel.cn/api/paas/v4/"
export CERES_API_KEY="$GLM_KEY"
export CERES_TEACHER_MODEL="${CERES_TEACHER_MODEL:-glm-5.3-flash}"

export CERES_STUDENT_API_BASE="https://open.bigmodel.cn/api/paas/v4/"
export CERES_STUDENT_API_KEY="$GLM_KEY"
export CERES_STUDENT_MODEL="${CERES_STUDENT_MODEL:-glm-5.3-flash}"
# Disable mock, use the real API; cache stays on (same (messages,seed) called once — cheaper & reproducible)
unset CERES_STUDENT_MOCK

echo "[env_glm] BigModel 环境已装载：teacher=${CERES_TEACHER_MODEL} student=${CERES_STUDENT_MODEL} base=${CERES_API_BASE}"
