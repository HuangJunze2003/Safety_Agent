#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llama/bin/llamafactory-cli}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/qwen2vl_lora_merged}"

export OMP_NUM_THREADS=1
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[ERROR] 未找到模型: $MODEL_PATH/config.json"
  exit 1
fi

echo "[INFO] 启动 LLaMA-Factory WebUI"
echo "[INFO] 模型路径: $MODEL_PATH"
echo "[INFO] 访问地址: http://127.0.0.1:${GRADIO_SERVER_PORT}"

"$LLAMAFACTORY_CLI" webui
