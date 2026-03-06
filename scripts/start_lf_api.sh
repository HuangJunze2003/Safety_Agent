#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llama/bin/llamafactory-cli}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-./models/base-vl}"
ADAPTER_PATH="${ADAPTER_PATH:-./outputs/qwen3vl_lora_local}"
MODEL_PATH="${MODEL_PATH:-./outputs/qwen2vl_lora_merged}"

export OMP_NUM_THREADS=1

HAS_ADAPTER=false
if [ -f "$ADAPTER_PATH/adapter_config.json" ]; then
  HAS_ADAPTER=true
fi

if [ "$HAS_ADAPTER" = true ] && [ ! -f "$BASE_MODEL_PATH/config.json" ]; then
  echo "[ERROR] 检测到 LoRA 适配器，但未找到基座模型: $BASE_MODEL_PATH/config.json"
  exit 1
fi

if [ "$HAS_ADAPTER" = false ] && [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[ERROR] 未找到可用模型。"
  echo "[ERROR] 期望其一存在:"
  echo "  1) LoRA 适配器: $ADAPTER_PATH/adapter_config.json（并且基座模型存在）"
  echo "  2) 合并模型: $MODEL_PATH/config.json"
  exit 1
fi

echo "[INFO] 启动 LLaMA-Factory API 服务"
echo "[INFO] OpenAPI: http://127.0.0.1:8000/docs"

if [ "$HAS_ADAPTER" = true ]; then
  echo "[INFO] 模式: Base + LoRA Adapter"
  echo "[INFO] 基座模型: $BASE_MODEL_PATH"
  echo "[INFO] 微调适配器: $ADAPTER_PATH"
  "$LLAMAFACTORY_CLI" api \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --template qwen2_vl \
    --infer_backend huggingface
else
  echo "[INFO] 模式: Merged Model"
  echo "[INFO] 模型路径: $MODEL_PATH"
  "$LLAMAFACTORY_CLI" api \
    --model_name_or_path "$MODEL_PATH" \
    --template qwen2_vl \
    --infer_backend huggingface
fi
