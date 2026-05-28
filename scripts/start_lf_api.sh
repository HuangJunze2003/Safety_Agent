#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llama/bin/llamafactory-cli}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-./models/base-vl}"
ADAPTER_PATH="${ADAPTER_PATH:-./outputs/qwen3vl_lora_long_fine}"
MODEL_PATH="${MODEL_PATH:-./outputs/qwen2vl_lora_merged}"
# merged：加载已合并权重（推荐，避免 LoRA merge 时 OOM）
# lora：基座 + 适配器（需预留足够显存，且勿与 WebUI 同时占满 GPU）
INFER_MODE="${INFER_MODE:-merged}"

export OMP_NUM_THREADS=1

HAS_ADAPTER=false
if [ -f "$ADAPTER_PATH/adapter_config.json" ]; then
  HAS_ADAPTER=true
fi
HAS_MERGED=false
if [ -f "$MODEL_PATH/config.json" ]; then
  HAS_MERGED=true
fi

if [ "$HAS_ADAPTER" = true ] && [ ! -f "$BASE_MODEL_PATH/config.json" ]; then
  echo "[ERROR] 检测到 LoRA 适配器，但未找到基座模型: $BASE_MODEL_PATH/config.json"
  exit 1
fi

USE_LORA=false
if [ "$INFER_MODE" = "lora" ]; then
  USE_LORA=true
elif [ "$INFER_MODE" = "merged" ]; then
  USE_LORA=false
else
  # auto：优先 merged，避免 API 启动时 merge LoRA 导致显存溢出
  if [ "$HAS_MERGED" = true ]; then
    USE_LORA=false
  elif [ "$HAS_ADAPTER" = true ]; then
    USE_LORA=true
  fi
fi

if [ "$USE_LORA" = false ] && [ "$HAS_MERGED" = false ]; then
  echo "[ERROR] 未找到合并模型: $MODEL_PATH/config.json"
  echo "[ERROR] 请先执行 bash scripts/export_lora_model.sh，或设置 INFER_MODE=lora 且保证显存充足。"
  exit 1
fi

if [ "$USE_LORA" = true ] && [ "$HAS_ADAPTER" = false ]; then
  echo "[ERROR] INFER_MODE=lora 但未找到适配器: $ADAPTER_PATH/adapter_config.json"
  exit 1
fi

echo "[INFO] 启动 LLaMA-Factory API 服务"
echo "[INFO] OpenAPI: http://127.0.0.1:8000/docs"

if [ "$USE_LORA" = true ]; then
  echo "[INFO] 模式: Base + LoRA Adapter（INFER_MODE=lora）"
  echo "[INFO] 基座模型: $BASE_MODEL_PATH"
  echo "[INFO] 微调适配器: $ADAPTER_PATH"
  "$LLAMAFACTORY_CLI" api \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --template qwen2_vl \
    --infer_backend huggingface
else
  echo "[INFO] 模式: Merged Model（INFER_MODE=merged）"
  echo "[INFO] 模型路径: $MODEL_PATH"
  "$LLAMAFACTORY_CLI" api \
    --model_name_or_path "$MODEL_PATH" \
    --template qwen2_vl \
    --infer_backend huggingface
fi
