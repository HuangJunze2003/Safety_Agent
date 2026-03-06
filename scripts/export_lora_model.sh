#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/llama/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llama/bin/llamafactory-cli}"
export OMP_NUM_THREADS=1

if [ ! -d "outputs/qwen3vl_lora_local" ]; then
  echo "[ERROR] 未找到 LoRA 训练结果目录: outputs/qwen3vl_lora_local"
  exit 1
fi

if [ ! -f "models/base-vl/config.json" ]; then
  echo "[ERROR] 未找到本地基座模型: models/base-vl/config.json"
  exit 1
fi

echo "[INFO] 导出并合并 LoRA 模型到 outputs/qwen2vl_lora_merged ..."
"$LLAMAFACTORY_CLI" export configs/llamafactory/qwen2vl_lora_export.yaml

echo "[INFO] 导出完成: outputs/qwen2vl_lora_merged"
