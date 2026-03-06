#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/data_processed
cp -f configs/llamafactory/dataset_info.json data/data_processed/dataset_info.json

echo "[INFO] dataset_info.json 已同步到 data/data_processed/"
echo "[INFO] 开始 LoRA 微调..."

llamafactory-cli train configs/llamafactory/qwen3vl_lora.yaml
