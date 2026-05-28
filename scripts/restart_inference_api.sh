#!/usr/bin/env bash
# 释放 8000 端口并重启推理 API（默认 merged 模式，避免 LoRA merge OOM）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export INFER_MODE="${INFER_MODE:-merged}"
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/qwen2vl_lora_merged}"
export CLIP_DEVICE="${CLIP_DEVICE:-cpu}"

if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp >/dev/null 2>&1 || true
fi
pkill -f "llamafactory-cli api" >/dev/null 2>&1 || true
sleep 2

echo "[INFO] 重启推理 API (INFER_MODE=$INFER_MODE)..."
exec bash scripts/start_lf_api.sh
