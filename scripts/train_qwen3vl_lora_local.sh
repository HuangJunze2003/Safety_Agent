#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2-VL-7B-Instruct}"
MODELSCOPE_MODEL_ID="${MODELSCOPE_MODEL_ID:-$MODEL_ID}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/base-vl}"
HF_CACHE_DIR="$PROJECT_ROOT/.cache/huggingface"

mkdir -p "$MODEL_DIR" "$HF_CACHE_DIR" data/data_processed
cp -f configs/llamafactory/dataset_info.json data/data_processed/dataset_info.json

export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/llama/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llama/bin/llamafactory-cli}"

echo "[INFO] 使用 Python: $PYTHON_BIN"
echo "[INFO] 使用 LLaMA-Factory CLI: $LLAMAFACTORY_CLI"
echo "[INFO] 模型将下载/读取于: $MODEL_DIR"
echo "[INFO] HuggingFace 缓存目录: $HF_CACHE_DIR"
echo "[INFO] HuggingFace 模型ID: $MODEL_ID"
echo "[INFO] ModelScope 模型ID(回退): $MODELSCOPE_MODEL_ID"

$PYTHON_BIN - <<'PY'
import importlib.util
import subprocess
import sys

required = ["huggingface_hub", "llamafactory", "modelscope"]
for pkg in required:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY

if [ ! -x "$LLAMAFACTORY_CLI" ]; then
  echo "[ERROR] 未找到可执行的 LLaMA-Factory CLI: $LLAMAFACTORY_CLI"
  echo "[INFO] 可尝试执行: $PYTHON_BIN -m pip install llamafactory"
  exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[INFO] 本地模型不存在，开始下载到工程目录 models/..."
  $PYTHON_BIN - <<PY
import traceback

model_id = "$MODEL_ID"
ms_model_id = "$MODELSCOPE_MODEL_ID"
model_dir = "$MODEL_DIR"

def download_from_hf() -> bool:
  try:
    from huggingface_hub import snapshot_download
    snapshot_download(
      repo_id=model_id,
      local_dir=model_dir,
      local_dir_use_symlinks=False,
      resume_download=True,
    )
    print("model_download_done_hf")
    return True
  except Exception:
    print("[WARN] HuggingFace 下载失败，准备尝试 ModelScope 回退。")
    traceback.print_exc()
    return False

def download_from_modelscope() -> bool:
  try:
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(
      model_id=ms_model_id,
      local_dir=model_dir,
      revision="master",
    )
    print("model_download_done_modelscope")
    return True
  except Exception:
    print("[ERROR] ModelScope 下载也失败。")
    traceback.print_exc()
    return False

ok = download_from_hf() or download_from_modelscope()
if not ok:
  raise SystemExit(
    "无法联网下载模型。请手动将模型文件放到 ./models/base-vl （至少包含 config.json）后重试。"
  )
PY
fi

echo "[INFO] 启动 LoRA 微调..."
"$LLAMAFACTORY_CLI" train configs/llamafactory/qwen3vl_lora_local.yaml
