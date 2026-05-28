#!/usr/bin/env bash
# 此脚本用于一键启动所有相关服务：后端模型 API 与 前端 Gradio 页面
set -euo pipefail

# 1. 切换到项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 2. 定义环境变量防止踩坑
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/huggingface_cache"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# 推理权重（Base + LoRA）；可改为 outputs/qwen3vl_lora_loss_curve_run 等
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/models/base-vl}"
export ADAPTER_PATH="${ADAPTER_PATH:-$PROJECT_ROOT/outputs/qwen3vl_lora_long_fine}"
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/qwen2vl_lora_merged}"
export INFER_MODE="${INFER_MODE:-merged}"
export QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-qwen3vl_lora_merged}"
export QWEN_API_BASE="${QWEN_API_BASE:-http://127.0.0.1:8000/v1}"
# 检索模型放 CPU，为推理 API 留出 GPU 显存
export CLIP_DEVICE="${CLIP_DEVICE:-cpu}"

# 统一锁定到 llama conda 环境的 Python，避免随当前 shell 漂移
LLAMA_PYTHON="${LLAMA_PYTHON:-/root/miniconda3/envs/llama/bin/python}"
if [ ! -x "$LLAMA_PYTHON" ]; then
    echo "[ERROR] 未找到 llama 环境的 Python 解释器: $LLAMA_PYTHON" >&2
    echo "[ERROR] 请确认 conda 环境 'llama' 已正确创建。" >&2
    exit 1
fi

# 用于保证关闭脚本时，能杀掉同时启动的两个子进程
cleanup() {
    echo ""
    echo "[INFO] 正在关闭所有服务..."
    # 杀掉这三个进程
    kill $API_PID $INGEST_PID $WEBUI_PID 2>/dev/null || true
    echo "[INFO] 服务已完全关闭 👋"
    exit 0
}

# 捕获 Ctrl+C 的发出的中断信号，如果捕捉到则执行 cleanup 清理函数
trap cleanup SIGINT SIGTERM

# 清理占用 8000 的僵死推理进程（常见于 LoRA merge OOM 后端口未释放）
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp >/dev/null 2>&1 || true
fi
pkill -f "llamafactory-cli api" >/dev/null 2>&1 || true
sleep 2

# =========================================================================
# 步骤 1： 启动后端大语言模型 API 服务 (Qwen-VL Inference Backend)
# =========================================================================
echo "[启动阶段 1/3] 正在拉起大模型后端 API 服务..."
bash scripts/start_lf_api.sh &
API_PID=$!

# =========================================================================
# 步骤 2： 启动数据入库与文档解析 API 服务 (Ingestion Service)
# =========================================================================
echo "[启动阶段 2/3] 正在拉起数据入库与文档解析 API 服务 (Port 8001)..."
"$LLAMA_PYTHON" scripts/run_ingestion_api.py &
INGEST_PID=$!

echo "[INFO] 等待模型 API 就绪（最多约 3 分钟）..."
API_READY=false
for _ in $(seq 1 90); do
    if curl -sf --max-time 3 "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
        API_READY=true
        break
    fi
    sleep 2
done
if [ "$API_READY" != true ]; then
    echo "[ERROR] 模型 API (8000) 未就绪。请检查上方是否出现 CUDA OOM 或 llamafactory-cli 报错。" >&2
    echo "[ERROR] 可尝试: export INFER_MODE=merged && bash scripts/start_lf_api.sh" >&2
    exit 1
fi
echo "[INFO] 模型 API 已就绪: http://127.0.0.1:8000"
echo "[INFO] 数据处理接口运行在 http://127.0.0.1:8001。"

# =========================================================================
# 步骤 3： 启动前端 Gradio 智能体面板
# =========================================================================
echo ""
echo "[启动阶段 3/3] 正在拉起 Agent WebUI 前端与智能体大脑..."
"$LLAMA_PYTHON" scripts/run_agent_webui.py &
WEBUI_PID=$!

echo ""
echo "================================================================="
echo "🎉 恭喜！所有服务已启动完毕！"
echo "👉 模型推理服务: http://127.0.0.1:8000"
echo "👉 后端入库服务: http://127.0.0.1:8001"
echo "👉 WebUI界面运行在: http://0.0.0.0:7860"
echo ""
echo "📝 若需停止服务，请按 [Ctrl + C] 。这会自动关闭前后端进程。"
echo "================================================================="

# 挂起主进程，等待子进程退出或者人为按 Ctrl+C 中断
wait $API_PID $INGEST_PID $WEBUI_PID

