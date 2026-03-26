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
python scripts/run_ingestion_api.py &
INGEST_PID=$!

echo "[INFO] 等待 API 接口预热启动 (约 10-15 秒)........."
# 等待15s确信服务启动
sleep 15
echo "[INFO] 如果上方出现了 Uvicorn running on http://127.0.0.1:8000 说明模型就绪。"
echo "[INFO] 数据处理接口运行在 http://127.0.0.1:8001。"

# =========================================================================
# 步骤 3： 启动前端 Gradio 智能体面板
# =========================================================================
echo ""
echo "[启动阶段 3/3] 正在拉起 Agent WebUI 前端与智能体大脑..."
/root/miniconda3/envs/llama/bin/python scripts/run_agent_webui.py &
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

