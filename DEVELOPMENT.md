# 安全生产管理智能体 (Safety_Agent) 开发文档

本项目是一个基于 **Qwen3-VL** 多模态大模型与 **RAG (检索增强生成)** 技术构建的智能化安全生产管理系统。它通过结合视觉识别、相似案例检索以及法律法规检索，为安全生产现场的隐患排查提供专业、合规的决策支持。

---

## 🏗️ 1. 系统架构与技术栈

### 核心技术栈
- **多模态模型**: Qwen-VL 系列 (Qwen2-VL/Qwen3-VL)
- **微调框架**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (用于 LoRA 微调与模型合并)
- **向量数据库**: [ChromaDB](https://www.trychroma.com/) (存储法律法规与图像特征)
- **检索模型**: CLIP (图像检索), BGE-M3/Bge-small (文本向量化)
- **开发框架**: LangChain, Python-Docx, PyMuPDF (数据处理)
- **前端交互**: Gradio (WebUI), TypeScript/React (Web App)

### 模块结构
- `src/agent/`: 智能体核心逻辑，包含工作流调度与 CoT（思维链）推理。
- `src/retriever/`: 混合检索引擎，负责图像相似度匹配与法律条文检索。
- `src/data_processor/`: 自动化数据提取与清洗模块。
- `configs/`: 存放 LLaMA-Factory 训练、导出配置。
- `scripts/`: 全流程操作脚本（预处理 -> 训练 -> 部署）。

---

## 🛠️ 2. 环境搭建

### 2.1 基础环境
- **Python**: 3.10+
- **CUDA**: 11.8+ (推荐 12.1+)
- **OS**: Linux (推荐) / Windows WSL2

### 2.2 安装依赖
```bash
# 建议创建虚拟环境
conda create -n safety_agent python=3.10
conda activate safety_agent

# 安装项目依赖
pip install -r requirements.txt

# 如需训练模型，需安装 LLaMA-Factory
# git clone https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory && pip install -e .[metrics,modelscope,qwen]
```

---

## 📂 3. 数据流程 (Data Pipeline)

### 3.1 原始数据预处理
系统支持从 PDF 规范、Word 文档中提取结构化知识。
```bash
# 执行数据解析逻辑，提取图像与文本
python scripts/run_data_preprocess.py
```

### 3.2 向量库构建
将解析后的数据注入向量数据库。
```bash
# 构建法律法规向量库
python scripts/add_laws_to_db.py
```

---

## ⚡ 4. 模型训练 (LoRA Fine-tuning)

本项目采用 LoRA 技术在特定安全场景下提升模型表现。

### 4.1 配置文件说明
- `configs/llamafactory/qwen3vl_lora.yaml`: 训练超参数设置（学习率、Batch Size、Epoch 等）。
- `configs/llamafactory/qwen2vl_lora_export.yaml`: 模型转换与合并配置。

### 4.2 执行训练
```bash
# 启动训练任务
bash scripts/train_qwen3vl_lora.sh
```

### 4.3 模型合并
```bash
# 将微调权重合并到 Base Model 以便部署
bash scripts/export_lora_model.sh
```

---

## 🚀 5. 部署与运行

### 5.1 一键启动
系统提供了一键启动脚本，同时拉起后端模型 API 与前端 Gradio 界面。
```bash
bash scripts/start_all_services.sh
```

### 5.2 核心组件访问
- **后端模型 API**: `http://127.0.0.1:8000` (OpenAI 兼容接口)
- **前端 WebUI**: `http://127.0.0.1:7860`

---

## 🧠 6. 核心逻辑分析：SafetyProductionAgent

`SafetyProductionAgent` 是系统的大脑，其工作流逻辑如下：

1. **多路输入**: 接收“现场图像” + “用户提问”。
2. **意图路由**: 采用“LLM 优先 + 规则兜底”的三分类策略，判断为 `greeting` / `legal_only` / `hazard_analysis`。
3. **混合检索 (Hybrid RAG)**:
   - **CLIP Retrieval**: 基于图片特征在历史案例库中寻找相似的违规照片。
   - **Chroma Retrieval**: 基于文本语义检索相关的法律条文（如《安全生产法》）。
4. **思维链推理 (CoT)**:
   - 视觉特征提取 -> 法规契合点分析 -> 风险等级判定 -> 整改措施建议。
5. **JSON 结构化输出**: 强制模型输出预定义格式，便于下游系统集成。

### 6.1 意图识别实现位置
- 核心文件：`src/agent/workflow.py`
   - `detect_intent(question, has_image)`: 对外统一入口。
   - `_detect_intent_with_llm(...)`: 调用推理服务做意图分类。
   - `_detect_intent(...)`: 正则规则兜底分类。
- Web 入口：`scripts/run_agent_webui.py`
   - 在进入 `agent.analyze(...)` 之前先判意图。
   - 仅当 `intent == hazard_analysis` 且无图片时提示上传现场图。

### 6.2 当前意图分类约定
- `greeting`: 问候、身份、能力、闲聊。
- `legal_only`: 法规检索、条文解释、处罚依据、流程要求。
- `hazard_analysis`: 现场隐患分析、风险研判、整改建议（尤其在有图时）。

### 6.3 已知问题与优化方向
- 已知问题：极短闲聊（如“今天天气怎么样”）在 LLM 分类失败并回退规则时，可能误分到 `hazard_analysis`。
- 建议优化：
   - 扩展 greeting 规则词典（天气、寒暄、口语缩写、脏词问候）。
   - 在无图场景对短文本增加 `greeting` 优先兜底。
   - 在 WebUI 增加 intent 调试显示，便于标注误判样本。
   - 增加离线评测集（意图样本 JSONL）做回归测试。

---

## 📅 7. 维护与贡献
- **数据更新**: 若法律法规更新，请重新运行 `scripts/add_laws_to_db.py`。
- **模型演进**: 可将用户反馈的错误案例加入 `data/qwen3vl_sft.jsonl` 进行二阶段微调。
