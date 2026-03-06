# 安全生产管理智能体 (Safety_Agent)

基于多模态大模型（Qwen-VL）的安全生产管理智能体，用于自动解析、检索并回答安全生产中的图文合规性问题。

## 🚀 项目特性
- **多模态理解**：利用 Qwen2-VL/Qwen3-VL 进行安全生产现场图片的合规性审核。
- **混合检索**：结合 CLIP 图像检索与 ChromaDB 法律法规检索。
- **LoRA 微调**：集成 LLaMA-Factory 训练脚本，支持针对特定领域的微调与合并导出。
- **全流程覆盖**：包含数据清洗、数据集构建、模型调优、RAG 检索及 Web 交互。

## 📂 目录结构

```text
├── configs/             # LLaMA-Factory 训练/导出配置文件
├── data/                # 数据目录 (Git 仅保留结构)
│   ├── data_processed/  # 解析后的 JSONL 数据与图像特征
│   └── images/          # 现场勘查图片
├── models/              # 本地模型目录 (已忽略)
├── outputs/             # 训练输出/合并模型 (已忽略)
├── prompts/             # 系统角色与 Prompt 模板
├── scripts/             # 工具脚本集 (预处理、训练、运行)
├── src/                 # 核心代码
│   ├── agent/           # 智能体逻辑与工作流
│   ├── data_processor/  # PDF/Word 提取与清洗
│   ├── evaluation/      # 模型评估脚本
│   └── retriever/       # CLIP 与词法检索引擎
└── web-service-app/     # 前端交互 Web 应用
```

## 🛠️ 快速上手

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据处理与检索构建
```bash
# 执行数据预处理
python scripts/run_data_preprocess.py
# 构建法规向量库
python scripts/add_laws_to_db.py
```

### 3. 模型训练 (LoRA)
```bash
bash scripts/train_qwen3vl_lora.sh
```

### 4. 运行应用
```bash
# 启动 API 与 WebUI
bash scripts/start_all_services.sh
```

## 📜 许可证
本项目遵循 MIT 开源协议。
