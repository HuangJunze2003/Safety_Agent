# 安全生产管理智能体（Qwen-VL）

本项目用于构建基于多模态大模型（Qwen-VL）的安全生产管理智能体。

## 目录结构

```text
data/
  data_raw/         # 存放原始 Word/PDF 报告
  data_processed/   # 存放提取出的图文 JSON
src/
  data_processor/   # 数据解析核心
  retriever/        # CLIP 图像特征检索
  agent/            # LangChain 智能体工作流
```
