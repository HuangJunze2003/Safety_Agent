#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行安全生产 Agent Demo")
    parser.add_argument("--image", type=Path, default=None, help="待分析图片路径；不传则自动选择 data/data_processed/images 下首张图片")
    parser.add_argument(
        "--question",
        type=str,
        default="请分析该现场照片是否存在安全隐患，并给出法律依据。",
        help="用户问题",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/data_processed/agent_demo_result.json"),
        help="结果输出 JSON 路径",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3vl_lora_local",
        help="当前推理服务中的模型名（默认接入本地训练产物名）",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="跳过 CLIP 重建索引（默认每次重建）",
    )
    return parser.parse_args()


def resolve_image_path(image_arg: Path | None, project_root: Path) -> Path:
    if image_arg is not None:
        image_path = image_arg if image_arg.is_absolute() else (project_root / image_arg)
        image_path = image_path.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        return image_path

    images_dir = (project_root / "data/data_processed/images").resolve()
    candidates = sorted(
        [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"未找到可用图片，请先执行数据预处理并确认目录存在图片: {images_dir}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    try:
        from agent.workflow import build_agent_from_env
    except Exception as exc:
        raise RuntimeError(
            "无法加载 Agent 工作流，请先安装依赖（至少需要 langchain、chromadb、transformers）"
        ) from exc

    if "QWEN_MODEL_NAME" not in os.environ:
        os.environ["QWEN_MODEL_NAME"] = args.model_name

    agent = build_agent_from_env()
    image_path = resolve_image_path(args.image, project_root)

    indexed_images = None
    if not args.skip_index:
        indexed_images = agent.image_retriever.index_images()

    result = agent.analyze(image_path=image_path, question=args.question)
    result["indexing"] = {
        "reindexed": not args.skip_index,
        "indexed_images": indexed_images,
    }
    result["effective_model_name"] = os.getenv("QWEN_MODEL_NAME", args.model_name)
    result["effective_image_path"] = str(image_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_name": result["effective_model_name"],
                "image_path": result["effective_image_path"],
                "indexed_images": indexed_images,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
