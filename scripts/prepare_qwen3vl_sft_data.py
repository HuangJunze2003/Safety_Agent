#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "你是一名国家注册安全工程师，擅长基于现场图片识别安全隐患，"
    "并结合《安全生产法》及地方规范给出可执行整改建议。"
)

USER_QUESTIONS = [
    "请分析该现场照片是否存在安全隐患，并给出法律依据。",
    "请识别图中主要风险点，并结合相关法条给出整改建议。",
    "请根据现场图像判断是否存在违规作业行为，并说明依据条款。",
]

ASSISTANT_TEMPLATE = """【隐患名称】\n{hazard_name}\n\n【风险等级】\n{risk_level}\n\n【法定依据（需引用具体条款）】\n{legal_basis}\n\n【整改建议】\n{suggestion}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 cases_metadata.json 转换为 Qwen 3.0-VL 微调 JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/data_processed/cases_metadata.json"),
        help="输入的 cases_metadata.json 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/data_processed/qwen3vl_sft.jsonl"),
        help="输出的微调 JSONL 路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于 user 问句采样",
    )
    return parser.parse_args()


def infer_risk_level(text: str) -> str:
    high_keywords = ["爆炸", "中毒", "窒息", "坍塌", "触电", "火灾", "有限空间", "高处坠落"]
    medium_keywords = ["防护", "警示", "违规", "未佩戴", "堆放", "通道"]

    if any(keyword in text for keyword in high_keywords):
        return "高风险"
    if any(keyword in text for keyword in medium_keywords):
        return "中风险"
    return "一般风险"


def build_record(case: dict[str, Any], question: str) -> dict[str, Any]:
    issue_text = str(case.get("issue_text", "")).strip()
    legal_basis = str(case.get("legal_basis", "")).strip()
    suggestion = str(case.get("suggestion", "")).strip() or "请按现行安全生产制度立即制定整改措施并闭环复查。"
    images = case.get("images", [])

    image_path = ""
    if isinstance(images, list) and images:
        first_image = images[0]
        if isinstance(first_image, dict):
            image_path = str(first_image.get("path", "")).strip()

    hazard_name = issue_text.split("；", 1)[0].split("。", 1)[0] if issue_text else "现场作业安全隐患"
    risk_level = infer_risk_level(issue_text + "\n" + suggestion)

    user_content = f"<image>\n{question}"
    assistant_content = ASSISTANT_TEMPLATE.format(
        hazard_name=hazard_name,
        risk_level=risk_level,
        legal_basis=legal_basis,
        suggestion=suggestion,
    )

    return {
        "id": case.get("id", ""),
        "source_file": case.get("source_file", ""),
        "images": [image_path] if image_path else [],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    cases = raw.get("cases", []) if isinstance(raw, dict) else []

    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for case in cases:
            legal_basis = str(case.get("legal_basis", "")).strip()
            if not legal_basis:
                skipped += 1
                continue
            question = random.choice(USER_QUESTIONS)
            record = build_record(case, question)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total_cases": len(cases),
        "kept_for_sft": kept,
        "filtered_missing_legal_basis": skipped,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
