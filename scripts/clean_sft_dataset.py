#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
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

ASSISTANT_TEMPLATE = """【隐患名称】
{hazard_name}

【风险等级】
{risk_level}

【法定依据（需引用具体条款）】
{legal_basis}

【整改建议】
{suggestion}"""

NOISE_PATTERNS = [
    r"^注[:：]",
    r"^附件\d+",
    r"^整改前照片$",
    r"^整改后照片$",
    r"^现场检查照片[:：]?$",
    r"^现场照片[:：]?$",
    r"^图[一二三四五六七八九十0-9]+$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗安全隐患数据并生成高质量 Qwen3-VL 微调集")
    parser.add_argument("--input", type=Path, default=Path("data/data_processed/cases_metadata.json"))
    parser.add_argument(
        "--clean-metadata-output",
        type=Path,
        default=Path("data/data_processed/cases_metadata.cleaned.json"),
    )
    parser.add_argument(
        "--clean-jsonl-output",
        type=Path,
        default=Path("data/data_processed/qwen3vl_sft.cleaned.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-image", action="store_true", help="仅保留有可用图片样本")
    return parser.parse_args()


def infer_risk_level(text: str) -> str:
    high_keywords = ["爆炸", "中毒", "窒息", "坍塌", "触电", "火灾", "有限空间", "高处坠落"]
    medium_keywords = ["防护", "警示", "违规", "未佩戴", "堆放", "通道"]
    if any(keyword in text for keyword in high_keywords):
        return "高风险"
    if any(keyword in text for keyword in medium_keywords):
        return "中风险"
    return "一般风险"


def strip_noise_lines(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    kept: list[str] = []
    for line in lines:
        if any(re.search(pattern, line) for pattern in NOISE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def looks_valid_legal(text: str) -> bool:
    text = strip_noise_lines(text)
    if len(text) < 10:
        return False
    return bool(
        re.search(r"《.+?》", text)
        or re.search(r"第.{0,8}条", text)
        or re.search(r"\bGB/?T?\s*\d+", text, flags=re.I)
        or any(token in text for token in ["安全生产法", "管理规定", "条例", "导则", "规范"])
    )


def looks_valid_suggestion(text: str) -> bool:
    text = strip_noise_lines(text)
    if len(text) < 6:
        return False
    return bool(re.search(r"整改|更换|设置|完善|维护|检查|加装|隔离|警示", text))


def first_valid_image(case: dict[str, Any]) -> str:
    images = case.get("images", [])
    if not isinstance(images, list):
        return ""
    for item in images:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("path", "")).strip()
        if image_path and Path(image_path).exists():
            return image_path
    return ""


def clean_case(case: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(case)
    cleaned["issue_text"] = strip_noise_lines(cleaned.get("issue_text", ""))
    cleaned["legal_basis"] = strip_noise_lines(cleaned.get("legal_basis", ""))
    cleaned["suggestion"] = strip_noise_lines(cleaned.get("suggestion", ""))
    return cleaned


def case_to_jsonl_record(case: dict[str, Any], question: str) -> dict[str, Any]:
    issue_text = str(case.get("issue_text", "")).strip()
    legal_basis = str(case.get("legal_basis", "")).strip()
    suggestion = str(case.get("suggestion", "")).strip()
    image_path = first_valid_image(case)

    hazard_name = issue_text.split("；", 1)[0].split("。", 1)[0] if issue_text else "现场作业安全隐患"
    risk_level = infer_risk_level(issue_text + "\n" + suggestion)

    assistant = ASSISTANT_TEMPLATE.format(
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
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": assistant},
        ],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    cases = raw.get("cases", []) if isinstance(raw, dict) else []

    cleaned_cases: list[dict[str, Any]] = []
    rejected = {
        "missing_or_invalid_legal": 0,
        "missing_or_invalid_suggestion": 0,
        "missing_image_when_required": 0,
    }

    for case in cases:
        c = clean_case(case)
        if not looks_valid_legal(c.get("legal_basis", "")):
            rejected["missing_or_invalid_legal"] += 1
            continue
        if not looks_valid_suggestion(c.get("suggestion", "")):
            rejected["missing_or_invalid_suggestion"] += 1
            continue
        if args.require_image and not first_valid_image(c):
            rejected["missing_image_when_required"] += 1
            continue
        cleaned_cases.append(c)

    args.clean_metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.clean_jsonl_output.parent.mkdir(parents=True, exist_ok=True)

    meta_payload = {
        "generated_at": raw.get("generated_at", ""),
        "total_cases": len(cleaned_cases),
        "source_total_cases": len(cases),
        "rejected": rejected,
        "cases": cleaned_cases,
    }
    args.clean_metadata_output.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.clean_jsonl_output.open("w", encoding="utf-8") as fout:
        for case in cleaned_cases:
            question = random.choice(USER_QUESTIONS)
            record = case_to_jsonl_record(case, question)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_total_cases": len(cases),
                "cleaned_cases": len(cleaned_cases),
                "rejected": rejected,
                "clean_metadata": str(args.clean_metadata_output),
                "clean_jsonl": str(args.clean_jsonl_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
