#!/usr/bin/env python3
"""Build a fixed offline evaluation set from cleaned SFT data."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "eval"
OUTPUT_CSV = OUTPUT_DIR / "eval_set.csv"
GUIDELINE_MD = OUTPUT_DIR / "label_guideline.md"
SOURCE_CANDIDATES = [
    ROOT / "data" / "data_processed" / "qwen3vl_sft.cleaned.with_image.jsonl",
    ROOT / "data" / "data_processed" / "qwen3vl_sft.cleaned.jsonl",
    ROOT / "data" / "data_processed" / "qwen3vl_sft.jsonl",
]
TARGET_SAMPLE_SIZE = 80
MIN_SAMPLE_SIZE = 50
RANDOM_SEED = 42


HAZARD_PATTERN = re.compile(r"【隐患名称】\s*(.*?)\s*(?:【风险等级】|$)", re.S)
LAW_PATTERN = re.compile(r"【法定依据（需引用具体条款）】\s*(.*?)\s*(?:【整改建议】|$)", re.S)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_field(pattern: re.Pattern[str], text: str, default: str = "") -> str:
    match = pattern.search(text or "")
    if not match:
        return default
    return normalize_text(match.group(1))


def load_records(source_jsonl: Path) -> list[dict]:
    records: list[dict] = []
    with source_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), {})
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), {})
            assistant_text = assistant_msg.get("content", "")
            records.append(
                {
                    "sample_id": item.get("id", ""),
                    "source_file": item.get("source_file", ""),
                    "image_path": (item.get("images") or [""])[0],
                    "query": normalize_text(user_msg.get("content", "")),
                    "gold_hazard_point": extract_field(HAZARD_PATTERN, assistant_text),
                    "gold_legal_clause": extract_field(LAW_PATTERN, assistant_text),
                    "gold_case_id": item.get("id", ""),
                }
            )
    return records


def write_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "source_file",
        "image_path",
        "query",
        "gold_hazard_point",
        "gold_legal_clause",
        "gold_case_id",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_guideline() -> None:
    content = """# Evaluation Label Guideline

## Scope
- Dataset: `data/eval/eval_set.csv`
- Size: fixed sample set (80 records)
- Gold fields:
  - `gold_hazard_point`: core hazard description in expected answer
  - `gold_legal_clause`: legal basis clause text
  - `gold_case_id`: case identifier used for retrieval verification

## Annotation Rules
1. Keep `gold_hazard_point` as concise hazard phrase from the assistant reference.
2. Keep `gold_legal_clause` as statute + clause text; preserve legal names.
3. `gold_case_id` must be stable and unique, copied from source item `id`.
4. If source field is missing, leave it empty and flag in manual review.

## Manual Review Checklist
- Hazard phrase is concrete and not generic.
- Legal clause includes specific legal source whenever available.
- Case ID maps to an existing source record.
- Image path is non-empty and file exists in dataset.
"""
    GUIDELINE_MD.write_text(content, encoding="utf-8")


def main() -> None:
    source_used: Path | None = None
    rows: list[dict] = []
    for source in SOURCE_CANDIDATES:
        if not source.exists():
            continue
        candidate_rows = load_records(source)
        if len(candidate_rows) >= MIN_SAMPLE_SIZE:
            source_used = source
            rows = candidate_rows
            break

    if source_used is None:
        counts = []
        for source in SOURCE_CANDIDATES:
            if source.exists():
                counts.append(f"{source.name}:{sum(1 for _ in source.open('r', encoding='utf-8'))}")
        raise ValueError(f"No source has >= {MIN_SAMPLE_SIZE} samples. Counts={', '.join(counts)}")

    sample_size = min(TARGET_SAMPLE_SIZE, len(rows))
    random.seed(RANDOM_SEED)
    selected = random.sample(rows, sample_size)
    write_csv(selected)
    write_guideline()
    print(f"Built eval set: {OUTPUT_CSV} ({len(selected)} rows) from {source_used.name}")
    print(f"Wrote guideline: {GUIDELINE_MD}")


if __name__ == "__main__":
    main()
