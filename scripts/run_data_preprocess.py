#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_processor.extractor import CaseExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行安全报告数据预处理并输出结构化元数据")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/data_raw/cases"),
        help="原始 Word/PDF 报告目录",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/data_processed"),
        help="预处理输出目录（将生成 images/ 与 cases_metadata.json）",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extractor = CaseExtractor(
        project_root=args.project_root,
        processed_dir=args.processed_dir,
    )
    cases = extractor.extract_all(input_dir=args.input_dir)

    metadata_path = args.project_root / args.processed_dir / "cases_metadata.json"
    manual_review_path = args.project_root / args.processed_dir / "manual_review_checklist.json"
    image_dir = args.project_root / args.processed_dir / "images"
    needs_manual_review = sum(1 for case in cases if bool(case.get("needs_manual_review")))

    summary = {
        "input_dir": str(args.input_dir),
        "processed_dir": str(args.processed_dir),
        "metadata_path": str(metadata_path),
        "manual_review_checklist_path": str(manual_review_path),
        "images_dir": str(image_dir),
        "total_cases": len(cases),
        "total_images": sum(len(case.get("images", [])) for case in cases),
        "needs_manual_review": needs_manual_review,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
