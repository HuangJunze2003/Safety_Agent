#!/usr/bin/env python3
"""Concatenate thesis sources into docs/thesis/thesis_all_in_one.md (single-file draft)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "docs" / "thesis"
OUT = THESIS / "thesis_all_in_one.md"

SECTIONS = [
    THESIS / "abstract_final.md",
    THESIS / "chapter1_introduction.md",
    THESIS / "chapter2_related_work.md",
    THESIS / "chapter3_system_design.md",
    THESIS / "chapter4_implementation.md",
    THESIS / "chapter5_experiments.md",
    THESIS / "chapter6_conclusion.md",
    THESIS / "references_thesis_master.md",
]

HEADER = """# 一体化论文稿（自动生成）

> **维护说明**：本文件由 `scripts/build_thesis_all_in_one.py` 从 `abstract_final.md`、各 `chapter*.md`、`references_thesis_master.md` **拼接生成**。修改请以分章与摘要源文件为准，保存后在本仓库根目录执行：  
> `python scripts/build_thesis_all_in_one.py`  
> 请勿手工大规模编辑本文件正文，以免与分章稿漂移。

---

"""


def main() -> None:
    chunks = [HEADER]
    for i, path in enumerate(SECTIONS):
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8").strip()
        chunks.append(text)
        if i < len(SECTIONS) - 1:
            chunks.append("\n\n---\n\n")
    OUT.write_text("".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
