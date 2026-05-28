#!/usr/bin/env python3
"""Concatenate thesis sources into docs/thesis/thesis_all_in_one.md (single-file draft).

顺序与 docs/thesis/thesis_main.md「正文来源」一致：
摘要 → 第1–6章 → 参考文献。
"""

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

> **维护说明**：本文件由 `scripts/build_thesis_all_in_one.py` 拼接生成，源文件依次为：  
> `abstract_final.md`、`chapter1_introduction.md` … `chapter6_conclusion.md`、`references_thesis_master.md`（第4章正文源为 `chapter4_implementation.md`，与 `thesis_main.md` 一致）。  
> 修改请以分章与摘要为准，在仓库根目录执行：  
> `python scripts/build_thesis_all_in_one.py`  
> 请勿对本文件正文做大规模手工修改，以免与分章稿漂移。

---

"""


def main() -> None:
    chunks = [HEADER]
    for i, path in enumerate(SECTIONS):
        if not path.is_file():
            raise FileNotFoundError(f"Missing thesis source: {path}")
        text = path.read_text(encoding="utf-8").strip()
        chunks.append(text)
        if i < len(SECTIONS) - 1:
            chunks.append("\n\n---\n\n")
    OUT.write_text("".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
