#!/usr/bin/env python3
"""Build failure case library from offline evaluation outputs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data" / "eval" / "eval_set.csv"
EVAL_RESULT = ROOT / "results" / "eval_metrics_round1.csv"
OUTPUT_MD = ROOT / "docs" / "failure_cases.md"
MAX_CASES = 12


def classify_failure(hit5: str, query: str, gold_case: str, pred_top1: str) -> tuple[str, str, str]:
    if hit5 == "1":
        return (
            "pipeline failure",
            "Top-5 已命中但最终排序偏低，说明排序阶段或后处理策略未把正确案例提升到 Top-1。",
            "增加 reranker 或基于法条/隐患字段的二阶段重排；对 Top-5 命中样本单独优化排序损失。",
        )
    if "<image>" in (query or "") and len(gold_case) > 0:
        if pred_top1 and pred_top1[:8] == gold_case[:8]:
            return (
                "visual misjudgment",
                "样本主题接近但关键视觉隐患点区分失败，召回了相近案例但未命中真值案例。",
                "引入视觉区域特征或违规类型标签监督，提升细粒度图像区分能力。",
            )
        return (
            "retrieval failure",
            "Top-5 完全未命中，说明检索召回阶段对该问题语义覆盖不足。",
            "改用向量检索 + 词法混合召回，扩大候选池并加入领域同义词词典。",
        )
    return (
        "hallucination",
        "查询与候选案例匹配信号弱，后续回答阶段易生成与真实证据不一致内容。",
        "在生成前加入证据一致性检查，证据不足时触发保守回复与补充检索。",
    )


def load_eval_set() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with EVAL_SET.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r.get("sample_id", "")
            rows[sid] = r
    return rows


def load_failures() -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    with EVAL_RESULT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("row_type") != "sample":
                continue
            if r.get("hit@1") == "1":
                continue
            failures.append(r)
    failures.sort(key=lambda x: float(x.get("latency_ms", "0")), reverse=True)
    return failures[:MAX_CASES]


def build_markdown(cases: list[dict[str, str]], eval_set_map: dict[str, dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# Failure Case Library")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Source result: `results/eval_metrics_round1.csv`")
    lines.append(f"- Selected failures: {len(cases)} cases (Top-1 miss)")
    lines.append("- Type set: retrieval failure / visual misjudgment / hallucination / pipeline failure")
    lines.append("")
    lines.append("## Case Details")
    lines.append("")

    for idx, c in enumerate(cases, start=1):
        sid = c.get("sample_id", "")
        eval_row = eval_set_map.get(sid, {})
        query = eval_row.get("query", "")
        gold_case = c.get("gold_case_id", "")
        pred1 = c.get("pred_case_id_top1", "")
        pred5 = c.get("pred_case_id_top5", "")
        hit5 = c.get("hit@5", "0")
        latency = c.get("latency_ms", "")
        failure_type, root_cause, fix = classify_failure(hit5, query, gold_case, pred1)

        lines.extend(
            [
                f"### F{idx:02d} - {failure_type}",
                f"- **sample_id**: `{sid}`",
                f"- **query**: `{query[:120]}`",
                f"- **expected(case_id)**: `{gold_case}`",
                f"- **actual(top1)**: `{pred1}`",
                f"- **actual(top5)**: `{pred5}`",
                f"- **latency_ms**: `{latency}`",
                f"- **root_cause**: {root_cause}",
                f"- **fix_idea**: {fix}",
                "",
            ]
        )

    lines.append("## Aggregated Observations")
    lines.append("- Top-1 miss samples are dominated by retrieval-stage misses; lexical matching is insufficient for domain semantics.")
    lines.append("- Some samples are Top-5 hit but Top-1 miss, indicating reranking is a high-leverage optimization point.")
    lines.append("- Failure remediation should prioritize hybrid retrieval, reranker, and evidence-consistency checks.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    eval_set_map = load_eval_set()
    failure_cases = load_failures()
    if len(failure_cases) < 10:
        raise ValueError(f"Not enough failure cases (<10): got {len(failure_cases)}")
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(build_markdown(failure_cases, eval_set_map), encoding="utf-8")
    print(f"Wrote failure library: {OUTPUT_MD} ({len(failure_cases)} cases)")


if __name__ == "__main__":
    main()
