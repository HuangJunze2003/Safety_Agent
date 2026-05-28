#!/usr/bin/env python3
"""Run baseline comparison and ablation on eval_set."""

from __future__ import annotations

import csv
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.relaxed_metrics import aggregate_relaxed_metrics  # noqa: E402
from evaluation.retrieval_metrics import DEFAULT_K, aggregate_ranking_metrics  # noqa: E402

EVAL_SET = ROOT / "data" / "eval" / "eval_set.csv"
RESULT_DIR = ROOT / "results"
COMPARISON_OUT = RESULT_DIR / "exp_comparison.csv"
ABLATION_OUT = RESULT_DIR / "ablation.csv"
RELAXED_COMPARISON_OUT = RESULT_DIR / "exp_comparison_relaxed.csv"
RELAXED_ABLATION_OUT = RESULT_DIR / "ablation_relaxed.csv"
TOP_K = DEFAULT_K
SEED = 42

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class Sample:
    sample_id: str
    source_file: str
    query: str
    hazard: str
    legal: str
    gold_case_id: str


def tokenize(text: str) -> set[str]:
    return {x.lower() for x in TOKEN_PATTERN.findall(text or "")}


def score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens.intersection(doc_tokens)) / len(query_tokens)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    return arr[int((len(arr) - 1) * q)]


def read_eval(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Sample(
                    sample_id=r.get("sample_id", ""),
                    source_file=r.get("source_file", ""),
                    query=r.get("query", ""),
                    hazard=r.get("gold_hazard_point", ""),
                    legal=r.get("gold_legal_clause", ""),
                    gold_case_id=r.get("gold_case_id", ""),
                )
            )
    return rows


def doc_text(sample: Sample, mode: str) -> str:
    if mode == "legal_only_rag":
        return sample.legal
    if mode == "case_only_rag":
        return " ".join([sample.hazard, sample.source_file, sample.sample_id])
    if mode == "dual_retrieval_agent":
        return " ".join([sample.hazard, sample.legal, sample.source_file, sample.sample_id])
    if mode == "dual_no_source_ablation":
        return " ".join([sample.hazard, sample.legal])
    return ""


def retrieve_ids(query: str, corpus: list[Sample], mode: str, k: int, rng: random.Random) -> list[str]:
    if mode == "no_rag_baseline":
        ids = [s.sample_id for s in corpus]
        rng.shuffle(ids)
        return ids[:k]

    q_tokens = tokenize(query)
    scored: list[tuple[float, str]] = []
    for s in corpus:
        scored.append((score(q_tokens, tokenize(doc_text(s, mode))), s.sample_id))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sid for _, sid in scored[:k]]


def corpus_index(samples: list[Sample]) -> dict[str, dict[str, str]]:
    return {
        s.sample_id: {"hazard": s.hazard, "legal": s.legal, "source_file": s.source_file}
        for s in samples
    }


def eval_mode(samples: list[Sample], mode: str) -> dict[str, float | str]:
    rng = random.Random(SEED)
    latencies: list[float] = []
    pred_at_5: list[list[str]] = []
    gold_ids: list[str] = []

    for s in samples:
        t0 = time.perf_counter()
        pred = retrieve_ids(s.query, samples, mode, 5, rng)
        latencies.append((time.perf_counter() - t0) * 1000)
        pred_at_5.append(pred)
        gold_ids.append(s.gold_case_id)

    ranking = aggregate_ranking_metrics(pred_at_5, gold_ids, TOP_K)
    relaxed = aggregate_relaxed_metrics(pred_at_5, gold_ids, corpus_index(samples), TOP_K)
    return {
        "setup": mode,
        "num_samples": len(samples),
        **ranking,
        **relaxed,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ms": percentile(latencies, 0.95),
    }


STRICT_FIELDS = [
    "mrr",
    "hit@1",
    "hit@3",
    "hit@5",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "recall@1",
    "recall@3",
    "recall@5",
]
RELAXED_FIELDS = [
    "hazard_mrr",
    "hazard_hit@1",
    "hazard_hit@3",
    "hazard_hit@5",
    "hazard_ndcg@3",
    "legal_mrr",
    "legal_hit@1",
    "legal_hit@3",
    "legal_hit@5",
    "legal_ndcg@3",
    "evidence_mrr",
    "evidence_hit@1",
    "evidence_hit@3",
    "evidence_hit@5",
    "evidence_ndcg@3",
]
METRIC_FIELDS = ["setup", "num_samples", *STRICT_FIELDS, *RELAXED_FIELDS, "latency_p50_ms", "latency_p95_ms"]


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    float_keys = {f for f in METRIC_FIELDS if f not in {"setup", "num_samples"}}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for r in rows:
            row_out: dict[str, str | int] = {
                "setup": str(r["setup"]),
                "num_samples": int(r["num_samples"]),
            }
            for key in float_keys:
                row_out[key] = f"{float(r[key]):.4f}"
            writer.writerow(row_out)


def main() -> None:
    samples = read_eval(EVAL_SET)
    if not samples:
        raise ValueError(f"Eval set missing or empty: {EVAL_SET}")

    comparison_setups = [
        "no_rag_baseline",
        "legal_only_rag",
        "case_only_rag",
        "dual_retrieval_agent",
    ]
    ablation_setups = [
        "dual_retrieval_agent",
        "dual_no_source_ablation",
    ]

    comparison_rows = [eval_mode(samples, s) for s in comparison_setups]
    ablation_rows = [eval_mode(samples, s) for s in ablation_setups]

    write_csv(COMPARISON_OUT, comparison_rows)
    write_csv(ABLATION_OUT, ablation_rows)
    write_relaxed_summary(RELAXED_COMPARISON_OUT, comparison_rows)
    write_relaxed_summary(RELAXED_ABLATION_OUT, ablation_rows)

    print(f"Wrote comparison: {COMPARISON_OUT}")
    print(f"Wrote ablation: {ABLATION_OUT}")
    print(f"Wrote relaxed summary: {RELAXED_COMPARISON_OUT}")
    for row in comparison_rows:
        print(
            f"  {row['setup']}: strict MRR={row['mrr']:.4f} Hit@1={row['hit@1']:.4f} | "
            f"evidence Hit@1={row['evidence_hit@1']:.4f} Hit@3={row['evidence_hit@3']:.4f}"
        )


def write_relaxed_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Compact table for thesis: strict vs evidence-relaxed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setup",
        "num_samples",
        "mrr",
        "hit@1",
        "hit@3",
        "ndcg@3",
        "evidence_mrr",
        "evidence_hit@1",
        "evidence_hit@3",
        "evidence_ndcg@3",
        "hazard_hit@3",
        "legal_hit@3",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    float_keys = {f for f in fields if f not in {"setup", "num_samples"}}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row_out: dict[str, str | int] = {
                "setup": str(r["setup"]),
                "num_samples": int(r["num_samples"]),
            }
            for key in float_keys:
                row_out[key] = f"{float(r[key]):.4f}"
            writer.writerow(row_out)


if __name__ == "__main__":
    main()
