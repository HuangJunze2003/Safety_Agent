#!/usr/bin/env python3
"""Offline evaluator for retrieval quality and latency."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.retrieval_metrics import (  # noqa: E402
    DEFAULT_K,
    aggregate_ranking_metrics,
    rank_of_gold,
)

DEFAULT_EVAL_SET = ROOT / "data" / "eval" / "eval_set.csv"
DEFAULT_OUTPUT = ROOT / "results" / "eval_metrics_round1.csv"
TOP_K = DEFAULT_K


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class EvalSample:
    sample_id: str
    query: str
    gold_case_id: str
    gold_hazard_point: str
    gold_legal_clause: str
    source_file: str

    @property
    def retrieval_text(self) -> str:
        return " ".join(
            [
                self.sample_id,
                self.source_file,
                self.gold_hazard_point,
                self.gold_legal_clause,
            ]
        )


def tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_PATTERN.findall(text or "")}


def overlap_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    inter = len(query_tokens.intersection(doc_tokens))
    return inter / len(query_tokens)


def load_eval_set(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                EvalSample(
                    sample_id=row.get("sample_id", ""),
                    query=row.get("query", ""),
                    gold_case_id=row.get("gold_case_id", ""),
                    gold_hazard_point=row.get("gold_hazard_point", ""),
                    gold_legal_clause=row.get("gold_legal_clause", ""),
                    source_file=row.get("source_file", ""),
                )
            )
    return samples


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def topk_case_ids(
    query: str,
    corpus: list[EvalSample],
    token_cache: dict[str, set[str]],
    k: int,
) -> list[str]:
    query_tokens = tokenize(query)
    scored: list[tuple[float, str]] = []
    for item in corpus:
        doc_tokens = token_cache[item.sample_id]
        scored.append((overlap_score(query_tokens, doc_tokens), item.sample_id))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [case_id for _, case_id in scored[:k]]


def run_eval(samples: list[EvalSample]) -> tuple[dict[str, float], list[dict[str, str]]]:
    token_cache: dict[str, set[str]] = {s.sample_id: tokenize(s.retrieval_text) for s in samples}
    latencies_ms: list[float] = []
    pred_at_5: list[list[str]] = []
    gold_ids: list[str] = []
    rows: list[dict[str, str]] = []

    for sample in samples:
        t0 = time.perf_counter()
        pred_5 = topk_case_ids(sample.query, samples, token_cache, 5)
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies_ms.append(latency_ms)
        pred_at_5.append(pred_5)
        gold_ids.append(sample.gold_case_id)
        rank = rank_of_gold(pred_5, sample.gold_case_id)

        rows.append(
            {
                "sample_id": sample.sample_id,
                "gold_case_id": sample.gold_case_id,
                "pred_case_id_top1": pred_5[0] if pred_5 else "",
                "pred_case_id_top3": "|".join(pred_5[:3]),
                "pred_case_id_top5": "|".join(pred_5[:5]),
                "gold_rank": str(rank) if rank is not None else "",
                "reciprocal_rank": f"{(1.0 / rank):.6f}" if rank else "0",
                "hit@1": "1" if sample.gold_case_id in pred_5[:1] else "0",
                "hit@3": "1" if sample.gold_case_id in pred_5[:3] else "0",
                "hit@5": "1" if sample.gold_case_id in pred_5[:5] else "0",
                "latency_ms": f"{latency_ms:.3f}",
                "manual_hazard_score_1to5": "",
                "manual_legal_score_1to5": "",
                "manual_overall_score_1to5": "",
                "manual_comments": "",
            }
        )

    metrics = aggregate_ranking_metrics(pred_at_5, gold_ids, TOP_K)
    metrics["num_samples"] = float(len(samples))
    metrics["latency_p50_ms"] = statistics.median(latencies_ms) if latencies_ms else 0.0
    metrics["latency_p95_ms"] = percentile(latencies_ms, 0.95)
    return metrics, rows


def write_output(path: Path, metrics: dict[str, float], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = ["mrr", "hit@1", "hit@3", "hit@5", "ndcg@1", "ndcg@3", "ndcg@5"]
    recall_fields = ["recall@1", "recall@3", "recall@5"]
    fields = [
        "row_type",
        "sample_id",
        "gold_case_id",
        "pred_case_id_top1",
        "pred_case_id_top3",
        "pred_case_id_top5",
        "gold_rank",
        "reciprocal_rank",
        "hit@1",
        "hit@3",
        "hit@5",
        "latency_ms",
        *metric_fields,
        *recall_fields,
        "latency_p50_ms",
        "latency_p95_ms",
        "manual_hazard_score_1to5",
        "manual_legal_score_1to5",
        "manual_overall_score_1to5",
        "manual_comments",
    ]
    summary_row: dict[str, str] = {
        "row_type": "summary",
        "sample_id": "",
        "gold_case_id": "",
        "pred_case_id_top1": "",
        "pred_case_id_top3": "",
        "pred_case_id_top5": "",
        "gold_rank": "",
        "reciprocal_rank": "",
        "hit@1": "",
        "hit@3": "",
        "hit@5": "",
        "latency_ms": "",
    }
    for key in metric_fields + recall_fields + ["latency_p50_ms", "latency_p95_ms"]:
        summary_row[key] = f"{metrics[key]:.4f}" if key in metrics else ""
    summary_row.update(
        {
            "manual_hazard_score_1to5": "",
            "manual_legal_score_1to5": "",
            "manual_overall_score_1to5": "",
            "manual_comments": "",
        }
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary_row)
        for row in rows:
            writer.writerow({"row_type": "sample", **row})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline retrieval evaluation.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_eval_set(args.eval_set)
    if not samples:
        raise ValueError(f"Eval set is empty: {args.eval_set}")
    metrics, rows = run_eval(samples)
    write_output(args.output, metrics, rows)
    print(f"Saved metrics to: {args.output}")
    print(
        " | ".join(
            [
                f"n={int(metrics['num_samples'])}",
                f"MRR={metrics['mrr']:.4f}",
                f"Hit@1={metrics['hit@1']:.4f}",
                f"Hit@3={metrics['hit@3']:.4f}",
                f"nDCG@3={metrics['ndcg@3']:.4f}",
                f"R@5={metrics['recall@5']:.4f}",
                f"P50={metrics['latency_p50_ms']:.3f}ms",
                f"P95={metrics['latency_p95_ms']:.3f}ms",
            ]
        )
    )


if __name__ == "__main__":
    main()
