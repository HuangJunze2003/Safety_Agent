"""Ranking metrics for case-level RAG retrieval (single gold label per query)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

DEFAULT_K = (1, 3, 5)


def rank_of_gold(ranked_ids: Sequence[str], gold_id: str) -> int | None:
    """Return 1-based rank of gold_id in ranked_ids, or None if absent."""
    if not gold_id:
        return None
    for idx, doc_id in enumerate(ranked_ids):
        if doc_id == gold_id:
            return idx + 1
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank


def dcg_at_k(rank: int | None, k: int) -> float:
    """Binary relevance: one relevant doc at `rank`, DCG uses (2^rel - 1) / log2(i+1)."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def ndcg_at_k(rank: int | None, k: int) -> float:
    """nDCG@K for a single relevant item (ideal rank = 1)."""
    ideal_dcg = 1.0 / math.log2(2)
    return dcg_at_k(rank, k) / ideal_dcg


def recall_at_k(predictions: Iterable[Sequence[str]], gold_ids: Sequence[str], k: int) -> float:
    """Fraction of queries whose gold id appears in the top-k list."""
    preds = list(predictions)
    golds = list(gold_ids)
    if not golds:
        return 0.0
    hits = sum(
        1 for pred, gold in zip(preds, golds) if gold and gold in pred[:k]
    )
    return hits / len(golds)


def aggregate_ranking_metrics(
    ranked_lists: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    k_values: Sequence[int] = DEFAULT_K,
) -> dict[str, float]:
    """Compute MRR, Hit@K, nDCG@K (and Recall@K alias) over a batch."""
    ranks = [rank_of_gold(pred, gold) for pred, gold in zip(ranked_lists, gold_ids)]
    n = len(ranks) or 1
    metrics: dict[str, float] = {"mrr": sum(reciprocal_rank(r) for r in ranks) / n}
    for k in k_values:
        hit = sum(hit_at_k(r, k) for r in ranks) / n
        metrics[f"hit@{k}"] = hit
        metrics[f"recall@{k}"] = hit
        metrics[f"ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in ranks) / n
    return metrics
