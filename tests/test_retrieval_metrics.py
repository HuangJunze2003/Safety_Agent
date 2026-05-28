"""Unit tests for RAG retrieval ranking metrics."""

from evaluation.retrieval_metrics import (
    aggregate_ranking_metrics,
    hit_at_k,
    ndcg_at_k,
    rank_of_gold,
    reciprocal_rank,
)


def test_rank_and_mrr():
    preds = [["a", "gold", "c"], ["x", "y", "z"]]
    golds = ["gold", "gold"]
    metrics = aggregate_ranking_metrics(preds, golds, k_values=(1, 3))
    assert rank_of_gold(preds[0], "gold") == 2
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(None) == 0.0
    assert metrics["mrr"] == 0.25
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 0.5


def test_ndcg_ideal_and_miss():
    assert ndcg_at_k(1, 3) == 1.0
    assert ndcg_at_k(3, 5) > 0.0
    assert ndcg_at_k(None, 5) == 0.0
    assert hit_at_k(2, 1) == 0.0
    assert hit_at_k(2, 3) == 1.0
