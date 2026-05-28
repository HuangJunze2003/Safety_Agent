"""Relaxed retrieval relevance for safety-case RAG evaluation."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
# GB/T, GBT, B50303, 条例/规定/规范 等法规标识
LEGAL_CODE_PATTERN = re.compile(
    r"GB\s*/?\s*T?\s*\d{4,6}|GBT\s*\d{4,6}|B\s*\d{5}|TSG\s*\d+[-\w]*|"
    r"《[^》]{2,40}》|条例|规定|规范|办法"
)
HAZARD_RISK_KEYWORDS = (
    "有限空间",
    "配电",
    "电气",
    "护笼",
    "栏杆",
    "临边",
    "危化",
    "化学品",
    "锅炉",
    "燃气",
    "消防",
    "警示",
    "隔离",
    "通风",
    "应急",
)


def tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_PATTERN.findall(text or "")}


def extract_legal_codes(text: str) -> set[str]:
    codes: set[str] = set()
    for m in LEGAL_CODE_PATTERN.findall(text or ""):
        norm = re.sub(r"\s+", "", m).upper()
        codes.add(norm)
    return codes


def extract_hazard_keywords(text: str) -> set[str]:
    hazard = text or ""
    keys = {kw for kw in HAZARD_RISK_KEYWORDS if kw in hazard}
    # 去掉 "序号:" 前缀，保留隐患描述主体
    body = re.sub(r"^\d+\s*[:：]\s*", "", hazard.strip())
    tokens = tokenize(body)
    return keys | {t for t in tokens if len(t) >= 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def hazard_relaxed_match(gold_hazard: str, cand_hazard: str, jaccard_threshold: float = 0.18) -> bool:
    """同隐患主题：风险关键词重叠或描述 token Jaccard 达阈。"""
    g_keys = extract_hazard_keywords(gold_hazard)
    c_keys = extract_hazard_keywords(cand_hazard)
    if g_keys & c_keys:
        return True
    return jaccard(tokenize(gold_hazard), tokenize(cand_hazard)) >= jaccard_threshold


def legal_relaxed_match(gold_legal: str, cand_legal: str) -> bool:
    """同法规条款：共享至少一条法规编号/文件标识。"""
    g_codes = extract_legal_codes(gold_legal)
    c_codes = extract_legal_codes(cand_legal)
    if g_codes and c_codes and (g_codes & c_codes):
        return True
    return jaccard(tokenize(gold_legal), tokenize(cand_legal)) >= 0.08


def evidence_relaxed_match(
    gold: Mapping[str, str],
    cand: Mapping[str, str],
) -> bool:
    """证据层放宽：隐患主题或法规依据任一匹配即视为相关。"""
    return hazard_relaxed_match(
        gold.get("hazard", ""), cand.get("hazard", "")
    ) or legal_relaxed_match(gold.get("legal", ""), cand.get("legal", ""))


def relaxed_rank(
    ranked_ids: Sequence[str],
    gold_id: str,
    corpus_by_id: Mapping[str, Mapping[str, str]],
    match_fn,
) -> int | None:
    """1-based rank of first relaxed-relevant doc in ranked_ids."""
    gold = corpus_by_id.get(gold_id, {})
    if not gold:
        return None
    for idx, doc_id in enumerate(ranked_ids):
        cand = corpus_by_id.get(doc_id, {})
        if doc_id == gold_id or match_fn(gold, cand):
            return idx + 1
    return None


def aggregate_relaxed_metrics(
    ranked_lists: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    corpus_by_id: Mapping[str, Mapping[str, str]],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Compute relaxed Hit/MRR/nDCG for hazard, legal, and combined evidence."""
    from evaluation.retrieval_metrics import hit_at_k, ndcg_at_k, reciprocal_rank

    n = len(gold_ids) or 1
    modes = {
        "hazard": lambda g, c: hazard_relaxed_match(g.get("hazard", ""), c.get("hazard", "")),
        "legal": lambda g, c: legal_relaxed_match(g.get("legal", ""), c.get("legal", "")),
        "evidence": evidence_relaxed_match,
    }
    out: dict[str, float] = {}
    for name, match_fn in modes.items():
        ranks = [
            relaxed_rank(pred, gold, corpus_by_id, match_fn)
            for pred, gold in zip(ranked_lists, gold_ids)
        ]
        out[f"{name}_mrr"] = sum(reciprocal_rank(r) for r in ranks) / n
        for k in k_values:
            hit = sum(hit_at_k(r, k) for r in ranks) / n
            out[f"{name}_hit@{k}"] = hit
            out[f"{name}_ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in ranks) / n
    return out
