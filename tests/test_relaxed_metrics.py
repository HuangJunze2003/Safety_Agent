"""Tests for relaxed retrieval relevance."""

from evaluation.relaxed_metrics import (
    evidence_relaxed_match,
    extract_legal_codes,
    hazard_relaxed_match,
    legal_relaxed_match,
)


def test_legal_code_overlap():
    a = "《危险化学品安全管理条例》 GB3836"
    b = "依据 GB3836 及防爆规范"
    assert legal_relaxed_match(a, b)


def test_hazard_keyword_overlap():
    g = "3:有限空间未设置警示标识"
    c = "6:现场有限空间告知牌与风险辨识的内容不一致"
    assert hazard_relaxed_match(g, c)


def test_evidence_relaxed_or():
    gold = {"hazard": "5: 平台直梯无护笼", "legal": "GBT33000-2016"}
    cand_h = {"hazard": "4:高处作业无护笼", "legal": "其他规范"}
    cand_l = {"hazard": "锅炉房锈蚀", "legal": "GBT33000-2016规定护笼"}
    assert evidence_relaxed_match(gold, cand_h) or evidence_relaxed_match(gold, cand_l)


def test_extract_legal_codes():
    codes = extract_legal_codes("GB/T 50303-2002 与《建筑电气工程施工质量验收规范》")
    assert any("50303" in c or "建筑电气" in c for c in codes)
