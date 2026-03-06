from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RetrievedClause:
    case_id: str
    source_file: str
    legal_basis: str
    suggestion: str
    score: float


class ClipFeatureRetriever:
    """轻量检索器（当前使用词面匹配，可替换为 CLIP 向量检索）。"""

    def __init__(
        self,
        metadata_path: str | Path = "data/data_processed/cases_metadata.cleaned.with_image.json",
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.cases = self._load_cases()

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedClause]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: list[RetrievedClause] = []
        for case in self.cases:
            corpus = "\n".join(
                [
                    str(case.get("issue_text", "")),
                    str(case.get("legal_basis", "")),
                    str(case.get("suggestion", "")),
                ]
            )
            corpus_tokens = self._tokenize(corpus)
            if not corpus_tokens:
                continue

            overlap = query_tokens.intersection(corpus_tokens)
            score = len(overlap) / max(len(query_tokens), 1)
            if score <= 0:
                continue

            scored.append(
                RetrievedClause(
                    case_id=str(case.get("id", "")),
                    source_file=str(case.get("source_file", "")),
                    legal_basis=str(case.get("legal_basis", "")).strip(),
                    suggestion=str(case.get("suggestion", "")).strip(),
                    score=score,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _load_cases(self) -> list[dict[str, Any]]:
        if not self.metadata_path.exists():
            return []
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return data.get("cases", []) if isinstance(data, dict) else []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text)
        tokens = re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z]+|\d+", normalized)
        return set(tokens)
