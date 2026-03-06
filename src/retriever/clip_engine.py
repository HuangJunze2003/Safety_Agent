from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


LOGGER = logging.getLogger(__name__)


@dataclass
class SimilarCase:
    case_id: str
    image_id: str
    image_path: str
    issue_text: str
    legal_basis: str
    suggestion: str
    source_file: str
    score: float


class CLIPCaseEngine:
    def __init__(
        self,
        project_root: str | Path = ".",
        images_dir: str | Path = "data/data_processed/images",
        metadata_path: str | Path = "data/data_processed/cases_metadata.json",
        db_path: str | Path = "data/data_processed/chroma_clip",
        model_name: str = "openai/clip-vit-base-patch32",
        collection_name: str = "hazard_cases_images",
        device: str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.images_dir = self.project_root / images_dir
        self.metadata_path = self.project_root / metadata_path
        self.db_path = self.project_root / db_path
        self.collection_name = collection_name

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.use_histogram_fallback = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        try:
            if not self.use_histogram_fallback:
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as exc:
            self.use_histogram_fallback = True
            LOGGER.warning(
                "CLIP 模型加载失败，已切换为本地直方图检索兜底模式: %s",
                exc,
            )

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.case_map, self.image_to_case = self._load_case_index()

    def index_images(self) -> int:
        if self.collection_name in {c.name for c in self.client.list_collections()}:
            self.client.delete_collection(self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        image_paths = sorted(
            [p for p in self.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )
        if not image_paths:
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for image_path in image_paths:
            image_id = image_path.stem
            vector = self._encode_image(image_path)
            if vector is None:
                continue

            case = self._find_case_for_image(image_path)
            case_id = str(case.get("id", ""))
            issue_text = str(case.get("issue_text", ""))
            legal_basis = str(case.get("legal_basis", ""))
            suggestion = str(case.get("suggestion", ""))
            source_file = str(case.get("source_file", ""))

            ids.append(image_id)
            embeddings.append(vector)
            documents.append(issue_text or image_id)
            metadatas.append(
                {
                    "case_id": case_id,
                    "image_id": image_id,
                    "image_path": self._to_project_relative(image_path),
                    "source_file": source_file,
                    "legal_basis": legal_basis,
                    "suggestion": suggestion,
                }
            )

        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        return len(ids)

    def search_similar_cases(self, query_image: str | Path, top_k: int = 3) -> list[SimilarCase]:
        query_path = Path(query_image)
        if not query_path.is_absolute():
            query_path = self.project_root / query_path
        query_path = query_path.resolve()

        if not query_path.exists():
            raise FileNotFoundError(f"查询图片不存在: {query_path}")

        vector = self._encode_image(query_path)
        if vector is None:
            return []

        result = self.collection.query(
            query_embeddings=[vector],
            n_results=max(top_k, 1),
            include=["metadatas", "distances", "documents"],
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]

        output: list[SimilarCase] = []
        for metadata, distance, document in zip(metadatas, distances, documents):
            metadata = metadata or {}
            case_id = str(metadata.get("case_id", ""))
            case = self.case_map.get(case_id, {})
            similarity = 1.0 - float(distance)
            output.append(
                SimilarCase(
                    case_id=case_id,
                    image_id=str(metadata.get("image_id", "")),
                    image_path=str(metadata.get("image_path", "")),
                    issue_text=str(case.get("issue_text", document or "")),
                    legal_basis=str(case.get("legal_basis", metadata.get("legal_basis", ""))),
                    suggestion=str(case.get("suggestion", metadata.get("suggestion", ""))),
                    source_file=str(case.get("source_file", metadata.get("source_file", ""))),
                    score=similarity,
                )
            )
        return output

    def _encode_image(self, image_path: Path) -> list[float] | None:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return None

        if self.use_histogram_fallback or self.model is None or self.processor is None:
            return self._encode_image_histogram(image)

        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu().tolist()

    @staticmethod
    def _encode_image_histogram(image: Image.Image) -> list[float]:
        histogram = image.histogram()
        total = float(sum(histogram))
        if total <= 0:
            return [0.0 for _ in histogram]
        return [float(value) / total for value in histogram]

    def _load_case_index(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if not self.metadata_path.exists():
            return {}, {}

        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        cases = raw.get("cases", []) if isinstance(raw, dict) else []

        case_map: dict[str, dict[str, Any]] = {}
        image_to_case: dict[str, dict[str, Any]] = {}
        for case in cases:
            case_id = str(case.get("id", ""))
            if case_id:
                case_map[case_id] = case

            for image in case.get("images", []) or []:
                path = image.get("path")
                if not path:
                    continue
                normalized = self._normalize_path(path)
                image_to_case[normalized] = case
        return case_map, image_to_case

    def _find_case_for_image(self, image_path: Path) -> dict[str, Any]:
        normalized = self._normalize_path(image_path)
        return self.image_to_case.get(normalized, {})

    def _normalize_path(self, path_like: str | Path) -> str:
        path = Path(path_like)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        else:
            path = path.resolve()
        return str(path)

    def _to_project_relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except Exception:
            return str(path)
