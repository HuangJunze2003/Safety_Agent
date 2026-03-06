"""CLIP 图像特征检索模块。"""

from .clip_engine import CLIPCaseEngine, SimilarCase
from .clip_retriever import ClipFeatureRetriever, RetrievedClause

__all__ = [
    "CLIPCaseEngine",
    "SimilarCase",
    "ClipFeatureRetriever",
    "RetrievedClause",
]
