from __future__ import annotations

import json
import base64
import os
import re
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from langchain_core.documents import Document

from retriever.clip_engine import CLIPCaseEngine

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception as exc:  # pragma: no cover
    raise RuntimeError("缺少 langchain-core，请先安装 langchain。") from exc


@dataclass
class AgentConfig:
    system_prompt_path: str | Path = "prompts/system_role_prompt.txt"
    api_base: str = "http://127.0.0.1:8000/v1"
    model_name: str = "qwen3vl_lora_local"
    api_key: str = ""
    project_root: str | Path = "."
    top_k: int = 3
    legal_top_k: int = 3
    cases_metadata_path: str | Path = "data/data_processed/cases_metadata.json"
    legal_db_dir: str | Path = "data/data_processed/chroma_legal"
    bge_model_name: str = "BAAI/bge-small-zh-v1.5"


class SafetyProductionAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.project_root = Path(self.config.project_root).resolve()
        self.image_retriever = CLIPCaseEngine(project_root=self.project_root)
        self.legal_retriever = LegalClauseRetriever(
            project_root=self.project_root,
            metadata_path=self.config.cases_metadata_path,
            persist_directory=self.config.legal_db_dir,
            embedding_model_name=self.config.bge_model_name,
        )
        self.system_prompt = self._load_system_prompt(self.config.system_prompt_path)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_prompt}\n\n"
                    "【意图识别与任务处理】\n"
                    "1. 如果用户提问是关于“图片中的隐患分析”、“安全状态研判”，请结合 [相似案例检索] 和 [法条检索] 给出完整报告。\n"
                    "2. 如果用户提问仅涉及“法律条文查询”、“通用法规解释”，请侧重使用 [法条检索] 内容进行专业解答，[相似案例检索] 仅作为参考或不展示。\n\n"
                    "【思维链分析要求】\n"
                    "在给出最终结论前，请先在 JSON 的 \"思维链\" 字段中进行一步步推理：\n"
                    "1. 视觉分析：图中观察到了哪些具体的物理特征或作业状态？\n"
                    "2. 法规匹配：这些特征对应了检索到的哪些法律条款？\n"
                    "3. 风险判定：结合图像证据和法条，确定隐患的严重程度。\n"
                    "4. 措施建议：基于隐患原因，提出针对性的技术与管理建议。\n\n"
                    "你必须严格输出 JSON 对象，且包含如下键：\n"
                    "思维链, 隐患定性, 法律依据, 整改措施, 参考案例。\n"
                    "其中 法律依据/整改措施/参考案例 必须是数组。"
                    "\n\n[相似案例检索]\n{retrieved_cases}"
                    "\n\n[法条检索]\n{retrieved_laws}",
                ),
                (
                    "user",
                    "<image>\n用户问题：{question}\n图像路径：{image_path}\n请基于图片与检索上下文给出结构化结论。",
                ),
            ]
        )

    def analyze(self, image_path: str | Path | None, question: str = "请分析该现场隐患") -> dict[str, Any]:
        # 清理 image_path 并解析绝对路径
        image_abs_path = self._resolve_image_path(image_path) if image_path else None

        # 意图识别逻辑：判断是否为纯法律咨询或寒暄
        greeting_keywords = ["你是谁", "你能做", "介绍", "你好", "谁是"]
        is_greeting = any(kw in question for kw in greeting_keywords)
        
        legal_keywords = ["法律", "法规", "条例", "规定", "怎么说", "条款", "依据"]
        is_legal_only = any(kw in question for kw in legal_keywords) and ("隐患" not in question and "危险" not in question and "图片" not in question)

        similar_cases = []
        if not is_legal_only and not is_greeting and image_abs_path:
            # 只有在非纯法律咨询、非寒暄且有图片时才调用隐患图库
            similar_cases = self.image_retriever.search_similar_cases(
                query_image=image_abs_path,
                top_k=self.config.top_k,
            )
        
        # 法律库根据是否为闲聊稍微调控 depth
        legal_top_k = 0 if is_greeting else (self.config.legal_top_k * 2 if is_legal_only else self.config.legal_top_k)
        
        laws = []
        if legal_top_k > 0:
            laws = self.legal_retriever.search(
                query=question,
                top_k=legal_top_k,
            )

        retrieved_cases = self._format_similar_cases(similar_cases)
        retrieved_laws = self._format_laws(laws)

        # 构造 prompt 参数
        prompt_kwargs = {
            "system_prompt": self.system_prompt,
            "retrieved_cases": retrieved_cases,
            "retrieved_laws": retrieved_laws,
            "question": question,
        }

        if not image_abs_path:
            # 纯文本模式构造
            messages = [
                (
                    "system",
                    "{system_prompt}\n\n"
                    "【意图识别与任务处理】\n"
                    "1. 如果用户提问是关于“图片中的隐患分析”、“安全状态研判”，请结合 [相似案例检索] 和 [法条检索] 给出完整报告。\n"
                    "2. 如果用户提问仅涉及“法律条文查询”、“通用法规解释”，请侧重使用 [法条检索] 内容进行专业解答，不要输出隐患分析的 JSON 结构，直接以自然语言回复。\n"
                    "3. 如果用户问“你是谁”、“你能做什么”等身份问题，请明确回答：“我是一个专业的安全生产管理专家智能体，既能回答相关法规问题，也能根据您上传的现场照片进行安全隐患排查、定性并提供合规的整改建议。”不要输出 JSON 结构。\n\n"
                    "【思维链分析要求】（仅在需要进行隐患排查时使用）\n"
                    "在给出最终结论前，请先在 JSON 的 \"思维链\" 字段中进行一步步推理：\n"
                    "1. 视觉分析：图中观察到了哪些具体的物理特征或作业状态？\n"
                    "2. 法规匹配：这些特征对应了检索到的哪些法律条款？\n"
                    "3. 风险判定：结合图像证据和法条，确定隐患的严重程度。\n"
                    "4. 措施建议：基于隐患原因，提出针对性的技术与管理建议。\n\n"
                    "如果是隐患排查，你必须严格输出 JSON 对象，包含如下键：\n"
                    "思维链, 隐患定性, 法律依据, 整改措施, 参考案例。\n"
                    "其中 法律依据/整改措施/参考案例 必须是数组。"
                    "\n\n[相似案例检索]\n{retrieved_cases}"
                    "\n\n[法条检索]\n{retrieved_laws}",
                ),
                (
                    "user",
                    "用户问题：{question}\n请基于检索上下文给出专业解答。如果涉及法律条文，请直接引用检索到的内容。",
                ),
            ]
            
            # 手动格式化 LangChain Message
            payload_messages = []
            for role, template in messages:
                content = template.format(**prompt_kwargs)
                payload_messages.append({"role": self._to_openai_role(role), "content": content})

        else:
            # 含有图片的模式构造
            prompt_kwargs["image_path"] = image_abs_path
            formatted_messages = self.prompt.format_messages(**prompt_kwargs)
            payload_messages = []
            for message in formatted_messages:
                role = self._to_openai_role(message.type)
                if role == "user":
                    with open(image_abs_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        base64_image = f"data:image/jpeg;base64,{encoded_string}"
                    payload_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": str(message.content),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": base64_image,
                                    },
                                },
                            ],
                        }
                    )
                else:
                    payload_messages.append(
                        {
                            "role": role,
                            "content": str(message.content),
                        }
                    )

        answer = self._chat_completion(payload_messages)

        parsed = self._parse_json_output(answer)
        return {
            "question": question,
            "image_path": str(image_path) if image_path else None,
            "similar_cases": [item.__dict__ for item in similar_cases] if similar_cases else [],
            "retrieved_laws": laws,
            "raw_answer": answer,
            "structured_output": parsed,
        }

    def _resolve_image_path(self, image_path: str | Path | None) -> str:
        if not image_path:
            return ""
        path = Path(image_path)
        if not path.is_absolute():
            path = self.project_root / path
        return str(path.resolve())

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "max_tokens": 2048,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"].get("content", ""))

    @staticmethod
    def _format_similar_cases(items: list[Any]) -> str:
        if not items:
            return "未检索到相似案例。"
        lines: list[str] = []
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"[{idx}] 来源: {item.source_file} | case_id: {item.case_id} | score: {item.score:.3f}\n"
                f"隐患描述: {item.issue_text}\n"
                f"法定依据: {item.legal_basis}\n"
                f"整改建议: {item.suggestion}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_laws(items: list[dict[str, Any]]) -> str:
        if not items:
            return "未检索到法条。"
        lines: list[str] = []
        for idx, item in enumerate(items, start=1):
            source = item.get("source_file", "未知文件")
            # 格式化输出更加友好
            lines.append(
                f"[{idx}] [{source}] (相似距离: {item['score']:.3f})\n"
                f"{item['legal_basis']}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _to_openai_role(role: str) -> str:
        mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        return mapping.get(role, role)

    @staticmethod
    def _load_system_prompt(path: str | Path) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"未找到 system prompt: {p}")
        return p.read_text(encoding="utf-8")

    def _resolve_image_path(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")
        return str(path)

    @staticmethod
    def _parse_json_output(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}


class LegalClauseRetriever:
    def __init__(
        self,
        project_root: str | Path,
        metadata_path: str | Path,
        persist_directory: str | Path,
        embedding_model_name: str,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.metadata_path = self.project_root / metadata_path
        self.persist_directory = self.project_root / persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.use_lexical_fallback = False
        self.store = None
        self.lexical_cases = self._load_cases()

        offline_mode = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        if offline_mode:
            self.use_lexical_fallback = True
            return

        try:
            embeddings_cls = self._load_hf_embeddings_cls()
            self.embedding = embeddings_cls(
                model_name=embedding_model_name,
                model_kwargs={"device": "cuda" if self._has_cuda() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            chroma_cls = self._load_chroma_cls()
            self.store = chroma_cls(
                collection_name="legal_clauses",
                embedding_function=self.embedding,
                persist_directory=str(self.persist_directory),
            )
            self._ensure_indexed()
        except Exception:
            self.use_lexical_fallback = True

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if self.use_lexical_fallback or self.store is None:
            return self._search_lexical(query=query, top_k=top_k)

        # Chroma 相似度搜索返回 (doc, score)，其中 score 是 L2 距离。
        # 距离越小表示越相似。
        results = self.store.similarity_search_with_score(query, k=max(top_k, 5)) 
        
        output: list[dict[str, Any]] = []
        for doc, score in results:
            output.append(
                {
                    "case_id": str(doc.metadata.get("case_id", "law_clause")),
                    "source_file": str(doc.metadata.get("source_file", "未知文件")),
                    "legal_basis": doc.page_content,
                    "score": float(score),
                }
            )
        # 按距离升序排列（最相似的在前）
        output.sort(key=lambda item: item["score"])
        return output[:top_k]

    def _search_lexical(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: list[dict[str, Any]] = []
        for case in self.lexical_cases:
            legal_basis = str(case.get("legal_basis", "")).strip()
            if not legal_basis:
                continue
            tokens = self._tokenize(legal_basis)
            if not tokens:
                continue

            overlap = query_tokens.intersection(tokens)
            if not overlap:
                continue

            similarity = len(overlap) / max(len(query_tokens), 1)
            scored.append(
                {
                    "case_id": str(case.get("id", "")),
                    "source_file": str(case.get("source_file", "")),
                    "legal_basis": legal_basis,
                    "score": float(1.0 - similarity),
                }
            )

        scored.sort(key=lambda item: item["score"])
        return scored[: max(top_k, 1)]

    def _ensure_indexed(self) -> None:
        existing = self.store._collection.count()
        if existing > 0:
            return

        cases = self._load_cases()
        docs: list[Document] = []
        for case in cases:
            legal_basis = str(case.get("legal_basis", "")).strip()
            if not legal_basis:
                continue
            docs.append(
                Document(
                    page_content=legal_basis,
                    metadata={
                        "case_id": str(case.get("id", "")),
                        "source_file": str(case.get("source_file", "")),
                    },
                )
            )
        if docs:
            self.store.add_documents(docs)

    def _load_cases(self) -> list[dict[str, Any]]:
        if not self.metadata_path.exists():
            return []
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return raw.get("cases", []) if isinstance(raw, dict) else []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text)
        tokens = re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z]+|\d+", normalized)
        return set(tokens)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _load_chroma_cls() -> Any:
        candidates = [
            ("langchain_chroma", "Chroma"),
            ("langchain_community.vectorstores", "Chroma"),
            ("langchain.vectorstores", "Chroma"),
        ]
        for module_name, attr in candidates:
            try:
                module = import_module(module_name)
                return getattr(module, attr)
            except Exception:
                continue
        raise RuntimeError("未找到 Chroma 实现，请安装 langchain-chroma 或 langchain-community。")

    @staticmethod
    def _load_hf_embeddings_cls() -> Any:
        candidates = [
            ("langchain_huggingface", "HuggingFaceEmbeddings"),
            ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
            ("langchain.embeddings", "HuggingFaceEmbeddings"),
        ]
        for module_name, attr in candidates:
            try:
                module = import_module(module_name)
                return getattr(module, attr)
            except Exception:
                continue
        raise RuntimeError(
            "未找到 HuggingFaceEmbeddings 实现，请安装 langchain-huggingface 或 langchain-community。"
        )


def build_agent_from_env() -> SafetyProductionAgent:
    cfg = AgentConfig(
        system_prompt_path=os.getenv("SYSTEM_PROMPT_PATH", "prompts/system_role_prompt.txt"),
        api_base=os.getenv("QWEN_API_BASE", "http://127.0.0.1:8000/v1"),
        model_name=os.getenv("QWEN_MODEL_NAME", "qwen3vl_lora_local"),
        api_key=os.getenv("QWEN_API_KEY", ""),
        top_k=int(os.getenv("RETRIEVE_TOP_K", "3")),
    )
    return SafetyProductionAgent(cfg)
