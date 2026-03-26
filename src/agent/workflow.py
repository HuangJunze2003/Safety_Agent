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


class IntentType:
    GREETING = "greeting"
    LEGAL_ONLY = "legal_only"
    HAZARD_ANALYSIS = "hazard_analysis"


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
                    "请先根据用户问题进行意图识别：\n"
                    "- greeting: 问候、身份、能力说明。\n"
                    "- legal_only: 仅法规咨询、条文解释、处罚依据、流程要求。\n"
                    "- hazard_analysis: 现场隐患分析、风险研判、整改建议（特别是含图片）。\n\n"
                    "处理规则：\n"
                    "1) greeting -> 简短自然语言回答，不输出 JSON。\n"
                    "2) legal_only -> 优先依据 [法条检索] 作答，不输出隐患分析 JSON。\n"
                    "3) hazard_analysis -> 结合 [相似案例检索] 与 [法条检索] 输出结构化 JSON。\n\n"
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

        # 意图识别逻辑：统一入口，优先识别寒暄，再区分法条咨询与隐患分析
        intent = self.detect_intent(question=question, has_image=bool(image_abs_path))
        is_greeting = intent == IntentType.GREETING
        is_legal_only = intent == IntentType.LEGAL_ONLY

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
                    "请先根据用户问题进行意图识别：greeting / legal_only / hazard_analysis。\n"
                    "1) greeting：问候、身份、能力问题。请简短回答，不输出 JSON。\n"
                    "2) legal_only：法规咨询、条文解释、处罚依据。请基于 [法条检索] 专业作答，不输出 JSON。\n"
                    "3) hazard_analysis：隐患排查、风险研判、整改建议。若无图片则明确说明结论不确定性，再给出通用排查建议。\n\n"
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
            "intent": intent,
            "similar_cases": [item.__dict__ for item in similar_cases] if similar_cases else [],
            "retrieved_laws": laws,
            "raw_answer": answer,
            "structured_output": parsed,
        }

    def detect_intent(self, question: str, has_image: bool) -> str:
        """Public intent classifier: LLM-first with regex fallback."""
        llm_intent = self._detect_intent_with_llm(question=question, has_image=has_image)
        if llm_intent in {IntentType.GREETING, IntentType.LEGAL_ONLY, IntentType.HAZARD_ANALYSIS}:
            return llm_intent
        return self._detect_intent(question=question, has_image=has_image)

    def _detect_intent_with_llm(self, question: str, has_image: bool) -> str | None:
        system_prompt = (
            "你是意图分类器，只做分类不做回答。"
            "你只能输出一个 JSON 对象，格式为: "
            '{"intent":"greeting|legal_only|hazard_analysis","confidence":0~1,"reason":"..."}'
            "。"
            "分类规则："
            "greeting=问候/身份/闲聊；"
            "legal_only=法规条文、合规解释、处罚依据；"
            "hazard_analysis=隐患排查、风险研判、整改建议。"
            "若有图片且语义不明确，优先 hazard_analysis。"
            "禁止输出除 JSON 外的任何文本。"
        )
        user_prompt = (
            f"问题: {question}\n"
            f"是否有图片: {'是' if has_image else '否'}\n"
            "请只返回 JSON。"
        )
        try:
            raw = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=120,
                timeout=12,
            )
            parsed = self._parse_json_output(raw)
            intent = str(parsed.get("intent", "")).strip()
            if intent in {IntentType.GREETING, IntentType.LEGAL_ONLY, IntentType.HAZARD_ANALYSIS}:
                return intent
        except Exception:
            return None
        return None

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def _detect_intent(self, question: str, has_image: bool) -> str:
        q = self._normalize_text(question)

        greeting_patterns = [
            r"你好|您好|hi|hello",
            r"你是谁|你是做什么|你能做什么|怎么用",
            r"介绍一下|自我介绍|help|帮助",
        ]
        legal_patterns = [
            r"法律|法规|条例|条款|依据|处罚|罚则|违法|合规",
            r"是否违法|是否合法|怎么规定|如何规定|第[一二三四五六七八九十0-9]+条",
        ]
        hazard_patterns = [
            r"隐患|危险|风险|违章|整改|排查|评估|研判|定级",
            r"图片|图中|照片|现场|作业|施工|设备|防护",
            r"看一下|帮我分析|判断一下",
        ]

        if any(re.search(p, q) for p in greeting_patterns):
            return IntentType.GREETING

        legal_hit = any(re.search(p, q) for p in legal_patterns)
        hazard_hit = any(re.search(p, q) for p in hazard_patterns)

        if has_image:
            # 有图时优先按隐患分析路由，除非明确仅问法规且没有隐患语义。
            if legal_hit and not hazard_hit:
                return IntentType.LEGAL_ONLY
            return IntentType.HAZARD_ANALYSIS

        if hazard_hit and not legal_hit:
            return IntentType.HAZARD_ANALYSIS
        if legal_hit:
            return IntentType.LEGAL_ONLY
        return IntentType.HAZARD_ANALYSIS

    def _resolve_image_path(self, image_path: str | Path | None) -> str:
        if not image_path:
            return ""
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
