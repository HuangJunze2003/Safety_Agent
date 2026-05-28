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

@dataclass
class AgentConfig:
    system_prompt_path: str | Path = "prompts/system_role_prompt.txt"
    api_base: str = "http://127.0.0.1:8000/v1"
    model_name: str = "qwen3vl_lora_long_fine"
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

    _HAZARD_JSON_INSTRUCTION = (
        "【思维链分析要求】\n"
        "在给出最终结论前，请先在 JSON 的 \"思维链\" 字段中进行一步步推理：\n"
        "1. 视觉分析：图中观察到了哪些具体的物理特征或作业状态？\n"
        "2. 法规匹配：这些特征对应了检索到的哪些法律条款？\n"
        "3. 风险判定：结合图像证据和法条，确定隐患的严重程度。\n"
        "4. 措施建议：基于隐患原因，提出针对性的技术与管理建议。\n\n"
        "你必须严格输出 JSON 对象，且包含如下键：\n"
        "思维链, 隐患定性, 法律依据, 整改措施, 参考案例。\n"
        "其中 法律依据/整改措施/参考案例 必须是数组。"
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
        
        laws: list[dict[str, Any]] = []
        if legal_top_k > 0:
            if is_legal_only:
                laws = self._retrieve_laws_for_legal_query(question, legal_top_k)
            else:
                laws = self.legal_retriever.search(query=question, top_k=legal_top_k)

        prompt_laws = (
            self._select_laws_for_prompt(question, laws, legal_top_k)
            if is_legal_only
            else laws
        )

        retrieved_cases = self._format_similar_cases(similar_cases)
        retrieved_laws = self._format_laws(prompt_laws)

        system_content = self._build_system_instruction(
            intent=intent,
            retrieved_cases=retrieved_cases,
            retrieved_laws=retrieved_laws,
        )
        user_content = self._build_user_instruction(
            intent=intent,
            question=question,
            image_path=image_abs_path,
        )
        payload_messages = self._build_payload_messages(
            system_content=system_content,
            user_content=user_content,
            image_abs_path=image_abs_path,
        )

        answer = self._chat_completion(payload_messages)
        if intent == IntentType.GREETING and self._is_meta_intent_only(answer):
            answer = self._default_greeting_reply()
        if is_legal_only and self._should_replace_legal_answer(question, answer):
            answer = self._build_legal_structured_answer(question, laws)

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

    def _build_system_instruction(
        self,
        intent: str,
        retrieved_cases: str,
        retrieved_laws: str,
    ) -> str:
        base = self.system_prompt.strip()
        if intent == IntentType.GREETING:
            return (
                f"{base}\n\n"
                "【当前任务】系统已判定本轮为 greeting（问候/身份/能力咨询）。\n"
                "请用简短、自然的普通话直接回答用户，说明你是安全生产管理智能助手，"
                "可协助法规咨询与现场隐患分析。\n"
                "严禁输出 JSON、严禁输出 intent 字段、严禁进行隐患结构化分析。"
            )
        if intent == IntentType.LEGAL_ONLY:
            return (
                f"{base}\n\n"
                "【当前任务】系统已判定本轮为 legal_only（法规咨询）。\n"
                "你必须针对用户问题作答，优先综合下方 [法条检索] 中与问题关键词（如动火、审批、流程）"
                "直接相关的条目，按以下结构用 Markdown 输出：\n"
                "0) **思维链**（必须放在最前，分 4-6 步写清：问题理解→检索筛选→条款比对→结论推导；"
                "可逐步引用 [法条检索] 编号）；\n"
                "1) **结论概述**（2-4 句）；\n"
                "2) **审批/管理流程要点**（分条，每条条目注明法规名称与条款号）；\n"
                "3) **关于用户点名的法规**（若检索未命中《安全生产法》/《消防法》动火审批专条，"
                "须如实说明并引用已命中的下位法、条例或行业规定）；\n"
                "4) **法规依据摘录**（引用检索原文，禁止只贴一条无关条款）。\n"
                "禁止：只复述一条与问题无关的法条、忽略检索列表前部更相关的内容、"
                "输出隐患分析 JSON、输出 intent 字段。\n"
                f"\n\n[法条检索]\n{retrieved_laws}"
            )
        return (
            f"{base}\n\n"
            "【当前任务】系统已判定本轮为 hazard_analysis（现场隐患分析）。\n"
            f"{self._HAZARD_JSON_INSTRUCTION}\n"
            f"\n\n[相似案例检索]\n{retrieved_cases}"
            f"\n\n[法条检索]\n{retrieved_laws}"
        )

    def _build_user_instruction(
        self,
        intent: str,
        question: str,
        image_path: str | None,
    ) -> str:
        if image_path:
            return (
                f"用户问题：{question}\n"
                f"图像路径：{image_path}\n"
                "请结合图片与检索上下文给出结论。"
            )
        if intent == IntentType.GREETING:
            return f"用户问题：{question}"
        if intent == IntentType.LEGAL_ONLY:
            return (
                f"用户问题：{question}\n"
                "请先输出 **思维链**（分步推理），再按系统要求的其余章节作答；"
                "必须覆盖审批/许可/票证、现场措施与验收等要点，"
                "并优先引用 [法条检索] 中含“动火”“审批”“明火”等关键词的条款。"
            )
        return (
            f"用户问题：{question}\n"
            "当前无现场图片，请先说明结论存在不确定性，再给出通用排查与整改建议。"
        )

    def _build_payload_messages(
        self,
        system_content: str,
        user_content: str,
        image_abs_path: str | None,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]
        if not image_abs_path:
            payload.append({"role": "user", "content": user_content})
            return payload

        with open(image_abs_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        base64_image = f"data:image/jpeg;base64,{encoded_string}"
        payload.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            }
        )
        return payload

    @staticmethod
    def _is_meta_intent_only(text: str) -> bool:
        parsed = SafetyProductionAgent._parse_json_output(text)
        if not parsed:
            return False
        allowed = {"intent", "confidence", "reason"}
        keys = set(parsed.keys())
        return keys.issubset(allowed) and "intent" in keys

    @staticmethod
    def _default_greeting_reply() -> str:
        return (
            "您好！我是安全生产管理智能体，面向法规咨询与现场隐患分析场景。"
            "您可以向我咨询安全生产法律法规，或上传现场照片让我结合历史案例与法条"
            "给出隐患研判与整改建议参考。"
        )

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
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=(15, timeout))
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"无法连接模型服务 {self.config.api_base}，请先启动 scripts/start_lf_api.sh "
                f"或 bash scripts/start_all_services.sh"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("模型服务响应超时，请稍后重试或检查 GPU 显存。") from exc
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"].get("content", ""))

    def _retrieve_laws_for_legal_query(self, question: str, top_k: int) -> list[dict[str, Any]]:
        fetch_k = max(top_k * 3, 12)
        expanded = self._expand_legal_query(question)
        primary = self.legal_retriever.search(query=question, top_k=fetch_k)
        extra: list[dict[str, Any]] = []
        if expanded != question:
            extra = self.legal_retriever.search(query=expanded, top_k=fetch_k)
        merged = self._merge_legal_hits(primary + extra)
        return self._rerank_legal_hits(question, merged)[:top_k]

    @staticmethod
    def _expand_legal_query(question: str) -> str:
        extras: list[str] = []
        if re.search(r"动火", question):
            extras.extend(["动火作业", "动火作业票", "审批", "专项安全技术措施"])
        if re.search(r"消防", question):
            extras.extend(["明火作业", "消防条例", "消防安全管理人批准"])
        if re.search(r"安全生产法", question):
            extras.extend(["危险作业", "安全生产法"])
        if not extras:
            return question
        return f"{question} {' '.join(extras)}"

    @staticmethod
    def _merge_legal_hits(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        merged: list[dict[str, Any]] = []
        for item in items:
            basis = str(item.get("legal_basis", ""))
            key = (str(item.get("source_file", "")), basis[:120])
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    @staticmethod
    def _legal_query_terms(question: str) -> list[str]:
        normalized = re.sub(r"\s+", "", question)
        tokens = re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z]+|\d+", normalized)
        stop = {
            "什么",
            "如何",
            "怎么",
            "哪些",
            "是否",
            "有没有",
            "要求",
            "规定",
            "流程",
            "相关",
            "法律法规",
            "法规",
            "法律",
        }
        return [t for t in tokens if t not in stop]

    def _rerank_legal_hits(self, question: str, laws: list[dict[str, Any]]) -> list[dict[str, Any]]:
        terms = self._legal_query_terms(question)
        focus = [t for t in ("动火", "动火作业", "审批", "明火", "作业票") if t in question]
        scored: list[tuple[float, dict[str, Any]]] = []
        for law in laws:
            text = f"{law.get('source_file', '')}{law.get('legal_basis', '')}"
            kw_score = float(sum(2 for t in terms if t in text))
            if focus and any(t in text for t in focus):
                kw_score += 8.0
            if "安全生产法" in question and "安全生产法" in text:
                kw_score += 4.0
            if "消防" in question and "消防" in text:
                kw_score += 3.0
            if not self._is_regulation_source(str(law.get("source_file", ""))):
                kw_score -= 6.0
            emb = float(law.get("score", 1.0))
            scored.append((kw_score * 10.0 - emb, law))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [law for _, law in scored]

    @staticmethod
    def _select_laws_for_prompt(
        question: str,
        laws: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        core_keywords = [k for k in ("动火", "动火作业", "审批", "明火", "作业票") if k in question]
        if not core_keywords:
            return laws[:limit]
        regulation_first = [
            law
            for law in laws
            if SafetyProductionAgent._is_regulation_source(str(law.get("source_file", "")))
        ]
        pool = regulation_first or laws
        relevant = [
            law
            for law in pool
            if any(k in str(law.get("legal_basis", "")) for k in core_keywords)
        ]
        others = [law for law in pool if law not in relevant]
        pool = relevant + others
        return pool[:limit]

    @staticmethod
    def _should_replace_legal_answer(question: str, answer: str) -> bool:
        q = SafetyProductionAgent._normalize_text(question)
        a = SafetyProductionAgent._normalize_text(answer)
        if len(answer.strip()) < 120:
            return True
        if "动火" in q and "动火" not in a:
            return True
        if ("审批" in q or "流程" in q) and "审批" not in a and "流程" not in a and "票" not in a:
            return True
        if "动火" in q and ("消防设计文件" in answer or "消防设计" in answer):
            return True
        return False

    @staticmethod
    def _extract_article_label(content: str) -> str:
        match = re.search(r"第[一二三四五六七八九十百零\d]+条", content)
        return match.group(0) if match else ""

    @staticmethod
    def _is_regulation_source(source_file: str) -> bool:
        source = str(source_file)
        if any(marker in source for marker in ("报告", "案卷", "检查表", "评估报告", "复查")):
            return False
        return any(marker in source for marker in ("法", "条例", "规定", "办法", "标准", "规范", "通知"))

    def _build_legal_thought_chain(
        self, question: str, laws: list[dict[str, Any]], relevant: list[dict[str, Any]]
    ) -> str:
        q = question.strip()
        named = []
        if "安全生产法" in q:
            named.append("《安全生产法》")
        if "消防" in q:
            named.append("消防法规")
        named_text = "、".join(named) if named else "相关安全生产与消防法规"

        top_sources: list[str] = []
        for law in relevant[:4]:
            src = str(law.get("source_file", "")).replace(".docx", "").replace(".pdf", "")
            article = self._extract_article_label(str(law.get("legal_basis", "")))
            label = f"{src}{article}" if article else src
            if label and label not in top_sources:
                top_sources.append(label)

        steps = [
            f"1. **问题理解**：用户询问「{q}」，核心关注 {named_text} 中对作业许可/审批程序的要求。",
            (
                "2. **检索筛选**：在 [法条检索] 中优先保留含“动火/明火/审批/作业票”等关键词的条目，"
                f"共命中 {len(relevant)} 条高度相关法规片段，"
                + (f"主要包括：{'；'.join(top_sources)}。" if top_sources else "需结合场景补充检索。")
            ),
            (
                "3. **条款比对**：将检索结果按效力层级比对——上位法确立危险作业/用火管理原则，"
                "下位条例与行业规定细化动火票、现场勘查、审批签字与验收留痕等程序。"
            ),
            "4. **结论推导**：综合可引用条款归纳审批流程要点；对检索未直接命中的法规须如实说明，并引用已命中的替代依据。",
        ]
        return "\n".join(steps)

    def _build_legal_structured_answer(self, question: str, laws: list[dict[str, Any]]) -> str:
        regulation_laws = [
            law for law in laws if self._is_regulation_source(str(law.get("source_file", "")))
        ]
        law_pool = regulation_laws or laws
        relevant = [
            law
            for law in law_pool
            if any(k in str(law.get("legal_basis", "")) for k in ("动火", "明火", "动火作业票"))
        ]
        thought = self._build_legal_thought_chain(question, laws, relevant)
        lines = [
            "### 📋 法规咨询答复",
            "",
            f"**您的问题**：{question.strip()}",
            "",
            "**思维链**",
            thought,
            "",
            "**结论概述**",
            "动火作业属于火灾爆炸风险较高的特殊作业，应落实作业许可（动火票）制度："
            "作业前勘查并编制专项安全技术措施、按程序分级审批，作业中监护与隔离，作业后验收留痕。",
            "具体审批主体与票证样式以行业规定、地方性法规和本单位制度为准。",
            "",
            "**审批与管理流程要点**（据本轮检索归纳）",
        ]

        step_lines: list[str] = []
        for law in relevant[:8]:
            content = str(law.get("legal_basis", "")).strip()
            source = str(law.get("source_file", "未知文件")).replace(".docx", "").replace(".pdf", "")
            article = self._extract_article_label(content)
            label = f"{source}{article}" if article else source
            for sent in re.split(r"[。；\n]", content):
                sent = sent.strip()
                if not sent:
                    continue
                if any(k in sent for k in ("审批", "批准", "动火作业票", "勘查", "验收", "监护", "票")):
                    step_lines.append(f"- **{label}**：{sent}。")
        if not step_lines:
            for law in relevant[:4]:
                source = str(law.get("source_file", "未知文件"))
                snippet = str(law.get("legal_basis", "")).strip()
                if len(snippet) > 220:
                    snippet = snippet[:220] + "…"
                step_lines.append(f"- **{source}**：{snippet}")
        lines.extend(step_lines[:10] or ["- 本轮检索未命中可直接引用的动火审批条款，请补充行业与场景信息后重试。"])

        lines.extend(["", "**关于您点名的《安全生产法》/消防法规**"])
        has_work_safety = any("安全生产法" in str(law.get("source_file", "")) for law in law_pool)
        fire_on_topic = [
            law
            for law in law_pool
            if "消防" in str(law.get("source_file", ""))
            and any(k in str(law.get("legal_basis", "")) for k in ("动火", "明火", "焊接"))
        ]
        if "安全生产法" in question and not has_work_safety:
            lines.append(
                "- 本轮检索**未召回到**《安全生产法》中直接写明“动火作业审批票证”的专条；"
                "上位法通常将动火纳入**危险作业**统一管理，审批细节由配套规章、行业标准及企业制度细化。"
            )
        if "消防" in question:
            if not fire_on_topic:
                lines.append(
                    "- 《消防法》对建设工程消防设计审查规定较多，对现场**动火票证**着墨较少；"
                    "动火审批更常见地体现在**消防条例**（如地方条例中的明火作业批准）及行业安全管理规定中。"
                )
            else:
                for law in fire_on_topic[:2]:
                    source = str(law.get("source_file", ""))
                    article = self._extract_article_label(str(law.get("legal_basis", "")))
                    snippet = str(law.get("legal_basis", "")).strip()
                    if len(snippet) > 280:
                        snippet = snippet[:280] + "…"
                    lines.append(f"- **{source}{article}**：{snippet}")

        lines.extend(["", "**法规依据摘录**"])
        for idx, law in enumerate(relevant[:5], start=1):
            source = str(law.get("source_file", "未知文件"))
            snippet = str(law.get("legal_basis", "")).strip()
            if len(snippet) > 360:
                snippet = snippet[:360] + "…"
            lines.append(f"{idx}. **{source}**\n   {snippet}")

        lines.append(
            "\n> 说明：以上内容由系统根据向量检索结果自动归纳；"
            "若需适用于特定行业（矿山、化工、建筑施工等），请补充场景以便匹配专门规定。"
        )
        return "\n".join(lines)

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
        model_name=os.getenv("QWEN_MODEL_NAME", "qwen3vl_lora_long_fine"),
        api_key=os.getenv("QWEN_API_KEY", ""),
        top_k=int(os.getenv("RETRIEVE_TOP_K", "3")),
    )
    return SafetyProductionAgent(cfg)
