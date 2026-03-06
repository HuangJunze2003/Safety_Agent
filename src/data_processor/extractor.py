from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import fitz
from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractContext:
    source_file: str
    source_type: str
    case_counter: int = 0
    pending_images: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CaseExtractor:
    ISSUE_KEYWORDS = (
        "问题隐患",
        "隐患问题",
        "隐患描述",
        "问题描述",
        "存在问题",
        "主要问题",
    )
    LEGAL_KEYWORDS = (
        "法定依据",
        "法律依据",
        "法规依据",
        "依据条款",
        "相关依据",
        "依据",
    )
    ADVICE_KEYWORDS = (
        "整改要求或建议",
        "建议",
        "整改建议",
        "整改措施",
        "处理建议",
        "防范措施",
        "整改要求",
    )

    def __init__(
        self,
        project_root: str | Path = ".",
        processed_dir: str | Path = "data/data_processed",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.processed_dir = self.project_root / processed_dir
        self.images_dir = self.processed_dir / "images"
        self.metadata_path = self.processed_dir / "cases_metadata.json"
        self.review_checklist_path = self.processed_dir / "manual_review_checklist.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(self, input_dir: str | Path) -> list[dict[str, Any]]:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_path}")

        cases: list[dict[str, Any]] = []
        for file_path in sorted(input_path.rglob("*")):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix == ".docx":
                cases.extend(self.extract_from_word(file_path))
            elif suffix == ".pdf":
                cases.extend(self.extract_from_pdf(file_path))

        review_items = self._mark_cases_for_manual_review(cases)
        self.save_cases_metadata(cases)
        self.save_manual_review_checklist(review_items)
        return cases

    def extract_from_word(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        doc = Document(path)
        context = ExtractContext(source_file=path.name, source_type="word")
        cases: list[dict[str, Any]] = []
        current_case: dict[str, Any] | None = None
        active_field = "issue_text"

        for paragraph in self._iter_docx_paragraphs(doc):
            text = self._normalize_text(paragraph.text)
            paragraph_image_ids = self._extract_images_from_paragraph(paragraph, path.stem, "docx")
            context.pending_images.extend(paragraph_image_ids)

            if not text:
                continue

            issue_keyword = self._match_keyword(text, self.ISSUE_KEYWORDS)
            if issue_keyword and self._is_case_start_text(text, issue_keyword):
                current_case = self._start_new_case(context, text)
                cases.append(current_case)
                self._bind_pending_images(current_case, context.pending_images, "issue")
                active_field = "issue_text"
                continue

            if current_case is None:
                continue

            legal_keyword = self._match_keyword(text, self.LEGAL_KEYWORDS, prefix_only=True)
            if legal_keyword:
                payload = self._extract_field_payload(text, legal_keyword)
                if payload:
                    current_case["legal_basis"] = self._merge_text(current_case.get("legal_basis", ""), payload)
                self._bind_pending_images(current_case, context.pending_images, "legal_basis")
                active_field = "legal_basis"
                continue

            advice_keyword = self._match_keyword(text, self.ADVICE_KEYWORDS, prefix_only=True)
            if advice_keyword:
                payload = self._extract_field_payload(text, advice_keyword)
                if payload:
                    current_case["suggestion"] = self._merge_text(current_case.get("suggestion", ""), payload)
                self._bind_pending_images(current_case, context.pending_images, "suggestion")
                active_field = "suggestion"
                continue

            if self._is_photo_caption_text(text):
                continue

            if active_field == "suggestion":
                current_case["suggestion"] = self._merge_text(current_case.get("suggestion", ""), text)
            elif active_field == "legal_basis":
                current_case["legal_basis"] = self._merge_text(current_case.get("legal_basis", ""), text)
            else:
                current_case["issue_text"] = self._merge_text(current_case["issue_text"], text)

        self._finalize_orphan_images(cases, context.pending_images)
        if context.warnings:
            LOGGER.warning("Word 解析告警 [%s]: %s", path.name, " | ".join(context.warnings))
        return cases

    def extract_from_pdf(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        context = ExtractContext(source_file=path.name, source_type="pdf")
        cases: list[dict[str, Any]] = []
        current_case: dict[str, Any] | None = None
        active_field = "issue_text"
        in_rectification_section = False

        try:
            pdf = fitz.open(path)
        except Exception as exc:
            raise RuntimeError(f"无法打开 PDF 文件: {path}") from exc

        with pdf:
            collected_pdf_lines: list[str] = []
            for page_index, page in enumerate(pdf, start=1):
                page_image_ids = self._extract_images_from_pdf_page(page, path.stem, page_index)
                context.pending_images.extend(page_image_ids)

                raw_text = page.get_text("text") or ""
                if not self._normalize_text(raw_text):
                    context.warnings.append(f"第 {page_index} 页未提取到文本，可能为扫描件")
                    ocr_text = self._ocr_page_placeholder(page)
                    raw_text = ocr_text or ""

                lines = self._split_lines(raw_text)
                if not lines:
                    continue
                collected_pdf_lines.extend(lines)

                for line in lines:
                    compact_line = line.replace(" ", "")
                    if "隐患整改复查情况" in compact_line:
                        in_rectification_section = True
                    if "整改前照片" in compact_line or "整改后照片" in compact_line:
                        in_rectification_section = True

                    issue_keyword = self._match_keyword(line, self.ISSUE_KEYWORDS)
                    if (
                        issue_keyword
                        and self._is_case_start_text(line, issue_keyword)
                        and not in_rectification_section
                    ):
                        current_case = self._start_new_case(context, line, page_index=page_index)
                        cases.append(current_case)
                        self._bind_pending_images(current_case, context.pending_images, "issue")
                        active_field = "issue_text"
                        continue

                    if current_case is None:
                        continue

                    legal_keyword = self._match_keyword(line, self.LEGAL_KEYWORDS, prefix_only=True)
                    if legal_keyword:
                        payload = self._extract_field_payload(line, legal_keyword)
                        if payload:
                            current_case["legal_basis"] = self._merge_text(current_case.get("legal_basis", ""), payload)
                        self._bind_pending_images(current_case, context.pending_images, "legal_basis")
                        active_field = "legal_basis"
                        continue

                    advice_keyword = self._match_keyword(line, self.ADVICE_KEYWORDS, prefix_only=True)
                    if advice_keyword:
                        payload = self._extract_field_payload(line, advice_keyword)
                        if payload:
                            current_case["suggestion"] = self._merge_text(current_case.get("suggestion", ""), payload)
                        self._bind_pending_images(current_case, context.pending_images, "suggestion")
                        active_field = "suggestion"
                        continue

                    if active_field == "issue_text" and self._looks_like_legal_line(line):
                        current_case["legal_basis"] = self._merge_text(current_case.get("legal_basis", ""), line)
                        self._bind_pending_images(current_case, context.pending_images, "legal_basis")
                        active_field = "legal_basis"
                        continue

                    if self._is_photo_caption_text(line):
                        continue

                    if active_field == "suggestion":
                        current_case["suggestion"] = self._merge_text(current_case.get("suggestion", ""), line)
                    elif active_field == "legal_basis":
                        current_case["legal_basis"] = self._merge_text(current_case.get("legal_basis", ""), line)
                    else:
                        current_case["issue_text"] = self._merge_text(current_case["issue_text"], line)

        self._finalize_orphan_images(cases, context.pending_images)
        self._backfill_pdf_missing_fields(cases, collected_pdf_lines)
        if context.warnings:
            LOGGER.warning("PDF 解析告警 [%s]: %s", path.name, " | ".join(context.warnings))
        return cases

    def save_cases_metadata(self, cases: list[dict[str, Any]], output_path: str | Path | None = None) -> Path:
        target = Path(output_path) if output_path else self.metadata_path
        target.parent.mkdir(parents=True, exist_ok=True)

        missing_legal = sum(1 for case in cases if not str(case.get("legal_basis", "")).strip())
        missing_suggestion = sum(1 for case in cases if not str(case.get("suggestion", "")).strip())
        needs_manual_review = sum(1 for case in cases if bool(case.get("needs_manual_review")))

        payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "total_cases": len(cases),
            "quality_summary": {
                "missing_legal_basis": missing_legal,
                "missing_suggestion": missing_suggestion,
                "needs_manual_review": needs_manual_review,
            },
            "cases": cases,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def save_manual_review_checklist(
        self,
        review_items: list[dict[str, Any]],
        output_path: str | Path | None = None,
    ) -> Path:
        target = Path(output_path) if output_path else self.review_checklist_path
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "total_review_items": len(review_items),
            "items": review_items,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def _start_new_case(
        self,
        context: ExtractContext,
        issue_text: str,
        page_index: int | None = None,
    ) -> dict[str, Any]:
        context.case_counter += 1
        case_id = f"{Path(context.source_file).stem}-{context.case_counter:04d}-{uuid4().hex[:8]}"
        case = {
            "id": case_id,
            "source_file": context.source_file,
            "source_type": context.source_type,
            "page": page_index,
            "issue_text": self._extract_field_payload_by_keywords(issue_text, self.ISSUE_KEYWORDS) or issue_text,
            "legal_basis": "",
            "suggestion": "",
            "images": [],
            "image_bindings": [],
        }
        return case

    def _iter_docx_paragraphs(self, doc: _Document):
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, doc)
                continue
            if child.tag.endswith("}tbl"):
                table = Table(child, doc)
                for row in table.rows:
                    for cell in row.cells:
                        yield from self._iter_cell_paragraphs(cell)

    def _iter_cell_paragraphs(self, cell: _Cell):
        for paragraph in cell.paragraphs:
            yield paragraph

    def _extract_images_from_paragraph(
        self,
        paragraph: Paragraph,
        source_stem: str,
        source_type: str,
    ) -> list[dict[str, str]]:
        image_items: list[dict[str, str]] = []
        for run in paragraph.runs:
            try:
                embed_ids = run._element.xpath(".//a:blip/@r:embed")
            except Exception:
                embed_ids = []

            for embed_id in embed_ids:
                image_part = paragraph.part.related_parts.get(embed_id)
                if not image_part:
                    continue
                content_type = getattr(image_part, "content_type", "")
                extension = self._extension_from_content_type(content_type)
                image_id = f"img-{source_stem}-{uuid4().hex[:12]}"
                image_name = f"{image_id}.{extension}"
                image_path = self.images_dir / image_name
                image_path.write_bytes(image_part.blob)
                image_items.append(
                    {
                        "image_id": image_id,
                        "path": f"data/data_processed/images/{image_name}",
                    }
                )
        return image_items

    def _extract_images_from_pdf_page(self, page: fitz.Page, source_stem: str, page_index: int) -> list[dict[str, str]]:
        image_items: list[dict[str, str]] = []
        for image_no, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            try:
                base_image = page.parent.extract_image(xref)
            except Exception:
                continue
            if not base_image:
                continue
            image_bytes = base_image.get("image")
            extension = base_image.get("ext", "png")
            if not image_bytes:
                continue
            image_id = f"img-{source_stem}-p{page_index:03d}-{image_no:02d}-{uuid4().hex[:6]}"
            image_name = f"{image_id}.{extension}"
            image_path = self.images_dir / image_name
            image_path.write_bytes(image_bytes)
            image_items.append(
                {
                    "image_id": image_id,
                    "path": f"data/data_processed/images/{image_name}",
                }
            )
        return image_items

    def _bind_pending_images(self, case: dict[str, Any], pending_images: list[dict[str, str]], field_name: str) -> None:
        if not pending_images:
            return

        existing_images = {img["image_id"] for img in case["images"]}
        for image_item in pending_images:
            image_id = image_item["image_id"]
            if image_id not in existing_images:
                case["images"].append(
                    {
                        "image_id": image_id,
                        "path": image_item["path"],
                    }
                )
            case["image_bindings"].append(
                {
                    "image_id": image_id,
                    "field": field_name,
                }
            )
        pending_images.clear()

    def _finalize_orphan_images(self, cases: list[dict[str, Any]], pending_images: list[dict[str, str]]) -> None:
        if not pending_images or not cases:
            return
        self._bind_pending_images(cases[-1], pending_images, "unassigned")

    def _ocr_page_placeholder(self, page: fitz.Page) -> str:
        """预留 OCR 扩展接口。

        后续可在此处接入 pytesseract 或 PaddleOCR：
        1. 将 page 渲染为图像
        2. 执行 OCR
        3. 返回识别文本
        """
        _ = page
        return ""

    @staticmethod
    def _split_lines(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _merge_text(existing: str, chunk: str) -> str:
        if not existing:
            return chunk
        return f"{existing}\n{chunk}" if chunk else existing

    @staticmethod
    def _extract_field_payload(text: str, keyword: str) -> str:
        if keyword not in text:
            return ""
        normalized = text.replace("：", ":")
        parts = normalized.split(keyword, 1)
        if len(parts) < 2:
            return ""
        payload = parts[1].lstrip(":： ")
        return payload.strip()

    @staticmethod
    def _match_keyword(text: str, keywords: tuple[str, ...], prefix_only: bool = False) -> str:
        normalized = text.replace(" ", "")
        for keyword in keywords:
            if prefix_only:
                if re.match(rf"^(\d+[、.．]\s*)?{re.escape(keyword)}\s*[：:]?", normalized):
                    return keyword
                continue
            if keyword in text:
                return keyword
        return ""

    def _extract_field_payload_by_keywords(self, text: str, keywords: tuple[str, ...]) -> str:
        matched = self._match_keyword(text, keywords)
        if not matched:
            return ""
        return self._extract_field_payload(text, matched)

    @staticmethod
    def _looks_like_legal_line(text: str) -> bool:
        compact = text.replace(" ", "")
        if "《" in compact and "》" in compact:
            return True
        if re.search(r"第.{0,8}条", compact):
            return True
        if re.search(r"\bGB/?T?\s*\d+", text, flags=re.I):
            return True
        if any(token in compact for token in ("安全生产法", "条例", "管理规定", "导则", "规范")):
            return True
        return False

    @staticmethod
    def _extract_issue_no(text: str) -> str:
        compact = text.replace(" ", "")
        patterns = [
            r"问题隐患(\d+)",
            r"^(\d+)[：:]",
            r"^(\d+)[、.．]",
        ]
        for pattern in patterns:
            match = re.search(pattern, compact)
            if match:
                return match.group(1)
        return ""

    def _extract_issue_blocks_from_pdf_lines(self, lines: list[str]) -> dict[str, dict[str, str]]:
        blocks: dict[str, dict[str, str]] = {}
        current_issue = ""
        active_field = "issue_text"
        in_rectification_section = False

        for line in lines:
            compact = line.replace(" ", "")
            if "隐患整改复查情况" in compact:
                in_rectification_section = True
            if "整改前照片" in compact or "整改后照片" in compact:
                in_rectification_section = True

            issue_keyword = self._match_keyword(line, self.ISSUE_KEYWORDS)
            if issue_keyword and self._is_case_start_text(line, issue_keyword) and not in_rectification_section:
                issue_no = self._extract_issue_no(line)
                if issue_no:
                    current_issue = issue_no
                    blocks.setdefault(current_issue, {"legal_basis": "", "suggestion": ""})
                    active_field = "issue_text"
                continue

            if not current_issue:
                continue

            legal_keyword = self._match_keyword(line, self.LEGAL_KEYWORDS, prefix_only=True)
            if legal_keyword:
                payload = self._extract_field_payload(line, legal_keyword)
                if payload:
                    blocks[current_issue]["legal_basis"] = self._merge_text(blocks[current_issue]["legal_basis"], payload)
                active_field = "legal_basis"
                continue

            advice_keyword = self._match_keyword(line, self.ADVICE_KEYWORDS, prefix_only=True)
            if advice_keyword:
                payload = self._extract_field_payload(line, advice_keyword)
                if payload:
                    blocks[current_issue]["suggestion"] = self._merge_text(blocks[current_issue]["suggestion"], payload)
                active_field = "suggestion"
                continue

            if self._is_photo_caption_text(line):
                continue

            if active_field == "issue_text" and self._looks_like_legal_line(line):
                blocks[current_issue]["legal_basis"] = self._merge_text(blocks[current_issue]["legal_basis"], line)
                active_field = "legal_basis"
                continue

            if active_field == "legal_basis":
                blocks[current_issue]["legal_basis"] = self._merge_text(blocks[current_issue]["legal_basis"], line)
            elif active_field == "suggestion":
                blocks[current_issue]["suggestion"] = self._merge_text(blocks[current_issue]["suggestion"], line)

        return blocks

    def _backfill_pdf_missing_fields(self, cases: list[dict[str, Any]], pdf_lines: list[str]) -> None:
        if not cases:
            return
        issue_blocks = self._extract_issue_blocks_from_pdf_lines(pdf_lines)
        if not issue_blocks:
            return

        for case in cases:
            if case.get("source_type") != "pdf":
                continue
            issue_no = self._extract_issue_no(str(case.get("issue_text", "")))
            if not issue_no:
                continue
            block = issue_blocks.get(issue_no)
            if not block:
                continue
            if not str(case.get("legal_basis", "")).strip() and block.get("legal_basis", "").strip():
                case["legal_basis"] = block["legal_basis"].strip()
            if not str(case.get("suggestion", "")).strip() and block.get("suggestion", "").strip():
                case["suggestion"] = block["suggestion"].strip()

    def _mark_cases_for_manual_review(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        review_items: list[dict[str, Any]] = []
        for case in cases:
            reasons: list[str] = []
            if not str(case.get("legal_basis", "")).strip():
                reasons.append("missing_legal_basis")
            if not str(case.get("suggestion", "")).strip():
                reasons.append("missing_suggestion")

            if reasons:
                case["needs_manual_review"] = True
                case["manual_review_reasons"] = reasons
                review_items.append(
                    {
                        "id": case.get("id", ""),
                        "source_file": case.get("source_file", ""),
                        "source_type": case.get("source_type", ""),
                        "page": case.get("page", None),
                        "issue_text": str(case.get("issue_text", ""))[:300],
                        "missing_fields": reasons,
                    }
                )
            else:
                case["needs_manual_review"] = False
                case["manual_review_reasons"] = []
        return review_items

    @staticmethod
    def _extension_from_content_type(content_type: str) -> str:
        mapping = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/gif": "gif",
            "image/bmp": "bmp",
            "image/tiff": "tiff",
            "image/webp": "webp",
        }
        return mapping.get(content_type.lower(), "png")

    @staticmethod
    def _is_photo_caption_text(text: str) -> bool:
        compact = text.replace(" ", "")
        return any(token in compact for token in ("现场检查照片", "现场照片", "图一", "图二", "图三"))

    @staticmethod
    def _is_case_start_text(text: str, matched_issue_keyword: str) -> bool:
        normalized = text.replace(" ", "")
        if "问题隐患、重大风险及建议" in normalized:
            return False
        if "现将安全隐患排查有关情况报告如下" in normalized:
            return False

        if re.search(rf"{re.escape(matched_issue_keyword)}\s*\d+\s*[：:]", normalized):
            return True
        if re.search(rf"{re.escape(matched_issue_keyword)}\s*\d+", normalized):
            return True
        if re.search(rf"{re.escape(matched_issue_keyword)}\s*[：:]", normalized):
            return True
        if re.search(r"^\d+[、.．]\s*", normalized):
            return True
        return False


def run_extraction(
    input_dir: str | Path = "data/data_raw/cases",
    project_root: str | Path = ".",
) -> list[dict[str, Any]]:
    extractor = CaseExtractor(project_root=project_root)
    return extractor.extract_all(input_dir=input_dir)
