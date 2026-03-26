import os
import sys
import json
import uuid
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document

# 设置路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.workflow import LegalClauseRetriever
from retriever.clip_engine import CLIPCaseEngine
from data_processor.extractor import CaseExtractor

app = FastAPI(title="Safety Agent Data Ingestion API")

# 获取配置
# 注意：这里我们复用 AgentConfig 中的路径
LEGAL_DB_DIR = "data/data_processed/chroma_legal"
CASES_METADATA_PATH = "data/data_processed/cases_metadata.json"
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


# -----------------
# 工具函数
# -----------------

def _metadata_path() -> Path:
    return PROJECT_ROOT / CASES_METADATA_PATH


def _load_cases() -> list[dict[str, Any]]:
    path = _metadata_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw.get("cases", []) if isinstance(raw, dict) else []
    except Exception:
        return []


def _save_cases(cases: list[dict[str, Any]]) -> None:
    path = _metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_cases(cases: list[dict[str, Any]], lib_type: str | None) -> list[dict[str, Any]]:
    if not lib_type:
        return cases
    if lib_type == "laws":
        return [c for c in cases if "legal_basis" in c]
    if lib_type == "hazards":
        return [c for c in cases if "issue_text" in c]
    return cases


def _legal_store():
    retriever = LegalClauseRetriever(
        project_root=PROJECT_ROOT,
        metadata_path=CASES_METADATA_PATH,
        persist_directory=LEGAL_DB_DIR,
        embedding_model_name=BGE_MODEL_NAME,
    )
    return retriever.store


def _sync_legal_add(case: dict[str, Any]) -> None:
    store = _legal_store()
    if store:
        store.add_documents(
            [Document(page_content=case.get("legal_basis", ""), metadata={"case_id": case.get("id", ""), "source_file": case.get("source_file", "")})],
            ids=[case.get("id", "")],
        )


def _sync_legal_delete(case_id: str) -> None:
    store = _legal_store()
    if store:
        # 按元数据 case_id 删除，兼容旧文档未设置 id 的场景
        try:
            store._collection.delete(where={"case_id": case_id})
        except Exception:
            try:
                store.delete(ids=[case_id])
            except Exception:
                pass

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    lib_type: str = Form(...), # "laws" 或 "hazards"
):
    """
    上传文件并自动处理入库
    lib_type: "laws" (法律法规库) 或 "hazards" (安全隐患库)
    """
    if lib_type not in ["laws", "hazards"]:
        raise HTTPException(status_code=400, detail="Invalid library type. Must be 'laws' or 'hazards'.")

    # 创建临时上传目录
    temp_dir = PROJECT_ROOT / "data" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if lib_type == "laws":
            return await process_laws(file_path)
        else:
            return await process_hazards(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if file_path.exists():
            file_path.unlink()

async def process_laws(file_path: Path):
    """处理法律法规文件并加入 ChromaDB"""
    from scripts.add_laws_to_db import extract_text_from_file, chunk_text
    
    # 1. 提取文本
    text = extract_text_from_file(file_path)
    if not text:
        return {"status": "error", "message": "Could not extract text from file."}
    
    # 2. 切分文本
    chunks = chunk_text(text)
    
    # 3. 加载现有的 metadata 
    metadata_path = PROJECT_ROOT / CASES_METADATA_PATH
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cases = data.get("cases", [])
    else:
        cases = []

    # 4. 构造新记录
    new_cases = []
    source_name = file_path.name.split('_', 1)[-1]
    for i, chunk in enumerate(chunks):
        new_case = {
            "id": f"law_{uuid.uuid4().hex[:8]}",
            "source_file": source_name,
            "legal_basis": chunk
        }
        new_cases.append(new_case)
        cases.append(new_case)

    # 5. 保存更新后的 metadata (LegalClauseRetriever 会读取这个文件来初始化/更新向量库)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=2)

    # 6. 强制刷新向量库
    # 我们实例化一个新的 retriever，它在初始化时会自动调用 _ensure_indexed 把新加的条目写进 Chroma
    # 注意：LegalClauseRetriever._ensure_indexed 如果发现库里有数据就不加了，我们需要改进它或者手动添加
    retriever = LegalClauseRetriever(
        project_root=PROJECT_ROOT,
        metadata_path=CASES_METADATA_PATH,
        persist_directory=LEGAL_DB_DIR,
        embedding_model_name=BGE_MODEL_NAME
    )
    
    # 由于原始代码核心逻辑在 _ensure_indexed 且有 existing > 0 就跳过的逻辑，
    # 我们直接手动添加新文档
    if retriever.store:
        docs = [Document(page_content=c["legal_basis"], metadata={"case_id": c["id"], "source_file": c["source_file"]}) for c in new_cases]
        retriever.store.add_documents(docs, ids=[c["id"] for c in new_cases])

    return {"status": "success", "message": f"Successfully processed {len(new_cases)} law clauses from {source_name}"}

async def process_hazards(file_path: Path):
    """处理安全隐患文件 (Word/PDF)"""
    # 1. 使用 CaseExtractor 提取
    extractor = CaseExtractor(project_root=PROJECT_ROOT)
    
    # 复制到临时目录因为 extract_all 扫描目录
    task_dir = file_path.parent / f"task_{uuid.uuid4().hex}"
    task_dir.mkdir()
    shutil.copy(file_path, task_dir / file_path.name.split('_', 1)[-1])
    
    try:
        new_cases = extractor.extract_all(task_dir)
        
        if not new_cases:
            return {"status": "error", "message": "No typical hazard cases found in the document."}
        
        # 2. 刷新图像向量库 (如果包含图片)
        engine = CLIPCaseEngine(project_root=PROJECT_ROOT)
        indexed_count = engine.index_images()
        
        return {
            "status": "success", 
            "message": f"Extracted {len(new_cases)} hazard cases. Indexed {indexed_count} images."
        }
    finally:
        shutil.rmtree(task_dir)


# -----------------
# 知识库 CRUD 接口
# -----------------


class KbCreateBody(BaseModel):
    lib_type: str  # laws / hazards
    content: str
    source_file: str | None = "manual"


class KbUpdateBody(BaseModel):
    lib_type: str
    case_id: str
    content: str
    source_file: str | None = None


class KbDeleteBody(BaseModel):
    case_id: str


@app.get("/kb")
def list_kb(lib_type: str | None = None):
    cases = _filter_cases(_load_cases(), lib_type)
    return {"cases": cases}


@app.post("/kb/create")
def create_kb(body: KbCreateBody):
    if body.lib_type not in {"laws", "hazards"}:
        raise HTTPException(status_code=400, detail="lib_type must be laws or hazards")

    cases = _load_cases()
    if body.lib_type == "laws":
        new_case = {
            "id": f"law_{uuid.uuid4().hex[:8]}",
            "source_file": body.source_file or "manual",
            "legal_basis": body.content.strip(),
        }
    else:
        new_case = {
            "id": f"hazard_{uuid.uuid4().hex[:8]}",
            "source_file": body.source_file or "manual",
            "issue_text": body.content.strip(),
        }

    cases.append(new_case)
    _save_cases(cases)

    if body.lib_type == "laws":
        _sync_legal_add(new_case)

    return {"status": "success", "case": new_case}


@app.post("/kb/update")
def update_kb(body: KbUpdateBody):
    if body.lib_type not in {"laws", "hazards"}:
        raise HTTPException(status_code=400, detail="lib_type must be laws or hazards")

    cases = _load_cases()
    found = False
    for case in cases:
        if case.get("id") == body.case_id:
            if body.lib_type == "laws":
                case["legal_basis"] = body.content.strip()
            else:
                case["issue_text"] = body.content.strip()
            if body.source_file:
                case["source_file"] = body.source_file
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="case not found")

    _save_cases(cases)

    if body.lib_type == "laws":
        _sync_legal_delete(body.case_id)
        case = next(c for c in cases if c.get("id") == body.case_id)
        _sync_legal_add(case)

    return {"status": "success"}


@app.post("/kb/delete")
def delete_kb(body: KbDeleteBody):
    cases = _load_cases()
    new_cases = [c for c in cases if c.get("id") != body.case_id]
    if len(new_cases) == len(cases):
        raise HTTPException(status_code=404, detail="case not found")

    _save_cases(new_cases)

    # 如果是法律库条目则同步向量库
    if body.case_id.startswith("law_"):
        _sync_legal_delete(body.case_id)

    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
