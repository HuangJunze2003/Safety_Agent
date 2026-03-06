import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.workflow import AgentConfig, LegalClauseRetriever

def test_retrieval():
    print("⏳ 正在初始化法条检索器...")
    config = AgentConfig(project_root=str(PROJECT_ROOT))
    retriever = LegalClauseRetriever(
        project_root=config.project_root,
        metadata_path=config.cases_metadata_path,
        persist_directory=config.legal_db_dir,
        embedding_model_name=config.bge_model_name
    )
    
    query = "安全生产监督管理暂行规定"
    print(f"🔍 正在检索关键词: '{query}'...")
    results = retriever.search(query, top_k=5)
    
    if not results:
        print("❌ 未检索到任何结果。")
        return

    print(f"✅ 找到 {len(results)} 条相关法条：\n")
    for i, res in enumerate(results):
        print(f"--- 结果 {i+1} (得分: {res.get('score', 'N/A'):.4f}) ---")
        print(f"来源文件: {res.get('source_file')}")
        content = res.get('legal_basis', '')
        # 只打印前 200 个字符
        print(f"内容摘要: {content[:200]}...")
        print()

if __name__ == "__main__":
    test_retrieval()
