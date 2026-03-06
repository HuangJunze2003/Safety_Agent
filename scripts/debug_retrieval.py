
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/graduation_prj").resolve()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.workflow import AgentConfig, LegalClauseRetriever

def test_specific_law():
    print("⏳ 正在初始化法条检索器...")
    config = AgentConfig(project_root=str(PROJECT_ROOT))
    retriever = LegalClauseRetriever(
        project_root=config.project_root,
        metadata_path=config.cases_metadata_path,
        persist_directory=config.legal_db_dir,
        embedding_model_name=config.bge_model_name
    )
    
    query = "劳动法第十条是什么"
    print(f"🔍 正在检索关键词: '{query}'...")
    results = retriever.search(query, top_k=5)
    
    if not results:
        print("❌ 未检索到任何结果。")
        return

    print(f"✅ 找到 {len(results)} 条相关法条：\n")
    for i, res in enumerate(results):
        print(f"[{i+1}] (得分: {res['score']:.3f})")
        print(f"来源: {res.get('source_file', '未知')}")
        print(f"内容: {res['legal_basis'][:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    test_specific_law()
