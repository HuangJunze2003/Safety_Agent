import gradio as gr
import os
import sys
from pathlib import Path

# 将 src 目录加入环境变量以导入你的 agent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from agent.workflow import build_agent_from_env
except Exception as e:
    print(f"无法加载 Agent 工作流，错误详情: {e}")
    sys.exit(1)

# 初始化智能体
print("正在初始化智能体，这可能需要一些时间（尤其是加载模型或连接数据库时）...")
agent = build_agent_from_env()
print("智能体初始化完成！")

def chat_with_agent(message, history):
    """
    处理界面传来的多模态消息：
    message 格式: {"text": "用户的文字描述", "files": ["图片绝对路径.jpg"]}
    """
    text_query = message.get("text", "请分析图片存在的安全隐患。")
    files = message.get("files", [])
    
    # 策略调整：如果是纯文本咨询（没有图片），且包含法律关键词或寒暄/自我介绍，则允许不传图片
    legal_keywords = ["法律", "法规", "条例", "规定", "怎么说", "条款", "内容是什么", "第一条", "法", "法条", "你是谁", "你能做", "介绍", "你好"]
    is_legal_query = any(kw in text_query for kw in legal_keywords)
    
    # 如果既没有图片，也不是法律/寒暄咨询，则提示上传图片
    if not files and not is_legal_query:
        return "⚠️ 请上传一张需要进行安全检查的现场照片，或者输入您想咨询的安全生产法律法规问题。"
    
    # 获取图片路径（如果有的话）
    image_path = files[0] if files else None
    
    try:
        # 调用分析流，如果没有图片则传入 None
        result = agent.analyze(image_path=image_path, question=text_query)
        
        # 提取结果并排版展示给用户
        # 尝试提取结构化输出以优先展示友好的 Markdown 格式
        structured = result.get("structured_output", {})
        raw_answer = result.get("raw_answer", "未能生成结果")
        
        reply = ""
        
        # 如果模型成功输出了预期的 JSON 结构
        if isinstance(structured, dict) and ("隐患定性" in structured or "思维链" in structured):
            # 增加思维链展示，使用 Gradio 原生支持的 Markdown details/summary 标签实现折叠效果
            # 这样就像常见的大模型客户端一样，可以点开查看“思考过程”
            thought = structured.get('思维链', '')
            if thought:
                reply += f"<details><summary>🧠 <b>思考过程</b> (点击展开)</summary>\n\n{thought}\n\n</details>\n\n---\n\n"

            reply += "### 🤖 智能体分析报告\n\n"

            if structured.get('隐患定性'):
                reply += f"**⚠️ 隐患定性:** {structured.get('隐患定性', '未知')}\n\n"
            
            # 法律依据模块
            laws_output = structured.get('法律依据', [])
            if laws_output:
                reply += "**📖 法律依据:**\n"
                for law in laws_output:
                    # 兼容字典格式
                    if isinstance(law, dict):
                        reply += f"- **{law.get('条款号', '')}**: {law.get('条款内容', '')}\n"
                    else:
                        reply += f"- {law}\n"
                reply += "\n"
                
            # 整改措施模块
            measures = structured.get('整改措施', [])
            if measures:
                reply += "**🛠️ 整改措施:**\n"
                for m in measures:
                    if isinstance(m, dict):
                        reply += f"- **[{m.get('措施类型', '措施')}]** {m.get('措施内容', '')}\n"
                    else:
                        reply += f"- {m}\n"
                reply += "\n"
                
            # 参考案卷
            refs = structured.get('参考案例', [])
            if refs:
                reply += "**📚 参考案例:**\n"
                for ref in refs:
                    reply += f"- {ref}\n"
                reply += "\n"
                
        else:
            # 如果模型输出不是 JSON，就原样打印
            reply += f"{raw_answer}\n\n"
        
        # 加上 RAG 检索的数据作为 Debug/参考尾部
        cases = result.get("similar_cases", [])
        laws = result.get("retrieved_laws", [])
        
        if cases or laws:
            reply += "---\n### 📚 检索增强参考 (RAG Sources)：\n\n"
            if cases:
                reply += "**📷 相似案卷检索:**\n"
                for i, case in enumerate(cases, 1):
                    reply += f"{i}. 相似度得分 {case.get('score', 0):.2f} - `{case.get('image_path', '')}`\n"
            
            if laws:
                reply += "\n**📜 法律条文检索:**\n"
                for i, law in enumerate(laws, 1):
                    # 如果 law 是字典（通常是这样），则格式化它
                    if isinstance(law, dict):
                        source = law.get("source_file", "未知源")
                        content = law.get("legal_basis", "")
                        score = law.get("score", 0)
                        reply += f"{i}. **[{source}]** (得分: {score:.2f})\n   {content[:200]}...\n\n"
                    else:
                        reply += f"{i}. {law}\n"
            
        return reply
        
    except Exception as e:
        return f"发生运行时错误: {str(e)}"

# 构建 Gradio 多模态交互界面
demo = gr.ChatInterface(
    fn=chat_with_agent,
    multimodal=True,
    title="👷 安全生产管理智能体 (Qwen-VL Agent)",
    description="请上传现场监控截图或工作环境照片，并输入查询问题，智能体将连接安全法规知识库为您进行多模态检索与违章分析。",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # 启动网页前端
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
