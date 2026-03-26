import gradio as gr
import os
import sys
from pathlib import Path
import requests

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

    # 服务入口统一采用 Agent 侧意图识别，避免前后端规则不一致。
    # 仅在识别为隐患分析且无图片时，提示用户上传现场图。
    has_image = bool(files)
    intent = agent.detect_intent(question=text_query, has_image=has_image)
    if not has_image and intent == "hazard_analysis":
        return "⚠️ 请上传一张需要进行安全检查的现场照片，或者改为咨询法规条文/通用合规问题。"
    
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

def upload_library_file(file, lib_type):
    """
    通过 Gradio 界面上传文件到知识库
    """
    if file is None:
        return "请先选择需要上传的文件。"
    
    file_path = file.name
    import requests
    try:
        url = "http://127.0.0.1:8001/upload"
        files = {'file': open(file_path, 'rb')}
        data = {'lib_type': "laws" if lib_type == "法律法规库" else "hazards"}
        
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            return response.json().get("message", "上传并处理成功！")
        else:
            return f"上传失败: {response.text}"
    except Exception as e:
        return f"发生错误: {str(e)}"


API_BASE = "http://127.0.0.1:8001"


def kb_fetch(lib_type):
    try:
        resp = requests.get(f"{API_BASE}/kb", params={"lib_type": lib_type})
        if resp.status_code != 200:
            return [], f"查询失败: {resp.text}"
        cases = resp.json().get("cases", [])
        # 只展示关键信息，使用 list-of-lists 适配 gr.Dataframe
        simplified: list[list[str]] = []
        for c in cases:
            simplified.append([
                c.get("id", ""),
                c.get("source_file", ""),
                (c.get("legal_basis", c.get("issue_text", "")) or "")[:200],
            ])
        return simplified, "查询成功"
    except Exception as e:
        return [], f"请求出错: {e}"


def kb_create(lib_type, content, source_file):
    if not content:
        return [], "内容不能为空"
    try:
        resp = requests.post(f"{API_BASE}/kb/create", json={
            "lib_type": lib_type,
            "content": content,
            "source_file": source_file or "manual",
        })
        if resp.status_code != 200:
            return [], f"新增失败: {resp.text}"
        return kb_fetch(lib_type)
    except Exception as e:
        return [], f"请求出错: {e}"


def kb_update(lib_type, case_id, content, source_file):
    if not case_id or not content:
        return [], "ID 和内容均不能为空"
    try:
        resp = requests.post(f"{API_BASE}/kb/update", json={
            "lib_type": lib_type,
            "case_id": case_id,
            "content": content,
            "source_file": source_file or None,
        })
        if resp.status_code != 200:
            return [], f"更新失败: {resp.text}"
        return kb_fetch(lib_type)
    except Exception as e:
        return [], f"请求出错: {e}"


def kb_delete(lib_type, case_id):
    if not case_id:
        return [], "ID 不能为空"
    try:
        resp = requests.post(f"{API_BASE}/kb/delete", json={"case_id": case_id})
        if resp.status_code != 200:
            return [], f"删除失败: {resp.text}"
        return kb_fetch(lib_type)
    except Exception as e:
        return [], f"请求出错: {e}"

# 构建界面
CHAT_CSS = """
:root { --page-padding: 12px; }
body { margin: 0; }
.gradio-container { max-width: 100% !important; padding: var(--page-padding) !important; min-height: 100vh; }
.chat-panel { display: flex; flex-direction: column; gap: 12px; min-height: calc(100vh - 140px); }
.chat-panel .chatbot { flex: 1 1 auto; min-height: 60vh; }
.chat-panel .message-row { width: 100% !important; }
.chat-panel textarea { min-height: 140px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CHAT_CSS) as demo:
    gr.Markdown("# 👷 安全生产管理智能体 (Qwen-VL Agent)")
    
    with gr.Tab("智能对话与检测"):
        with gr.Column(elem_classes=["chat-panel"]):
            gr.ChatInterface(
                fn=chat_with_agent,
                multimodal=True,
                description="请上传现场监控截图或工作环境照片，并输入查询问题，智能体将为您进行多模态检索与违章分析。",
                fill_height=True, # 启用填充高度以实现自适应
            )
    
    with gr.Tab("知识库录入"):
        gr.Markdown("### 📥 上传新资料到系统数据库")
        with gr.Row():
            # 放宽前端文件类型限制，避免浏览器误判；后台仍会按支持的类型解析
            file_input = gr.File(
                label="选择文件 (建议 PDF/Word/TXT)",
                file_types=None,
            )
            lib_type = gr.Radio(["法律法规库", "安全隐患库"], label="目标知识库", value="法律法规库")
        
        upload_button = gr.Button("🚀 上传并开始分析/入库")
        output_txt = gr.Textbox(label="操作状态")
        
        upload_button.click(
            fn=upload_library_file,
            inputs=[file_input, lib_type],
            outputs=output_txt
        )

    with gr.Tab("知识库管理"):
        gr.Markdown("### 📚 查看 / 增删改 知识库条目")
        lib_select = gr.Dropdown(["laws", "hazards"], value="laws", label="库类型 (laws=法律法规, hazards=安全隐患)")
        kb_table = gr.Dataframe(headers=["id", "source_file", "legal_basis"], interactive=False)
        status_box = gr.Textbox(label="操作状态", interactive=False)

        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新列表")
            del_id = gr.Textbox(label="删除/更新用 ID", lines=1)
            del_btn = gr.Button("🗑️ 删除")
        with gr.Row():
            new_content = gr.Textbox(label="内容 (法律条文或隐患描述)", lines=4)
            new_source = gr.Textbox(label="来源文件", value="manual")
        with gr.Row():
            create_btn = gr.Button("➕ 新增")
            update_btn = gr.Button("✏️ 更新")

        def _wrap_fetch(lib_type):
            data, msg = kb_fetch(lib_type)
            return data, msg

        refresh_btn.click(_wrap_fetch, inputs=lib_select, outputs=[kb_table, status_box])

        create_btn.click(
            lambda lt, c, s: kb_create(lt, c, s),
            inputs=[lib_select, new_content, new_source],
            outputs=[kb_table, status_box],
        )

        update_btn.click(
            lambda lt, cid, c, s: kb_update(lt, cid, c, s),
            inputs=[lib_select, del_id, new_content, new_source],
            outputs=[kb_table, status_box],
        )

        del_btn.click(
            lambda lt, cid: kb_delete(lt, cid),
            inputs=[lib_select, del_id],
            outputs=[kb_table, status_box],
        )

        # 首次加载自动刷新
        demo.load(_wrap_fetch, inputs=lib_select, outputs=[kb_table, status_box])

if __name__ == "__main__":
    # 启动网页前端
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

