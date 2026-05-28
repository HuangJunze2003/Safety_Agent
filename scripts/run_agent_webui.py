import inspect
import json
import re
import gradio as gr
import os
import sys
from pathlib import Path
import requests
import gradio.themes.base as gradio_theme_base

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

def _format_thought_block(thought: str) -> str:
    thought = thought.strip()
    if not thought:
        return ""
    return (
        f"<details><summary>🧠 <b>思维链</b> (点击展开)</summary>\n\n"
        f"{thought}\n\n</details>\n\n---\n\n"
    )


def _extract_thought_chain(raw_answer: str, structured: dict | None) -> str:
    structured = structured or {}
    thought = structured.get("思维链")
    if thought:
        return str(thought).strip()

    text = (raw_answer or "").strip()
    if not text:
        return ""

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("思维链"):
                return str(parsed["思维链"]).strip()
        except Exception:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict) and parsed.get("思维链"):
                    return str(parsed["思维链"]).strip()
            except Exception:
                pass

    section_patterns = [
        r"##\s*思维链\s*\n+(.*?)(?=\n##\s|\n\*\*[^*]+\*\*|\n###\s|\Z)",
        r"\*\*思维链\*\*\s*\n+(.*?)(?=\n\*\*[^*]+\*\*|\n###\s|\Z)",
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _remove_thought_section(text: str) -> str:
    if not text:
        return text
    patterns = [
        r"##\s*思维链\s*\n+.*?(?=\n##\s|\n\*\*[^*]+\*\*|\n###\s|\Z)",
        r"\*\*思维链\*\*\s*\n+.*?(?=\n\*\*[^*]+\*\*|\n###\s|\Z)",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.DOTALL)
    return cleaned.strip()


def _prepend_thought_to_reply(reply: str, raw_answer: str, structured: dict | None) -> str:
    thought = _extract_thought_chain(raw_answer, structured)
    if not thought:
        return reply
    body = _remove_thought_section(raw_answer).strip()
    if structured and ("隐患定性" in structured or "法律依据" in structured):
        return _format_thought_block(thought) + reply
    if body and body != reply.strip():
        return _format_thought_block(thought) + body + ("\n\n" if reply else "")
    if body and not reply.strip():
        return _format_thought_block(thought) + body + "\n\n"
    return _format_thought_block(thought) + reply


def _model_api_ready() -> bool:
    api_base = os.getenv("QWEN_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    try:
        resp = requests.get(f"{api_base}/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def build_agent_reply(message: dict) -> str:
    """
    根据多模态输入生成助手回复 Markdown。
    message 格式: {"text": "...", "files": ["图片绝对路径.jpg"]}
    """
    if not _model_api_ready():
        return (
            "⚠️ **模型推理服务未就绪**（`http://127.0.0.1:8000` 无响应）。\n\n"
            "请先执行：`bash scripts/start_all_services.sh`\n\n"
            "若日志出现 CUDA OOM，请使用：`export INFER_MODE=merged` 后重启。"
        )

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
        structured = result.get("structured_output", {})
        raw_answer = result.get("raw_answer", "未能生成结果")
        intent = result.get("intent", "")

        reply = ""

        # 寒暄/身份类：直接展示自然语言，不走隐患 JSON 模板
        if intent == "greeting":
            reply = f"{raw_answer.strip()}\n\n"
        elif intent == "legal_only":
            reply = _prepend_thought_to_reply("", raw_answer, structured)
            if not reply.strip():
                reply = f"{raw_answer.strip()}\n\n"
        # 如果模型成功输出了预期的隐患分析 JSON 结构
        elif isinstance(structured, dict) and ("隐患定性" in structured or "思维链" in structured):
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

            reply = _prepend_thought_to_reply(reply, raw_answer, structured)
                
        else:
            reply = _prepend_thought_to_reply("", raw_answer, structured)
            if not reply.strip():
                reply = f"{raw_answer}\n\n"
        
        # 加上 RAG 检索的数据作为 Debug/参考尾部
        cases = result.get("similar_cases", [])
        laws = result.get("retrieved_laws", [])
        
        if cases or laws:
            reply += "---\n"
            if intent == "legal_only" and laws:
                reply += (
                    "<details><summary>📚 <b>本轮检索条文原文</b>（点击展开，共 "
                    f"{len(laws)} 条）</summary>\n\n"
                )
            else:
                reply += "### 📚 检索增强参考 (RAG Sources)：\n\n"
            if cases:
                reply += "**📷 相似案卷检索:**\n"
                for i, case in enumerate(cases, 1):
                    reply += f"{i}. 相似度得分 {case.get('score', 0):.2f} - `{case.get('image_path', '')}`\n"
            
            if laws:
                if intent != "legal_only":
                    reply += "\n**📜 法律条文检索:**\n"
                for i, law in enumerate(laws, 1):
                    if isinstance(law, dict):
                        source = law.get("source_file", "未知源")
                        content = law.get("legal_basis", "")
                        score = law.get("score", 0)
                        preview = content[:200] + ("..." if len(content) > 200 else "")
                        reply += f"{i}. **[{source}]** (距离: {score:.2f})\n   {preview}\n\n"
                    else:
                        reply += f"{i}. {law}\n"
            if intent == "legal_only" and laws:
                reply += "</details>\n"
            
        return reply
        
    except Exception as e:
        return f"发生运行时错误: {str(e)}"


def _user_display_content(message: dict) -> str:
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if files:
        path = files[0]
        name = Path(path).name if isinstance(path, str) else "image"
        return f"{text}\n\n📷 `{name}`" if text else f"📷 `{name}`"
    return text or "（空消息）"


def handle_chat_submit(
    message: dict | None,
    history: list | None,
) -> tuple[dict, list]:
    history = list(history or [])
    if not message:
        return {"text": "", "files": []}, history

    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        return message, history

    reply = build_agent_reply(message)
    history.append({"role": "user", "content": _user_display_content(message)})
    history.append({"role": "assistant", "content": reply})
    return {"text": "", "files": []}, history


def clear_chat_history() -> tuple[list, dict]:
    return [], {"text": "", "files": []}


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

# 构建界面（强制浅色：同步覆盖 Gradio 的 *_dark 变量，避免跟随系统深色模式）
_THEME_SET_KEYS = set(inspect.signature(gradio_theme_base.Base.set).parameters) - {"self"}


def _build_light_theme() -> gr.themes.ThemeClass:
    light_tokens = {
        "body_background_fill": "#f5f7fb",
        "body_text_color": "#1e293b",
        "body_text_color_subdued": "#64748b",
        "background_fill_primary": "#ffffff",
        "background_fill_secondary": "#f1f5f9",
        "block_background_fill": "#ffffff",
        "block_border_color": "#e2e8f0",
        "block_label_text_color": "#475569",
        "block_title_text_color": "#0f172a",
        "block_title_background_fill": "transparent",
        "panel_background_fill": "#ffffff",
        "panel_border_color": "#e2e8f0",
        "input_background_fill": "#ffffff",
        "input_background_fill_focus": "#ffffff",
        "input_background_fill_hover": "#f8fafc",
        "input_border_color": "#cbd5e1",
        "input_border_color_focus": "#2563eb",
        "button_primary_background_fill": "#2563eb",
        "button_primary_text_color": "#ffffff",
        "button_secondary_background_fill": "#f1f5f9",
        "button_secondary_text_color": "#1e293b",
        "border_color_primary": "#e2e8f0",
        "border_color_accent": "#93c5fd",
        "border_color_accent_subdued": "#bfdbfe",
        "color_accent": "#2563eb",
        "color_accent_soft": "#dbeafe",
        "link_text_color": "#2563eb",
        "code_background_fill": "#f1f5f9",
        "table_text_color": "#1e293b",
        "table_border_color": "#e2e8f0",
        "table_even_background_fill": "#ffffff",
        "table_odd_background_fill": "#f8fafc",
        "table_row_focus": "#dbeafe",
        "accordion_text_color": "#1e293b",
        "checkbox_label_background_fill": "#ffffff",
        "checkbox_label_text_color": "#1e293b",
        "radio_circle": "#2563eb",
    }
    theme_kwargs: dict[str, str] = {}
    for key, value in light_tokens.items():
        if key in _THEME_SET_KEYS:
            theme_kwargs[key] = value
        dark_key = f"{key}_dark"
        if dark_key in _THEME_SET_KEYS:
            theme_kwargs[dark_key] = value
    return gr.themes.Default(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
    ).set(**theme_kwargs)


LIGHT_THEME = _build_light_theme()

APP_CSS = """
:root { color-scheme: light; }

.gradio-container {
    color-scheme: light !important;
    --body-background-fill: #f5f7fb !important;
    --body-text-color: #1e293b !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
    background: #f5f7fb !important;
}

body { margin: 0; background: #f5f7fb !important; }

.app-title h2, .app-title p { margin: 0 !important; }
.app-title h2 { font-size: 1.2rem !important; color: #0f172a !important; }
.app-hint { font-size: 0.8rem !important; color: #64748b !important; margin: 0 0 8px !important; }

/* 对话页：上聊天、下输入 */
.chat-page { gap: 10px !important; }
.chat-history {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.chat-history .bubble-wrap,
.chat-history .message-wrap { color: #1e293b !important; }
.chat-history .user-row,
.chat-history [class*="user"] .message {
    background: #dbeafe !important;
}
.chat-history .bot-row,
.chat-history [class*="bot"] .message {
    background: #f8fafc !important;
}

.chat-input-row { align-items: stretch !important; gap: 8px !important; }
.chat-input-box {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}
.chat-input-box textarea {
    min-height: 48px !important;
    max-height: 96px !important;
}
.chat-send-btn { min-width: 88px !important; height: 48px !important; }

footer { display: none !important; }

/* ========== 知识库页（录入 / 管理）========== */
.kb-page { gap: 12px !important; }
.kb-section-title h3, .kb-section-title p {
    margin: 0 0 8px !important;
    font-size: 1rem !important;
    color: #0f172a !important;
}

/* 单选：避免深色胶囊按钮 */
.kb-page fieldset,
.kb-page .gr-radio {
    background: transparent !important;
    border: none !important;
}
.kb-page .gr-radio label,
.kb-page .form-radio label,
.kb-page label[data-testid] {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
.kb-page .gr-radio label.selected,
.kb-page .gr-radio input:checked + label,
.kb-page .form-radio input:checked + label {
    background: #dbeafe !important;
    color: #0f172a !important;
    border-color: #2563eb !important;
}

/* 表格：强制浅色可读（修复深色底+深色字） */
.kb-table,
.kb-table .wrap,
.kb-table .table-wrap,
.kb-table table,
.kb-table thead,
.kb-table tbody,
.kb-table tr,
.kb-table th,
.kb-table td,
.kb-table [role="grid"],
.kb-table [role="row"],
.kb-table [role="columnheader"],
.kb-table [role="gridcell"] {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
}
.kb-table thead th,
.kb-table [role="columnheader"] {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}
.kb-table tbody tr:nth-child(even) td,
.kb-table tbody tr:nth-child(even) [role="gridcell"] {
    background-color: #f8fafc !important;
}
.kb-table .cell-wrap,
.kb-table .svelte-bzkl0l {
    color: #1e293b !important;
    background: transparent !important;
}

.kb-status textarea {
    min-height: 56px !important;
    max-height: 120px !important;
}
.kb-upload .wrap {
    min-height: 120px !important;
    max-height: 160px !important;
}

.kb-actions { gap: 8px !important; align-items: flex-end !important; }

@media (prefers-color-scheme: dark) {
    .gradio-container { color-scheme: light !important; }
    .chat-history .bubble-wrap { background: #ffffff !important; }
    .kb-table, .kb-table table, .kb-table td, .kb-table th {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
}
"""

with gr.Blocks(theme=LIGHT_THEME, css=APP_CSS, title="安全生产管理智能体") as demo:
    gr.Markdown("## 👷 安全生产管理智能体", elem_classes=["app-title"])
    
    with gr.Tab("智能对话与检测"):
        with gr.Column(elem_classes=["chat-page"]):
            gr.Markdown(
                "上传现场照片并提问，或纯文字咨询法规 / 隐患。",
                elem_classes=["app-hint"],
            )
            chatbot = gr.Chatbot(
                value=[],
                type="messages",
                show_label=False,
                height=560,
                autoscroll=True,
                elem_classes=["chat-history"],
                placeholder="对话将显示在这里…",
            )
            with gr.Row(elem_classes=["chat-input-row"]):
                msg_input = gr.MultimodalTextbox(
                    placeholder="输入问题，可附现场照片…",
                    show_label=False,
                    lines=2,
                    max_lines=6,
                    submit_btn=False,
                    elem_classes=["chat-input-box"],
                    scale=9,
                )
                send_btn = gr.Button("发送", variant="primary", elem_classes=["chat-send-btn"], scale=1)
            clear_btn = gr.Button("清空对话", size="sm")

            submit_inputs = [msg_input, chatbot]
            submit_outputs = [msg_input, chatbot]
            msg_input.submit(handle_chat_submit, submit_inputs, submit_outputs)
            send_btn.click(handle_chat_submit, submit_inputs, submit_outputs)
            clear_btn.click(clear_chat_history, outputs=[chatbot, msg_input])
    
    with gr.Tab("知识库录入"):
        with gr.Column(elem_classes=["kb-page"]):
            gr.Markdown("### 📥 上传新资料到系统数据库", elem_classes=["kb-section-title"])
            with gr.Row(equal_height=True):
                file_input = gr.File(
                    label="选择文件（PDF / Word / TXT）",
                    file_types=None,
                    elem_classes=["kb-upload"],
                    scale=3,
                )
                lib_type = gr.Radio(
                    ["法律法规库", "安全隐患库"],
                    label="目标知识库",
                    value="法律法规库",
                    elem_classes=["kb-radio"],
                    scale=2,
                )
            upload_button = gr.Button(
                "🚀 上传并开始分析 / 入库",
                variant="primary",
            )
            output_txt = gr.Textbox(
                label="操作状态",
                lines=2,
                max_lines=6,
                elem_classes=["kb-status"],
            )

            upload_button.click(
                fn=upload_library_file,
                inputs=[file_input, lib_type],
                outputs=output_txt,
            )

    with gr.Tab("知识库管理"):
        with gr.Column(elem_classes=["kb-page"]):
            gr.Markdown("### 📚 知识库条目管理", elem_classes=["kb-section-title"])
            with gr.Row(elem_classes=["kb-actions"]):
                lib_select = gr.Dropdown(
                    choices=["laws", "hazards"],
                    value="laws",
                    label="库类型",
                    info="laws=法律法规，hazards=安全隐患",
                    scale=4,
                )
                refresh_btn = gr.Button("🔄 刷新列表", variant="secondary", scale=1)

            kb_table = gr.Dataframe(
                headers=["id", "source_file", "legal_basis"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
                max_height=420,
                elem_classes=["kb-table"],
            )
            status_box = gr.Textbox(
                label="操作状态",
                lines=2,
                max_lines=4,
                interactive=False,
                elem_classes=["kb-status"],
            )

            with gr.Accordion("✏️ 新增 / 更新 / 删除", open=False):
                with gr.Row():
                    del_id = gr.Textbox(
                        label="条目 ID（删除或更新时填写）",
                        lines=1,
                        scale=2,
                    )
                    new_source = gr.Textbox(
                        label="来源文件",
                        value="manual",
                        scale=2,
                    )
                new_content = gr.Textbox(
                    label="内容（法律条文或隐患描述）",
                    lines=3,
                    max_lines=8,
                )
                with gr.Row():
                    create_btn = gr.Button("➕ 新增", variant="primary")
                    update_btn = gr.Button("✏️ 更新", variant="secondary")
                    del_btn = gr.Button("🗑️ 删除", variant="stop")

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

