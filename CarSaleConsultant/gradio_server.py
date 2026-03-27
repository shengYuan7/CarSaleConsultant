import time
import gradio as gr
from vectors_retrieval import save_vectors_db, init_chain


def do_user(user_message, history):
    # 使用新的消息格式，处理 history 为空或 None 的情况
    if history is None:
        history = []
    history.append({"role": "user", "content": user_message})
    return '', history


def _extract_text(content):
    """从 Gradio Chatbot 的 content 中提取纯文本（兼容字符串和 list[dict] 格式）"""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                parts.append(str(item['text']))
            elif isinstance(item, str):
                parts.append(item)
        return ' '.join(parts)
    return str(content)


def do_it(history):
    if history is None or len(history) == 0:
        return []
    # 获取最后一个用户消息（Gradio 6 可能返回 content 为 [{"text":"...","type":"text"}] 格式）
    raw_content = history[-1].get("content", "")
    question = _extract_text(raw_content)
    if not question.strip():
        return history

    try:
        # 把用户提问输入给AI机器人
        res = bot.invoke({'input': question})
        resp = res.get('answer')
        # 确保转换为字符串（兼容 LangChain 返回的 TextAccessor 等类型）
        resp = str(resp) if resp is not None else ''
        if not resp:
            resp = '这个问题，我建议你直接问人工客服!'
    except Exception as e:
        resp = f'抱歉，处理您的请求时出了点问题：{str(e)}。建议您稍后重试或联系人工客服。'

    # 添加助手消息
    history.append({"role": "assistant", "content": ""})

    # 流式输出
    for char in resp:
        history[-1]["content"] += char
        time.sleep(0.03)
        yield history


def run_gradio_server():
    # 苹果风格CSS设计
    css = """
    :root {
        --apple-white: #ffffff;
        --apple-light-gray: #f5f5f7;
        --apple-gray: #86868b;
        --apple-dark-gray: #1d1d1f;
        --apple-blue: #0071e3;
        --apple-light-blue: #2997ff;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif;
        background-color: var(--apple-white);
        color: var(--apple-dark-gray);
        margin: 0;
        padding: 0;
    }

    .gradio-container {
        max-width: 920px !important;
        margin: 0 auto !important;
        padding: 40px 20px !important;
        background: var(--apple-white) !important;
    }

    /* 头部标题样式 */
    .apple-header {
        text-align: center;
        margin-bottom: 40px;
        padding: 30px 0;
        border-bottom: 1px solid var(--apple-light-gray);
    }

    .apple-title {
        font-size: 48px;
        font-weight: 700;
        color: var(--apple-dark-gray);
        letter-spacing: -0.5px;
        margin: 0 0 12px 0;
        background: linear-gradient(135deg, var(--apple-dark-gray) 0%, var(--apple-gray) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .apple-subtitle {
        font-size: 21px;
        font-weight: 400;
        color: var(--apple-gray);
        margin: 0;
        line-height: 1.4;
    }

    /* 聊天区域样式 */
    .chat-container {
        background: var(--apple-white);
        border-radius: 20px;
        border: 1px solid var(--apple-light-gray);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        overflow: hidden;
        margin-bottom: 30px;
    }

    .gr-chatbot {
        background: var(--apple-white) !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        min-height: 450px !important;
        padding: 24px !important;
        height: 500px !important;
    }

    /* 输入区域样式 */
    .input-container {
        background: var(--apple-light-gray);
        padding: 24px;
        border-top: 1px solid rgba(0, 0, 0, 0.05);
    }

    .input-row {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-bottom: 12px;
    }

    .button-row {
        display: flex;
        justify-content: flex-end;
        gap: 12px;
    }

    .apple-textbox {
        width: 100% !important;
        border-radius: 12px !important;
        border: 1px solid var(--apple-gray) !important;
        background: var(--apple-white) !important;
        padding: 16px 20px !important;
        font-size: 17px !important;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
    }

    .apple-textbox:focus {
        border-color: var(--apple-blue) !important;
        box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1) !important;
        outline: none !important;
    }

    .apple-button {
        border-radius: 12px !important;
        background: var(--apple-blue) !important;
        color: var(--apple-white) !important;
        border: none !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        min-width: 100px !important;
    }

    .apple-button:hover {
        background: var(--apple-light-blue) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 113, 227, 0.3) !important;
    }

    .apple-button:active {
        transform: translateY(0) !important;
    }

    .clear-button {
        background: transparent !important;
        color: var(--apple-gray) !important;
        border: 1px solid var(--apple-gray) !important;
        padding: 12px 24px !important;
        min-width: 100px !important;
    }

    .clear-button:hover {
        background: var(--apple-light-gray) !important;
        color: var(--apple-dark-gray) !important;
        box-shadow: none !important;
    }

    /* 快捷问题区域样式 */
    .quick-questions-section {
        background: var(--apple-light-gray);
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
    }

    .quick-questions-title {
        font-size: 19px;
        font-weight: 600;
        color: var(--apple-dark-gray);
        margin-bottom: 16px;
        text-align: center;
    }

    .quick-questions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
    }

    .question-chip {
        background: var(--apple-white);
        border: 1px solid var(--apple-light-gray);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 15px;
        color: var(--apple-dark-gray);
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }

    .question-chip:hover {
        background: var(--apple-blue);
        color: var(--apple-white);
        border-color: var(--apple-blue);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.2);
    }

    /* 响应式设计 */
    @media (max-width: 768px) {
        .apple-title {
            font-size: 36px;
        }

        .apple-subtitle {
            font-size: 18px;
        }

        .button-row {
            justify-content: space-between;
        }

        .quick-questions-grid {
            grid-template-columns: 1fr;
        }
    }
    """

    # 常用客户问题
    common_questions = [
        "宝马的发动机性能怎么样？会不会烧机油？",
        "奔驰E级的后排真的像网上说的那么舒服吗？",
        "奥迪Quattro和宝马的四驱，哪个更厉害？",
        "贷款买车，利息怎么算的？有没有免息？",
        "我想买个车跑婚庆，买哪款最合适？",
        "我开惯了日系车，换德系会不会不习惯？",
    ]

    with gr.Blocks(title='小智汽车顾问 | 专业购车顾问') as instance:
        with gr.Column(elem_classes="gradio-container"):
            # 苹果风格的头部
            with gr.Column(elem_classes="apple-header"):
                gr.HTML("""
                    <h1 class="apple-title">小智汽车顾问</h1>
                    <p class="apple-subtitle">专业购车顾问，为您提供一对一的购车咨询服务</p>
                """)

            # 聊天容器
            with gr.Column(elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder='👋 您好！我是您的专业购车顾问，很高兴为您服务...',
                    elem_classes="gr-chatbot"
                )

            # 输入区域
            with gr.Column(elem_classes="input-container"):
                # 第一行：文本输入框（占据整个宽度）
                with gr.Row(elem_classes="input-row"):
                    msg = gr.Textbox(
                        placeholder="请输入您的问题...",
                        container=False,
                        autofocus=True,
                        elem_classes="apple-textbox",
                        show_label=False,
                        scale=12  # 确保输入框占据最大宽度
                    )

                # 第二行：按钮
                with gr.Row(elem_classes="button-row"):
                    submit_btn = gr.Button(
                        "发送",
                        elem_classes="apple-button"
                    )
                    clear = gr.Button(
                        "清除对话",
                        elem_classes="apple-button clear-button"
                    )

            # 快捷问题区域
            with gr.Column(elem_classes="quick-questions-section"):
                gr.HTML("""
                    <div class="quick-questions-title">💡 常用问题快捷提问</div>
                """)

                # 创建两列布局的快捷问题
                with gr.Row():
                    with gr.Column():
                        for i, question in enumerate(common_questions[:5]):
                            gr.Button(
                                question,
                                elem_classes="question-chip",
                                size="sm"
                            ).click(
                                lambda q=question: q,
                                outputs=msg
                            )

                    with gr.Column():
                        for i, question in enumerate(common_questions[5:]):
                            gr.Button(
                                question,
                                elem_classes="question-chip",
                                size="sm"
                            ).click(
                                lambda q=question: q,
                                outputs=msg
                            )

        # 事件绑定（流式输出需要启用 queue）
        msg.submit(
            do_user,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            do_it,
            chatbot,
            chatbot
        )

        submit_btn.click(
            do_user,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            do_it,
            chatbot,
            chatbot
        )

        clear.click(
            lambda: [],
            None,
            chatbot,
            queue=False
        )

    # 启动服务（默认 18888 端口，可通过环境变量 GRADIO_SERVER_PORT 覆盖）
    instance.queue()
    import os as _os
    _port = int(_os.environ.get('GRADIO_SERVER_PORT', 18888))
    instance.launch(
        server_name='0.0.0.0',
        server_port=_port,
        css=css,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
    )


def init():
    save_vectors_db()
    global bot
    bot = init_chain()


if __name__ == '__main__':
    # 初始化AI机器人
    init()
    run_gradio_server()