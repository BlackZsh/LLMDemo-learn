"""
文本处理API完整Demo
演示硅基流动文本对话API的使用
"""
import gradio as gr
from api.text_api import text_api
from config.settings import settings
from loguru import logger
import sys

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)


def convert_history_to_tuples(messages):
    """
    将messages格式转换为tuples格式（用于API调用）
    
    Args:
        messages: OpenAI格式的消息列表 [{"role": "user", "content": "..."}, ...]
        
    Returns:
        tuples格式 [("user_msg", "bot_msg"), ...]
    """
    history = []
    i = 0
    while i < len(messages):
        if i + 1 < len(messages) and messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            history.append((messages[i]["content"], messages[i+1]["content"]))
            i += 2
        else:
            i += 1
    return history


def chat_response(message, history):
    """
    处理聊天响应（非流式）
    
    Args:
        message: 用户输入的消息 (OpenAI格式)
        history: 历史对话记录 (OpenAI格式)
        
    Yields:
        更新后的历史记录
    """
    try:
        # 验证配置
        settings.validate()
        
        # 立即添加用户消息到历史
        history.append({"role": "user", "content": message})
        yield history
        
        # 添加空的助手消息
        history.append({"role": "assistant", "content": ""})
        yield history
        
        # 转换历史格式用于API调用
        chat_history = convert_history_to_tuples(history[:-2])  # 不包括当前消息
        
        # 调用API
        logger.info(f"用户输入: {message}")
        response = text_api.chat(message=message, history=chat_history)
        logger.info(f"API响应: {response[:100]}...")
        
        # 更新助手回复
        history[-1]["content"] = response
        yield history
        
    except ValueError as e:
        error_msg = f"⚠️ 配置错误: {str(e)}\n\n请在项目根目录创建 .env 文件并配置 SILICONFLOW_API_KEY"
        logger.error(error_msg)
        history[-1]["content"] = error_msg
        yield history
    except Exception as e:
        error_msg = f"❌ 系统错误: {str(e)}"
        logger.error(error_msg)
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = error_msg
        yield history


def chat_stream_response(message, history):
    """
    处理流式聊天响应
    
    Args:
        message: 用户输入的消息 (OpenAI格式)
        history: 历史对话记录 (OpenAI格式)
        
    Yields:
        流式更新的历史记录
    """
    try:
        # 验证配置
        settings.validate()
        
        # 立即添加用户消息到历史
        history.append({"role": "user", "content": message})
        yield history
        
        # 添加空的助手消息
        history.append({"role": "assistant", "content": ""})
        yield history
        
        # 转换历史格式用于API调用
        chat_history = convert_history_to_tuples(history[:-2])  # 不包括当前消息
        
        # 调用流式API
        logger.info(f"用户输入(流式): {message}")
        full_response = ""
        for chunk in text_api.chat_stream(message=message, history=chat_history):
            full_response += chunk
            history[-1]["content"] = full_response
            yield history
        
        logger.info(f"流式API响应完成，总长度: {len(full_response)}")
        
    except ValueError as e:
        error_msg = f"⚠️ 配置错误: {str(e)}\n\n请在项目根目录创建 .env 文件并配置 SILICONFLOW_API_KEY"
        logger.error(error_msg)
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = error_msg
        yield history
    except Exception as e:
        error_msg = f"❌ 系统错误: {str(e)}"
        logger.error(error_msg)
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = error_msg
        yield history


def create_demo():
    """创建Gradio演示界面"""
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        # 标题和说明
        gr.HTML("""
        <div class="header">
            <h1>🤖 硅基流动文本对话 Demo</h1>
            <p>基于 OpenAI 兼容接口的大语言模型对话系统</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 📋 功能说明
                - ✅ 支持多轮对话
                - ✅ 历史记录保存
                
                ### ⚙️ 当前配置
                """)
                
                # 显示当前配置
                with gr.Group():
                    gr.Markdown(f"""
                    - **模型**: `{settings.TEXT_MODEL}`
                    - **最大Token**: `{settings.MAX_TOKENS}`
                    - **温度**: `{settings.TEMPERATURE}`
                    - **Top-P**: `{settings.TOP_P}`
                    """)
                
                # 示例提示词
                gr.Markdown("""
                ### 💡 示例提示词
                - 请介绍一下硅基流动
                - 解释一下什么是机器学习
                - 帮我写一首关于春天的诗
                """)
        
            with gr.Column(scale=2):
                # 聊天界面 - 使用新的messages格式
                chatbot = gr.Chatbot(
                    label="对话窗口",
                    height=500,
                    type="messages",  # 使用OpenAI格式，避免弃用警告
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入消息",
                        placeholder="请输入消息...",
                        lines=2,
                        scale=4
                    )
                    
                with gr.Row():
                    # submit_btn = gr.Button("📤 发送", variant="primary", scale=1)
                    stream_btn = gr.Button("📤 发送", variant="secondary", scale=1)
                    clear_btn = gr.Button("🗑️ 清空", scale=1)
        
        # 底部信息
        gr.Markdown("""
        ---
        ### 📖 API文档
        - [硅基流动API文档](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions)
        - 本Demo使用OpenAI兼容接口标准
        
        ### 🔑 配置说明
        1. 在项目根目录创建 `.env` 文件
        2. 配置 `SILICONFLOW_API_KEY=你的API密钥`
        3. API密钥获取: https://cloud.siliconflow.cn/account/ak
        """)
        
        # 事件绑定 - 普通发送（非流式，但立即显示用户消息）
        def handle_submit_wrapper(message, history):
            """处理普通提交的包装函数"""
            if not message or not message.strip():
                return history, message
            
            # 先清空输入框，然后逐步更新对话
            yield history, ""
            
            # 调用生成器函数，逐步更新UI
            for updated_history in chat_response(message, history):
                yield updated_history, ""
        
        # submit_btn.click(
        #     fn=handle_submit_wrapper,
        #     inputs=[msg, chatbot],
        #     outputs=[chatbot, msg]
        # )
        
        # 事件绑定 - 流式发送（推荐使用，体验更好）
        def handle_stream_wrapper(message, history):
            """处理流式提交的包装函数"""
            if not message or not message.strip():
                return history, message
            
            # 先清空输入框，然后流式更新对话
            yield history, ""
            
            # 流式获取响应
            for updated_history in chat_stream_response(message, history):
                yield updated_history, ""
        
        # Enter键提交 - 使用流式（推荐）
        msg.submit(
            fn=handle_stream_wrapper,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        # 流式发送按钮
        stream_btn.click(
            fn=handle_stream_wrapper,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        # 清空按钮
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg])
    
    return demo


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("启动硅基流动文本对话 Demo")
    logger.info("=" * 60)
    
    try:
        # 验证配置
        settings.validate()
        logger.info(f"✅ 配置验证通过")
        logger.info(f"📝 使用模型: {settings.TEXT_MODEL}")
        logger.info(f"🔗 API地址: {settings.SILICONFLOW_BASE_URL}")
        
    except ValueError as e:
        logger.error(f"❌ 配置错误: {e}")
        logger.info("💡 请按以下步骤配置:")
        logger.info("   1. 复制 env_template.txt 为 .env")
        logger.info("   2. 编辑 .env 文件，填入你的 API Key")
        logger.info("   3. API Key获取地址: https://cloud.siliconflow.cn/account/ak")
        return
    
    # 创建并启动Demo
    demo = create_demo()
    
    logger.info("🚀 启动 Gradio 服务...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )


if __name__ == "__main__":
    main()

