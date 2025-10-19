"""
硅基流动多模态AI助手Demo - 完整版
集成文本对话、语音合成、图像识别功能
"""
import gradio as gr
import os
import time
from pathlib import Path
from loguru import logger
import sys

# 导入API模块
from api.text_api import text_api
from api.tts_api import tts_api
from api.vision_api import vision_api
from config.settings import settings

# ==================== 配置日志 ====================
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)


# ==================== 工具函数 ====================

def ensure_output_dirs():
    """
    确保输出目录存在
    创建音频和图片的保存目录
    """
    Path(settings.AUDIO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.IMAGE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def convert_history_to_tuples(messages):
    """
    将OpenAI格式的消息列表转换为元组格式
    
    Args:
        messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        
    Returns:
        [("user_msg", "assistant_msg"), ...]
    
    说明:
        这个转换是为了兼容text_api的历史记录格式
    """
    history = []
    i = 0
    while i < len(messages):
        # 每次处理一对user-assistant消息
        if i + 1 < len(messages) and messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            history.append((messages[i]["content"], messages[i+1]["content"]))
            i += 2
        else:
            i += 1
    return history


# ==================== 文本对话功能 ====================

def chat_stream_response(message, history):
    """
    处理流式聊天响应（纯文本对话，参考 demo_text.py 的清晰逻辑）
    
    Args:
        message: 用户输入的消息
        history: 历史对话记录（OpenAI格式）
        
    Yields:
        更新的历史记录
        
    功能流程:
        1. 立即添加用户消息到历史 → 用户立即看到自己的输入
        2. 添加空的助手消息 → 准备接收流式输出
        3. 流式获取AI回复 → 逐字显示，体验流畅
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
        
        logger.info(f"流式响应完成，总长度: {len(full_response)}")
        
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


def synthesize_speech(text):
    """
    语音合成功能（独立函数，作为可选的后处理步骤）
    
    Args:
        text: 要合成语音的文本
        
    Returns:
        str: 音频文件路径，失败返回 None
    """
    try:
        if not text or not text.strip():
            return None
            
        logger.info("开始语音合成...")
        
        # 生成唯一的音频文件名
        timestamp = int(time.time())
        audio_filename = f"tts_{timestamp}.wav"
        audio_path = os.path.join(settings.AUDIO_OUTPUT_DIR, audio_filename)
        
        # 调用TTS API
        tts_api.synthesize(text=text, save_path=audio_path)
        
        logger.info(f"✅ 语音合成成功: {audio_path}")
        return audio_path
        
    except Exception as e:
        logger.warning(f"⚠️ 语音合成失败: {str(e)}")
        return None


# ==================== 图像识别功能 ====================

def analyze_image(image, question):
    """
    分析图片内容
    
    Args:
        image: 图片文件（来自摄像头或上传）
                - None: 没有图片
                - str: 图片文件路径
                - numpy.ndarray: 图片数组（摄像头拍摄）
        question: 用户对图片的提问
                 - 如果为空，默认描述图片内容
                 - 如果有内容，回答用户问题
        
    Returns:
        str: 图片分析结果
        
    功能说明:
        1. 支持摄像头拍照
        2. 支持上传本地图片
        3. 可以描述图片内容
        4. 可以回答关于图片的问题
    """
    try:
        # ===== 步骤1: 检查图片是否存在 =====
        if image is None:
            return "⚠️ 请先上传图片或使用摄像头拍照"
        
        # ===== 步骤2: 处理不同类型的图片输入 =====
        import numpy as np
        from PIL import Image
        
        if isinstance(image, np.ndarray):
            # 如果是numpy数组（来自摄像头），先保存为临时文件
            logger.info("处理摄像头图片")
            
            # 转换为PIL Image
            pil_image = Image.fromarray(image)
            
            # 保存为临时文件
            timestamp = int(time.time())
            temp_image_path = os.path.join(settings.IMAGE_OUTPUT_DIR, f"camera_{timestamp}.jpg")
            pil_image.save(temp_image_path)
            
            image_source = temp_image_path
            logger.info(f"摄像头图片已保存: {temp_image_path}")
            
        else:
            # 如果是文件路径，直接使用
            image_source = image
            logger.info(f"处理上传图片: {image_source}")
        
        # ===== 步骤3: 构建提示词 =====
        if question and question.strip():
            # 用户有具体问题
            prompt = question
            logger.info(f"用户问题: {question}")
        else:
            # 默认描述图片
            prompt = None  # vision_api会使用默认提示词
            logger.info("使用默认描述模式")
        
        # ===== 步骤4: 调用Vision API =====
        logger.info("开始分析图片...")
        description = vision_api.describe_image(
            image_source=image_source,
            prompt=prompt,
            detail="auto"  # 自动选择分析详细程度
        )
        
        logger.info(f"图片分析完成，结果长度: {len(description)}")
        return description
        
    except Exception as e:
        error_msg = f"❌ 图片分析失败: {str(e)}"
        logger.error(error_msg)
        return error_msg


# ==================== Gradio界面 ====================

def create_demo():
    """
    创建Gradio多模态演示界面
    
    界面结构:
        - Tab 1: 文本对话（支持语音播报）
        - Tab 2: 图像识别（支持摄像头和上传）
    """
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
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
    .info-box {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="硅基流动多模态AI助手Demo") as demo:
        
        # ==================== 页面标题 ====================
        gr.HTML("""
        <div class="header">
            <h1>🤖 多模态AI助手 Demo</h1>
            <p>文本对话 · 语音合成 · 图像识别 - 一站式AI体验</p>
        </div>
        """)
        
        # ==================== Tab 1: 文本对话 ====================
        with gr.Tab("💬 智能对话"):
            with gr.Row():
                # 左侧：配置和说明
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📋 功能说明
                    - ✅ 多轮对话记忆
                    - ✅ 流式实时输出
                    - 🔊 语音播报（可选）
                    
                    ### ⚙️ 当前配置
                    """)
                    
                    with gr.Group():
                        gr.Markdown(f"""
                        - **文本模型**: `{settings.TEXT_MODEL}`
                        - **语音模型**: `{settings.TTS_MODEL}`
                        - **最大Token**: `{settings.MAX_TOKENS}`
                        - **温度**: `{settings.TEMPERATURE}`
                        """)
                    
                    gr.Markdown("""
                    ### 💡 示例问题
                    - 请介绍一下硅基流动
                    - 写一个Python快速排序
                    - 解释什么是大语言模型
                    - 帮我写首关于AI的诗
                    """)
                
                # 右侧：对话界面
                with gr.Column(scale=2):
                    # 聊天窗口（使用messages格式）
                    chatbot = gr.Chatbot(
                        label="对话窗口",
                        height=450,
                        type="messages",  # OpenAI格式
                        avatar_images=(None, "🤖")
                    )
                    
                    # 音频播放器
                    audio_output = gr.Audio(
                        label="语音播报",
                        type="filepath",
                        autoplay=True,  # 自动播放
                        visible=True
                    )
                    
                    # 输入区域
                    with gr.Row():
                        msg = gr.Textbox(
                            label="输入消息",
                            placeholder="请输入消息...",
                            lines=2,
                            scale=4
                        )
                    
                    # 控制按钮
                    with gr.Row():
                        enable_tts = gr.Checkbox(
                            label="🔊 语音播报",
                            value=False,
                            info="勾选后将回复转为语音"
                        )
                        send_btn = gr.Button("📤 发送", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑️ 清空", scale=1)
            
            # ===== 事件绑定 =====
            
            # 包装函数：处理流式响应（参考 demo_text.py 的清晰结构）
            def handle_stream_wrapper(message, history, enable_tts):
                """
                处理流式提交的包装函数
                
                功能:
                    1. 检查输入是否为空
                    2. 清空输入框
                    3. 流式更新对话
                    4. （可选）生成语音
                """
                if not message or not message.strip():
                    return history, "", None
                
                # 先清空输入框
                yield history, "", None
                
                # 流式获取文本响应
                final_history = None
                for updated_history in chat_stream_response(message, history):
                    final_history = updated_history
                    yield updated_history, "", None
                
                # 如果启用TTS，在文本响应完成后生成语音
                audio_path = None
                if enable_tts and final_history and final_history[-1]["role"] == "assistant":
                    response_text = final_history[-1]["content"]
                    audio_path = synthesize_speech(response_text)
                
                # 最终输出（带或不带音频）
                yield final_history, "", audio_path
            
            # Enter键提交
            msg.submit(
                fn=handle_stream_wrapper,
                inputs=[msg, chatbot, enable_tts],
                outputs=[chatbot, msg, audio_output]
            )
            
            # 发送按钮
            send_btn.click(
                fn=handle_stream_wrapper,
                inputs=[msg, chatbot, enable_tts],
                outputs=[chatbot, msg, audio_output]
            )
            
            # 清空按钮
            clear_btn.click(
                fn=lambda: ([], "", None),
                outputs=[chatbot, msg, audio_output]
            )
        
        # ==================== Tab 2: 图像识别 ====================
        with gr.Tab("🖼️ 图像识别"):
            with gr.Row():
                # 左侧：图片输入
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📸 图片来源
                    支持两种方式输入图片:
                    """)
                    
                    with gr.Tabs():
                        # 摄像头拍照
                        with gr.Tab("📷 摄像头"):
                            camera_input = gr.Image(
                                label="摄像头",
                                sources=["webcam"],  # 只允许摄像头
                                type="numpy"
                            )
                        
                        # 上传图片
                        with gr.Tab("📁 上传"):
                            upload_input = gr.Image(
                                label="上传图片",
                                sources=["upload"],  # 只允许上传
                                type="filepath"
                            )
                    
                    gr.Markdown("""
                    ### 💡 使用说明
                    1. **摄像头模式**:
                       - 点击"📷摄像头"标签
                       - 点击相机图标拍照
                       - 照片会自动分析
                    
                    2. **上传模式**:
                       - 点击"📁上传"标签
                       - 拖拽或点击上传图片
                       - 支持 JPG, PNG, WebP 等格式
                    
                    3. **提问模式**:
                       - 在右侧输入具体问题
                       - 留空则自动描述图片内容
                    
                    ### ⚙️ 当前模型
                    """)
                    
                    gr.Markdown(f"`{settings.VLM_MODEL}`")
                
                # 右侧：识别结果
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 识别结果")
                    
                    # 问题输入框
                    question_input = gr.Textbox(
                        label="向AI提问（可选）",
                        placeholder="例如: 图片中有几个人? 这是什么地方? 图片的主题是什么?",
                        lines=2
                    )
                    
                    # 分析按钮
                    analyze_btn = gr.Button("🔍 分析图片", variant="primary", size="lg")
                    
                    # 结果显示
                    result_output = gr.Textbox(
                        label="分析结果",
                        lines=15,
                        placeholder="分析结果将显示在这里...",
                        show_copy_button=True  # 显示复制按钮
                    )
                    
                    # 示例问题
                    with gr.Accordion("📖 示例问题", open=False):
                        gr.Markdown("""
                        **描述类**:
                        - 请详细描述这张图片
                        - 图片的主要内容是什么？
                        - 这张图片给你什么感觉？
                        
                        **识别类**:
                        - 图片中有什么物体？
                        - 能认出这是什么地方吗？
                        - 图片中有几个人？
                        
                        **分析类**:
                        - 图片的主题是什么？
                        - 这张图片可能在表达什么？
                        - 图片的色调和氛围如何？
                        """)
            
            # ===== 事件绑定 =====
            
            # 定义统一的分析函数
            def analyze_with_source(source_type, camera_img, upload_img, question):
                """
                根据图片来源进行分析
                
                Args:
                    source_type: 图片来源类型（自动判断）
                    camera_img: 摄像头图片
                    upload_img: 上传的图片
                    question: 用户问题
                """
                # 判断使用哪个图片源
                if camera_img is not None:
                    return analyze_image(camera_img, question)
                elif upload_img is not None:
                    return analyze_image(upload_img, question)
                else:
                    return "⚠️ 请先上传图片或使用摄像头拍照"
            
            # 分析按钮点击事件
            analyze_btn.click(
                fn=lambda c, u, q: analyze_with_source("both", c, u, q),
                inputs=[camera_input, upload_input, question_input],
                outputs=[result_output]
            )
            
            # 摄像头拍照后自动分析
            camera_input.change(
                fn=lambda img, q: analyze_image(img, q) if img is not None else "",
                inputs=[camera_input, question_input],
                outputs=[result_output]
            )
            
            # 上传图片后自动分析
            upload_input.change(
                fn=lambda img, q: analyze_image(img, q) if img is not None else "",
                inputs=[upload_input, question_input],
                outputs=[result_output]
            )
        
        # ==================== 底部信息 ====================
        gr.Markdown("""
        ---
        ### 📚 相关资源
        - [硅基流动官网](https://siliconflow.cn)
        - [API文档](https://docs.siliconflow.cn)
        - [获取API Key](https://cloud.siliconflow.cn/account/ak)
        
        ### 🔧 技术栈
        - **文本**: OpenAI兼容API (Chat Completions)
        - **语音**: TTS API (Text-to-Speech)
        - **视觉**: Vision Language Model (VLM)
        - **框架**: Gradio 4.0+ · Python 3.9+
        """)
    
    return demo


# ==================== 主函数 ====================

def main():
    """程序入口"""
    logger.info("=" * 60)
    logger.info("启动硅基流动多模态AI助手Demo")
    logger.info("=" * 60)
    
    try:
        # ===== 1. 验证配置 =====
        settings.validate()
        logger.info("✅ 配置验证通过")
        
        # ===== 2. 创建输出目录 =====
        ensure_output_dirs()
        logger.info(f"✅ 输出目录已创建:")
        logger.info(f"   - 音频: {settings.AUDIO_OUTPUT_DIR}")
        logger.info(f"   - 图片: {settings.IMAGE_OUTPUT_DIR}")
        
        # ===== 3. 显示配置信息 =====
        logger.info(f"📝 文本模型: {settings.TEXT_MODEL}")
        logger.info(f"🔊 语音模型: {settings.TTS_MODEL}")
        logger.info(f"🖼️ 图像模型: {settings.VLM_MODEL}")
        logger.info(f"🔗 API地址: {settings.SILICONFLOW_BASE_URL}")
        
    except ValueError as e:
        logger.error(f"❌ 配置错误: {e}")
        logger.info("💡 请按以下步骤配置:")
        logger.info("   1. 复制 env_template.txt 为 .env")
        logger.info("   2. 编辑 .env 文件，填入你的 API Key")
        logger.info("   3. API Key获取: https://cloud.siliconflow.cn/account/ak")
        return
    
    # ===== 4. 创建并启动Demo =====
    demo = create_demo()
    
    logger.info("🚀 启动 Gradio 服务...")
    logger.info("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7862,        # 端口号
        share=True,              # 创建公网链接（可选）
        show_error=True          # 显示错误信息
    )


if __name__ == "__main__":
    main()

