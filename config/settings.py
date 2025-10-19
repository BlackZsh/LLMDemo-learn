"""
硅基流动API配置文件
"""
import os
from typing import Dict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings:
    """配置类"""
    
    # 硅基流动API配置
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
    
    # 文本模型配置
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.7"))
    
    # 语音识别模型配置
    SPEECH_MODEL: str = os.getenv("SPEECH_MODEL", "FunAudioLLM/SenseVoiceSmall")
    
    # 语音合成模型配置
    TTS_MODEL: str = os.getenv("TTS_MODEL", "fishaudio/fish-speech-1.5")
    
    # 图像生成模型配置
    IMAGE_MODEL: str = os.getenv("IMAGE_MODEL", "stabilityai/stable-diffusion-3-5-large")
    
    # 图像识别模型配置 (Vision Language Model)
    # 根据官方文档推荐：https://docs.siliconflow.cn/cn/userguide/capabilities/multimodal-vision
    VLM_MODEL: str = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    
    # 音频文件保存路径
    AUDIO_OUTPUT_DIR: str = os.getenv("AUDIO_OUTPUT_DIR", "outputs/audio")
    IMAGE_OUTPUT_DIR: str = os.getenv("IMAGE_OUTPUT_DIR", "outputs/images")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    def get_headers(self) -> Dict[str, str]:
        """获取API请求头"""
        return {
            "Authorization": f"Bearer {self.SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def validate(self) -> bool:
        """验证配置是否完整"""
        if not self.SILICONFLOW_API_KEY:
            raise ValueError("SILICONFLOW_API_KEY 未设置！请在 .env 文件中配置")
        return True


# 创建全局配置实例
settings = Settings()

