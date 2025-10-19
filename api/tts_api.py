"""
语音合成API - 调用硅基流动TTS (Text-to-Speech) 模型
将文本转换为语音
"""
import requests
import base64
from typing import Optional
from loguru import logger
from config.settings import settings


class TTSAPI:
    """
    语音合成API封装类
    
    功能：
    - 将文本转换为语音
    - 支持多种发音人
    - 返回音频数据或保存为文件
    """

    def __init__(self):
        """
        初始化TTS API
        
        从配置文件读取:
        - API基础URL
        - TTS模型名称
        - 请求头（包含API Key）
        """
        self.base_url = settings.SILICONFLOW_BASE_URL
        self.model = settings.TTS_MODEL
        self.headers = settings.get_headers()

    def synthesize(
            self,
            text: str,
            voice: Optional[str] = None,
            speed: float = 1.0,
            save_path: Optional[str] = None
    ) -> Optional[bytes]:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本内容（不能为空）
            voice: 发音人ID（可选，使用默认发音人）
            speed: 语速倍率，范围0.5-2.0，默认1.0
            save_path: 保存音频文件的路径（可选，不传则只返回音频数据）
            
        Returns:
            bytes: 音频数据（WAV或MP3格式）
            如果保存文件成功，返回None
            
        Raises:
            Exception: API调用失败时抛出异常
            
        Example:
            >>> tts_api = TTSAPI()
            >>> # 直接获取音频数据
            >>> audio_data = tts_api.synthesize("你好，欢迎使用")
            >>> # 或保存为文件
            >>> tts_api.synthesize("你好", save_path="output.wav")
        """
        try:
            # 参数验证
            if not text or not text.strip():
                raise ValueError("文本内容不能为空")

            # 构建请求URL
            url = f"{self.base_url}/audio/speech"

            # 构建请求体
            payload = {
                "model": self.model,
                "input": text,
                "voice": voice or "fnlp/MOSS-TTSD-v0.5:alex",  # 默认发音人
                "speed": max(0.5, min(2.0, speed))  # 限制速度范围
            }

            logger.info(f"调用TTS API: {self.model}, 文本长度: {len(text)}")

            # 发送POST请求
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=60
            )

            # 检查响应状态
            response.raise_for_status()

            # 获取音频数据
            audio_data = response.content

            logger.info(f"TTS API响应成功，音频大小: {len(audio_data)} 字节")

            # 如果指定了保存路径，则保存文件
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"音频已保存到: {save_path}")
                return None

            # 否则返回音频数据
            return audio_data

        except requests.exceptions.Timeout:
            error_msg = "TTS API请求超时"
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"TTS API调用失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"TTS处理失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def synthesize_to_base64(self, text: str, voice: Optional[str] = None) -> str:
        """
        将文本转换为语音，并返回Base64编码的音频数据
        
        这个方法适用于Web前端直接播放音频
        
        Args:
            text: 要转换的文本
            voice: 发音人ID（可选）
            
        Returns:
            str: Base64编码的音频数据
            
        Example:
            >>> tts_api = TTSAPI()
            >>> base64_audio = tts_api.synthesize_to_base64("你好")
            >>> # 在HTML中使用: <audio src="data:audio/wav;base64,{base64_audio}">
        """
        try:
            # 获取音频数据
            audio_data = self.synthesize(text=text, voice=voice)

            # 转换为Base64
            if audio_data:
                base64_audio = base64.b64encode(audio_data).decode('utf-8')
                logger.info(f"音频已转换为Base64，长度: {len(base64_audio)}")
                return base64_audio
            else:
                raise Exception("未能获取音频数据")

        except Exception as e:
            logger.error(f"Base64转换失败: {str(e)}")
            raise


# 创建全局实例，方便其他模块直接导入使用
tts_api = TTSAPI()

# 示例代码（仅用于测试）
if __name__ == "__main__":
    """
    测试TTS API功能
    运行方式: python -m api.tts_api
    """
    try:
        # 测试语音合成
        test_text = "你好，这是硅基流动语音合成测试。"

        print("正在测试TTS API...")
        audio_data = tts_api.synthesize(test_text, voice="fnlp/MOSS-TTSD-v0.5:alex", save_path="test_tts.wav")

        print("✅ TTS测试成功！音频已保存为 test_tts.wav")

    except Exception as e:
        print(f"❌ TTS测试失败: {e}")
