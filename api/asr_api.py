"""
语音识别API - 调用硅基流动ASR (Automatic Speech Recognition) 模型
将语音转换为文字

官方文档：https://docs.siliconflow.cn/cn/api-reference/audio-transcriptions

支持的模型：
- FunAudioLLM/SenseVoiceSmall（推荐，支持多语言）
- iic/SenseVoiceSmall（备选）
"""
import requests
import base64
from typing import Optional, Union
from pathlib import Path
from loguru import logger
from config.settings import settings


class ASRAPI:
    """
    语音识别API封装类
    
    功能：
    - 将语音转换为文字
    - 支持多种音频格式（mp3, wav, m4a, flac等）
    - 支持本地音频文件或URL
    - 支持麦克风录音识别
    
    支持的语言：
    - 中文（zh）
    - 英文（en）
    - 日语（ja）
    - 韩语（ko）
    - 粤语（yue）
    - 等多种语言
    """
    
    def __init__(self):
        """
        初始化ASR API
        
        从配置文件读取:
        - API基础URL
        - ASR模型名称
        - 请求头（包含API Key）
        """
        self.base_url = settings.SILICONFLOW_BASE_URL
        self.model = settings.SPEECH_MODEL
        self.headers = {
            "Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}"
        }
        
        logger.info(f"ASR API 初始化: {self.model}")
    
    def encode_audio_to_base64(self, audio_path: str) -> str:
        """
        将本地音频文件编码为Base64
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            str: Base64编码的音频数据
        """
        try:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            logger.info(f"音频已编码，大小: {len(audio_data)} 字节")
            return base64_audio
            
        except Exception as e:
            logger.error(f"音频编码失败: {str(e)}")
            raise
    
    def transcribe(
        self,
        audio_source: Union[str, bytes],
        language: Optional[str] = "auto",
        response_format: str = "json"
    ) -> str:
        """
        语音识别（语音转文字）
        
        Args:
            audio_source: 音频来源，可以是：
                         - 本地文件路径（str）如 "audio.mp3"
                         - 音频URL（str）如 "https://example.com/audio.mp3"
                         - 音频二进制数据（bytes）
            language: 指定语言代码（可选）
                     - "auto": 自动检测（默认）
                     - "zh": 中文
                     - "en": 英文
                     - "ja": 日语
                     - "yue": 粤语
            response_format: 响应格式
                            - "json": JSON格式（默认）
                            - "text": 纯文本格式
        
        Returns:
            str: 识别出的文字内容
            
        Raises:
            Exception: API调用失败时抛出异常
            
        Example:
            >>> asr_api = ASRAPI()
            >>> # 识别本地音频
            >>> text = asr_api.transcribe("recording.mp3")
            >>> # 识别网络音频
            >>> text = asr_api.transcribe("https://example.com/audio.mp3")
            >>> # 指定语言
            >>> text = asr_api.transcribe("audio.mp3", language="zh")
        """
        try:
            # 1. 处理音频数据
            if isinstance(audio_source, bytes):
                # 二进制数据，直接编码
                logger.info("使用二进制音频数据")
                audio_file = ("audio.wav", audio_source, "audio/wav")
                
            elif isinstance(audio_source, str) and audio_source.startswith(('http://', 'https://')):
                # 网络音频URL
                logger.info(f"使用音频URL: {audio_source[:50]}...")
                # 先下载音频
                response = requests.get(audio_source, timeout=30)
                response.raise_for_status()
                audio_file = ("audio.mp3", response.content, "audio/mpeg")
                
            elif isinstance(audio_source, str) and Path(audio_source).exists():
                # 本地文件路径
                logger.info(f"使用本地音频: {audio_source}")
                
                # 检测文件格式
                file_ext = Path(audio_source).suffix.lower()
                mime_types = {
                    '.mp3': 'audio/mpeg',
                    '.wav': 'audio/wav',
                    '.m4a': 'audio/m4a',
                    '.flac': 'audio/flac',
                    '.ogg': 'audio/ogg',
                    '.webm': 'audio/webm'
                }
                mime_type = mime_types.get(file_ext, 'audio/mpeg')
                
                # 读取文件
                with open(audio_source, 'rb') as f:
                    audio_data = f.read()
                
                audio_file = (Path(audio_source).name, audio_data, mime_type)
                
            else:
                raise ValueError(f"无效的音频来源: {audio_source}")
            
            # 2. 构建请求
            url = f"{self.base_url}/audio/transcriptions"
            
            # 构建multipart/form-data请求
            files = {
                'file': audio_file
            }
            
            data = {
                'model': self.model,
            }
            
            # 添加可选参数
            if language and language != "auto":
                data['language'] = language
            
            if response_format:
                data['response_format'] = response_format
            
            logger.info(f"调用ASR API: {self.model}")
            logger.info(f"语言设置: {language}")
            
            # 3. 发送请求
            response = requests.post(
                url,
                headers=self.headers,
                files=files,
                data=data,
                timeout=60
            )
            
            # 4. 检查响应
            response.raise_for_status()
            
            # 5. 解析结果
            if response_format == "text":
                text = response.text
            else:
                result = response.json()
                text = result.get('text', '')
            
            logger.info(f"ASR识别成功，文本长度: {len(text)}")
            return text
            
        except requests.exceptions.Timeout:
            error_msg = "ASR API请求超时"
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"ASR API调用失败: {str(e)}"
            logger.error(error_msg)
            # 尝试获取详细错误信息
            try:
                error_detail = e.response.json() if hasattr(e, 'response') else str(e)
                logger.error(f"详细错误: {error_detail}")
            except:
                pass
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"ASR处理失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def transcribe_with_timestamps(
        self,
        audio_source: Union[str, bytes],
        language: Optional[str] = "auto"
    ) -> dict:
        """
        语音识别（带时间戳）
        
        Args:
            audio_source: 音频来源
            language: 语言代码
            
        Returns:
            dict: 包含文本和时间戳信息
                {
                    "text": "完整文本",
                    "segments": [
                        {"start": 0.0, "end": 2.5, "text": "第一句话"},
                        {"start": 2.5, "end": 5.0, "text": "第二句话"}
                    ]
                }
        """
        try:
            # 使用 verbose_json 格式获取详细信息
            url = f"{self.base_url}/audio/transcriptions"
            
            # 处理音频文件
            if isinstance(audio_source, str) and Path(audio_source).exists():
                with open(audio_source, 'rb') as f:
                    audio_data = f.read()
                audio_file = (Path(audio_source).name, audio_data, 'audio/mpeg')
            else:
                raise ValueError("时间戳功能需要本地音频文件")
            
            files = {'file': audio_file}
            data = {
                'model': self.model,
                'response_format': 'verbose_json'  # 获取详细信息
            }
            
            if language and language != "auto":
                data['language'] = language
            
            response = requests.post(
                url,
                headers=self.headers,
                files=files,
                data=data,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info("ASR识别成功（含时间戳）")
            return result
            
        except Exception as e:
            logger.error(f"带时间戳的ASR识别失败: {str(e)}")
            raise


# 创建全局实例，方便其他模块直接导入使用
asr_api = ASRAPI()


# 示例代码（仅用于测试）
if __name__ == "__main__":
    """
    测试ASR API功能
    运行方式: python -m api.asr_api
    """
    try:
        # 测试语音识别
        test_audio = "test_audio.mp3"
        
        print("正在测试ASR API...")
        text = asr_api.transcribe(test_audio, language="zh")
        
        print(f"✅ ASR测试成功！")
        print(f"识别文本: {text}")
        
    except Exception as e:
        print(f"❌ ASR测试失败: {e}")
        print("提示: 请确保test_audio.mp3存在，或使用音频URL测试")

