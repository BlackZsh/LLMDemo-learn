"""
图像识别API - 调用硅基流动VLM (Vision Language Model) 模型
对图像进行理解和描述

官方文档：https://docs.siliconflow.cn/cn/userguide/capabilities/multimodal-vision

支持的模型系列：
- Qwen2-VL 系列：仅视觉理解（推荐）
- Qwen2.5-VL 系列：高性能视觉理解
- GLM 系列：仅视觉理解
- DeepSeek-VL2 系列：仅视觉理解
- Qwen3-VL 系列：视觉+视频理解
- Qwen3-Omni 系列：全模态（视觉+音频+视频）
"""
import requests
import base64
from typing import Optional, Union
from pathlib import Path
from loguru import logger
from config.settings import settings


class VisionAPI:
    """
    图像识别API封装类
    
    功能：
    - 图像内容描述
    - 图像问答（VQA）
    - 支持本地图片文件或图片URL
    - 支持摄像头拍照
    - 支持多图对比
    
    支持的模型：
    - Qwen/Qwen2-VL-7B-Instruct（默认推荐）
    - Qwen/Qwen2.5-VL-72B-Instruct（高性能版本）
    - THUDM/glm-4v-plus（GLM视觉模型）
    - deepseek-ai/DeepSeek-VL2（DeepSeek视觉模型）
    """
    
    def __init__(self):
        """
        初始化Vision API
        
        从配置文件读取:
        - API基础URL
        - VLM模型名称
        - 请求头（包含API Key）
        """
        self.base_url = settings.SILICONFLOW_BASE_URL
        self.model = settings.VLM_MODEL
        self.headers = settings.get_headers()
        
        logger.info(f"Vision API 初始化: {self.model}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将本地图片文件编码为Base64字符串
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            str: Base64编码的图片数据
            
        Raises:
            FileNotFoundError: 文件不存在
            Exception: 编码失败
        """
        try:
            # 检查文件是否存在
            if not Path(image_path).exists():
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            # 读取图片文件
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # 转换为Base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            logger.info(f"图片已编码，大小: {len(image_data)} 字节")
            return base64_image
            
        except Exception as e:
            logger.error(f"图片编码失败: {str(e)}")
            raise
    
    def describe_image(
        self, 
        image_source: Union[str, bytes],
        prompt: Optional[str] = None,
        detail: str = "auto"
    ) -> str:
        """
        描述图片内容或回答关于图片的问题
        
        Args:
            image_source: 图片来源，可以是：
                         - 本地文件路径（str）如 "photo.jpg"
                         - 图片URL（str）如 "https://example.com/image.jpg"
                         - 图片二进制数据（bytes）
            prompt: 对图片的提问（可选）
                   - 如果不传，默认描述图片内容
                   - 可以问具体问题，如"图片中有几个人？"
            detail: 图片分析详细程度
                   - "auto": 自动选择（默认）
                   - "low": 快速分析
                   - "high": 详细分析
            
        Returns:
            str: 图片描述或问题回答
            
        Example:
            >>> vision_api = VisionAPI()
            >>> # 描述本地图片
            >>> desc = vision_api.describe_image("photo.jpg")
            >>> # 询问图片内容
            >>> answer = vision_api.describe_image("photo.jpg", "图片中有什么动物？")
            >>> # 使用URL
            >>> desc = vision_api.describe_image("https://example.com/cat.jpg")
        """
        try:
            # 1. 处理图片数据
            if isinstance(image_source, bytes):
                # 如果是二进制数据，直接编码
                image_data = base64.b64encode(image_source).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{image_data}"
                logger.info("使用二进制图片数据")
                
            elif image_source.startswith(('http://', 'https://')):
                # 如果是URL，直接使用
                image_url = image_source
                logger.info(f"使用图片URL: {image_url[:50]}...")
                
            else:
                # 如果是本地文件路径，先编码
                image_data = self.encode_image_to_base64(image_source)
                image_url = f"data:image/jpeg;base64,{image_data}"
                logger.info(f"使用本地图片: {image_source}")
            
            # 2. 构建提示词
            if not prompt:
                # 默认提示词：描述图片内容
                prompt = "请详细描述这张图片的内容，包括主要物体、场景、颜色、氛围等信息。"
            
            # 3. 构建消息
            # 使用OpenAI Vision API兼容格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": detail
                            }
                        }
                    ]
                }
            ]
            
            # 4. 构建请求
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.MAX_TOKENS,
                "temperature": 0.7
            }
            
            logger.info(f"调用Vision API: {self.model}")
            
            # 5. 发送请求
            response = requests.post(
                url, 
                json=payload, 
                headers=self.headers, 
                timeout=60
            )
            
            # 6. 检查响应
            response.raise_for_status()
            
            # 7. 解析结果
            result = response.json()
            description = result["choices"][0]["message"]["content"]
            
            logger.info(f"Vision API响应成功，描述长度: {len(description)}")
            return description
            
        except FileNotFoundError as e:
            error_msg = f"文件未找到: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
        except requests.exceptions.Timeout:
            error_msg = "Vision API请求超时"
            logger.error(error_msg)
            return f"❌ {error_msg}"
        except requests.exceptions.RequestException as e:
            error_msg = f"Vision API调用失败: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
        except Exception as e:
            error_msg = f"图片分析失败: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def analyze_image_detailed(self, image_source: Union[str, bytes]) -> dict:
        """
        详细分析图片，返回结构化信息
        
        Args:
            image_source: 图片来源（文件路径、URL或二进制数据）
            
        Returns:
            dict: 包含多个维度的分析结果
                {
                    "description": "整体描述",
                    "objects": "物体列表",
                    "scene": "场景类型",
                    "colors": "主要颜色"
                }
        """
        try:
            # 分别提问不同维度
            description = self.describe_image(image_source, "请整体描述这张图片")
            objects = self.describe_image(image_source, "列出图片中的主要物体")
            scene = self.describe_image(image_source, "这是什么类型的场景？")
            colors = self.describe_image(image_source, "图片的主要颜色是什么？")
            
            return {
                "description": description,
                "objects": objects,
                "scene": scene,
                "colors": colors
            }
            
        except Exception as e:
            logger.error(f"详细分析失败: {str(e)}")
            return {
                "error": str(e)
            }


# 创建全局实例，方便其他模块直接导入使用
vision_api = VisionAPI()


# 示例代码（仅用于测试）
if __name__ == "__main__":
    """
    测试Vision API功能
    运行方式: python -m api.vision_api
    """
    try:
        # 测试图片描述（需要准备一张测试图片）
        test_image = "test_image.jpg"
        
        print("正在测试Vision API...")
        description = vision_api.describe_image(test_image)
        
        print(f"✅ Vision测试成功！")
        print(f"图片描述: {description}")
        
    except Exception as e:
        print(f"❌ Vision测试失败: {e}")
        print("提示: 请确保test_image.jpg存在，或使用图片URL测试")

