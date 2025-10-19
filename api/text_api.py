"""
文本对话API - 调用硅基流动LLM模型
"""
import requests
from typing import Optional, Dict, Any, Generator
from loguru import logger
from config.settings import settings


class TextAPI:
    """文本对话API封装"""
    
    def __init__(self):
        self.base_url = settings.SILICONFLOW_BASE_URL
        self.model = settings.TEXT_MODEL
        self.headers = settings.get_headers()
    
    def chat(
        self, 
        message: str, 
        history: Optional[list] = None,
        stream: bool = False
    ) -> str:
        """
        发送聊天消息
        
        Args:
            message: 用户消息
            history: 历史对话记录 [(user_msg, assistant_msg), ...]
            stream: 是否流式输出
            
        Returns:
            模型回复
        """
        try:
            # 构建消息列表
            messages = []
            if history:
                for user_msg, assistant_msg in history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            messages.append({"role": "user", "content": message})
            
            # 调用API
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.MAX_TOKENS,
                "temperature": settings.TEMPERATURE,
                "stream": stream
            }
            
            logger.info(f"调用文本API: {self.model}")
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            
            logger.info(f"文本API响应成功，长度: {len(reply)}")
            return reply
            
        except Exception as e:
            logger.error(f"文本API调用失败: {str(e)}")
            return f"❌ 文本API调用失败: {str(e)}"
    
    def chat_stream(
        self, 
        message: str, 
        history: Optional[list] = None
    ) -> Generator[str, None, None]:
        """
        流式聊天
        
        Args:
            message: 用户消息
            history: 历史对话记录
            
        Yields:
            流式输出的文本片段
        """
        try:
            # 构建消息列表
            messages = []
            if history:
                for user_msg, assistant_msg in history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            messages.append({"role": "user", "content": message})
            
            # 调用API
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.MAX_TOKENS,
                "temperature": settings.TEMPERATURE,
                "stream": True
            }
            
            logger.info(f"调用流式文本API: {self.model}")
            response = requests.post(
                url, 
                json=payload, 
                headers=self.headers, 
                timeout=60, 
                stream=True
            )
            response.raise_for_status()
            
            # 流式读取响应
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = line_str[6:]
                        if data == '[DONE]':
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
            
            logger.info("流式文本API响应完成")
            
        except Exception as e:
            logger.error(f"流式文本API调用失败: {str(e)}")
            yield f"❌ 流式文本API调用失败: {str(e)}"


# 创建全局实例
text_api = TextAPI()

