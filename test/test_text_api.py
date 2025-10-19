"""
文本API测试脚本 - 快速验证硅基流动API调用
"""
import requests
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def test_text_api():
    """测试文本对话API"""
    
    # 配置
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        print("❌ 错误: 未找到 SILICONFLOW_API_KEY")
        print("💡 请创建 .env 文件并配置: SILICONFLOW_API_KEY=你的密钥")
        print("🔗 获取地址: https://cloud.siliconflow.cn/account/ak")
        return
    
    base_url = "https://api.siliconflow.cn/v1"
    model = "Qwen/Qwen2.5-7B-Instruct"
    
    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 测试问题
    test_messages = [
        "你好，请介绍一下你自己",
        "用Python写一个快速排序算法",
        "解释一下什么是深度学习"
    ]
    
    print("=" * 80)
    print("🚀 硅基流动文本API测试")
    print("=" * 80)
    print(f"📝 模型: {model}")
    print(f"🔗 API地址: {base_url}")
    print("=" * 80)
    
    # 测试每个问题
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*80}")
        print(f"测试 {i}/{len(test_messages)}")
        print(f"{'='*80}")
        print(f"💬 用户: {message}")
        print("-" * 80)
        
        try:
            # 构建请求
            url = f"{base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message}
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.7,
                "stream": False
            }
            
            # 发送请求
            print("⏳ 正在调用API...")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                
                # 提取回复
                reply = result["choices"][0]["message"]["content"]
                
                # 显示结果
                print(f"🤖 助手: {reply}")
                
                # 显示token使用情况
                if "usage" in result:
                    usage = result["usage"]
                    print("-" * 80)
                    print(f"📊 Token使用: ")
                    print(f"   - 输入: {usage.get('prompt_tokens', 0)}")
                    print(f"   - 输出: {usage.get('completion_tokens', 0)}")
                    print(f"   - 总计: {usage.get('total_tokens', 0)}")
                
                print("✅ 测试成功!")
                
            else:
                print(f"❌ API调用失败!")
                print(f"状态码: {response.status_code}")
                print(f"响应: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时!")
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络错误: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
        except Exception as e:
            print(f"❌ 未知错误: {e}")
    
    print(f"\n{'='*80}")
    print("🎉 测试完成!")
    print("=" * 80)


def test_stream_api():
    """测试流式输出API"""
    
    # 配置
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        print("❌ 错误: 未找到 SILICONFLOW_API_KEY")
        return
    
    base_url = "https://api.siliconflow.cn/v1"
    model = "Qwen/Qwen2.5-7B-Instruct"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("\n" + "=" * 80)
    print("⚡ 流式输出测试")
    print("=" * 80)
    
    message = "请写一首关于人工智能的五言绝句"
    print(f"💬 用户: {message}")
    print("-" * 80)
    print("🤖 助手: ", end="", flush=True)
    
    try:
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": True
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60, stream=True)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = line_str[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
            
            print("\n" + "-" * 80)
            print("✅ 流式测试成功!")
        else:
            print(f"\n❌ API调用失败: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    
    print("=" * 80)


if __name__ == "__main__":
    # 运行普通测试
    test_text_api()
    
    # 运行流式测试
    test_stream_api()

