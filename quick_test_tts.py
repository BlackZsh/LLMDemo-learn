"""
快速测试硅基流动 TTS API
最简单的测试方式
"""
import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置（直接在这里修改或使用.env文件）
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
# 如果.env不存在，可以直接在下面填入你的API Key
# API_KEY = "sk-xxxxxxxxxxxxxxxxx"

def quick_test():
    """快速测试TTS API"""
    
    print("🎤 快速测试硅基流动 TTS API\n")
    
    # 1. 检查API Key
    if not API_KEY or API_KEY == "your_api_key_here":
        print("❌ API Key未配置！")
        print("\n配置方法:")
        print("方法1: 在本文件第14行直接填入 API_KEY = \"你的密钥\"")
        print("方法2: 在 .env 文件中配置 SILICONFLOW_API_KEY")
        print("\n获取API Key: https://cloud.siliconflow.cn/account/ak")
        return
    
    print(f"✅ API Key: {API_KEY[:10]}...\n")
    
    # 2. 构建请求
    url = "https://api.siliconflow.cn/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "fishaudio/fish-speech-1.5",
        "input": "你好，这是语音合成测试。",
        "voice": "alex",
        "speed": 1.0
    }
    
    print(f"📤 请求URL: {url}")
    print(f"📦 模型: {data['model']}")
    print(f"📝 文本: {data['input']}")
    print(f"🔊 发音人: {data['voice']}\n")
    
    # 3. 发送请求
    try:
        print("⏳ 正在发送请求...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"📥 状态码: {response.status_code}\n")
        
        # 4. 处理响应
        if response.status_code == 200:
            # 保存音频
            output_file = "quick_test_output.mp3"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print("✅ 成功！")
            print(f"📁 音频已保存: {output_file}")
            print(f"📊 文件大小: {file_size:,} 字节 ({file_size/1024:.2f} KB)")
            print("\n🎵 可以播放 quick_test_output.mp3 听一下效果！")
            
        else:
            print("❌ 请求失败！")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {response.text}\n")
            
            # 常见错误提示
            if response.status_code == 403:
                print("💡 403错误解决方法:")
                print("  - 检查API Key是否正确")
                print("  - 确认API Key有TTS权限")
                print("  - 检查账户余额是否充足")
            elif response.status_code == 401:
                print("💡 401错误解决方法:")
                print("  - API Key格式错误或已失效")
                print("  - 重新获取API Key")
            elif response.status_code == 400:
                print("💡 400错误解决方法:")
                print("  - 请求参数有误")
                print("  - 检查模型名称是否正确")
    
    except requests.exceptions.Timeout:
        print("❌ 请求超时，请检查网络连接")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    quick_test()

