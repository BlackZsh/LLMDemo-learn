"""
硅基流动 TTS API 本地测试脚本
支持多种模型和参数配置
"""
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
OUTPUT_DIR = "../outputs/audio"

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def test_tts_synthesis(
    text: str,
    model: str = "fishaudio/fish-speech-1.5",
    voice: str = "fnlp/MOSS-TTSD-v0.5:alex", # 带上模型名字写，最好去找枚举，否则一直报400.
    response_format: str = "mp3",
    speed: float = 1.0,
    output_filename: str = None
):
    """
    测试TTS语音合成
    
    Args:
        text: 要合成的文本
        model: TTS模型名称
        voice: 发音人（fish-speech模型使用）
        response_format: 音频格式 (mp3/wav/opus/pcm)
        speed: 语速 (0.25-4.0)
        output_filename: 输出文件名（可选）
    
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*60}")
    print(f"🎤 测试模型: {model}")
    print(f"📝 输入文本: {text}")
    print(f"🔊 发音人: {voice}")
    print(f"⚡ 语速: {speed}x")
    print(f"{'='*60}")
    
    # 检查API Key
    if not API_KEY or API_KEY == "your_api_key_here":
        print("❌ 错误: API Key未配置！")
        print("请在 .env 文件中设置 SILICONFLOW_API_KEY")
        print("获取地址: https://cloud.siliconflow.cn/account/ak")
        return False
    
    # 构建请求
    url = f"{BASE_URL}/audio/speech"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 根据不同模型构建请求数据
    if "fish-speech" in model:
        # FishAudio 模型参数
        data = {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed
        }
    elif "MOSS" in model:
        # MOSS 模型参数（支持多说话人对话）
        data = {
            "model": model,
            "input": text,
            "response_format": response_format,
            "sample_rate": 32000,
            "speed": speed,
            "gain": 0
        }
    else:
        # 通用参数
        data = {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed
        }
    
    print(f"\n📤 发送请求到: {url}")
    print(f"📦 请求参数: {data}")
    
    try:
        # 发送POST请求
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=60
        )
        
        print(f"📥 响应状态码: {response.status_code}")
        
        # 检查响应
        if response.status_code == 200:
            # 生成输出文件名
            if not output_filename:
                model_name = model.split('/')[-1]
                output_filename = f"tts_{model_name}_{voice}_{response_format}"
            
            output_path = os.path.join(OUTPUT_DIR, f"{output_filename}.{response_format}")
            
            # 保存音频文件
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print(f"✅ 成功！音频已保存")
            print(f"📁 文件路径: {output_path}")
            print(f"📊 文件大小: {file_size:,} 字节 ({file_size/1024:.2f} KB)")
            return True
            
        else:
            print(f"❌ 请求失败！")
            print(f"状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
            # 提供具体的错误提示
            if response.status_code == 403:
                print("\n💡 提示:")
                print("  - 403错误通常表示API Key无效或没有权限")
                print("  - 请检查 .env 文件中的 SILICONFLOW_API_KEY")
                print("  - 确保API Key有TTS服务的访问权限")
            elif response.status_code == 401:
                print("\n💡 提示:")
                print("  - 401错误表示认证失败")
                print("  - 请确认API Key格式正确")
            elif response.status_code == 400:
                print("\n💡 提示:")
                print("  - 400错误表示请求参数有误")
                print("  - 请检查模型名称、文本内容等参数")
            
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时！请检查网络连接")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请检查网络连接")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        return False


def test_moss_dialogue():
    """测试 MOSS 模型的多人对话功能"""
    text = "[S1]你好，今天过得怎么样？[S2]我很好，谢谢你的关心！[S1]听到你很好，我很高兴。"
    return test_tts_synthesis(
        text=text,
        model="fnlp/MOSS-TTSD-v0.5",
        voice="",  # MOSS不使用voice参数
        response_format="mp3",
        speed=1.0,
        output_filename="tts_moss_dialogue"
    )


def test_fish_speech():
    """测试 Fish Speech 模型"""
    text = "你好，这是硅基流动语音合成测试。欢迎使用 Fish Speech 模型。"
    return test_tts_synthesis(
        text=text,
        model="fishaudio/fish-speech-1.5",
        voice="fnlp/MOSS-TTSD-v0.5:alex",
        response_format="mp3",
        speed=1.0,
        output_filename="tts_fish_speech"
    )


def test_different_speeds():
    """测试不同语速"""
    text = "这是语速测试。"
    speeds = [0.5, 1.0, 1.5, 2.0]
    
    print(f"\n{'='*60}")
    print("🎵 测试不同语速")
    print(f"{'='*60}")
    
    results = []
    for speed in speeds:
        result = test_tts_synthesis(
            text=text,
            model="fishaudio/fish-speech-1.5",
            voice="alex",
            speed=speed,
            output_filename=f"tts_speed_{speed}"
        )
        results.append((speed, result))
    
    print(f"\n{'='*60}")
    print("📊 语速测试结果汇总:")
    for speed, result in results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"  速度 {speed}x: {status}")
    print(f"{'='*60}")


def main():
    """主测试函数"""
    print("="*60)
    print("🚀 硅基流动 TTS API 本地测试")
    print("="*60)
    
    if not API_KEY or API_KEY == "your_api_key_here":
        print("\n❌ 请先配置API Key!")
        print("\n步骤:")
        print("1. 复制 env_template.txt 为 .env")
        print("2. 在 .env 文件中设置你的 SILICONFLOW_API_KEY")
        print("3. 获取API Key: https://cloud.siliconflow.cn/account/ak")
        return
    
    print(f"\n✅ API Key已配置: {API_KEY[:10]}...")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    
    # 测试菜单
    print("\n" + "="*60)
    print("请选择测试项目:")
    print("1. 测试 Fish Speech 模型（基础测试）")
    print("2. 测试 MOSS 模型（多人对话）")
    print("3. 测试不同语速")
    print("4. 运行所有测试")
    print("5. 自定义测试")
    print("="*60)
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    if choice == "1":
        test_fish_speech()
    elif choice == "2":
        test_moss_dialogue()
    elif choice == "3":
        test_different_speeds()
    elif choice == "4":
        print("\n🔄 运行所有测试...")
        test_fish_speech()
        test_moss_dialogue()
        test_different_speeds()
    elif choice == "5":
        # 自定义测试
        print("\n自定义测试:")
        text = input("输入要合成的文本: ").strip()
        model = input("模型 [fishaudio/fish-speech-1.5]: ").strip() or "fishaudio/fish-speech-1.5"
        voice = input("发音人 [alex]: ").strip() or "fnlp/MOSS-TTSD-v0.5:alex"
        speed = float(input("语速 [1.0]: ").strip() or "1.0")
        
        test_tts_synthesis(
            text=text,
            model=model,
            voice=voice,
            speed=speed
        )
    else:
        print("❌ 无效选项")
    
    print("\n" + "="*60)
    print("✨ 测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()

