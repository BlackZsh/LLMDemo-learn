"""
硅基流动 ASR API 本地测试脚本
支持多种音频格式和语言识别
"""
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

# 配置
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
OUTPUT_DIR = "../outputs/audio"

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# 支持的ASR模型列表
SUPPORTED_MODELS = {
    "sensevoice": "FunAudioLLM/SenseVoiceSmall",
    "sensevoice-alt": "iic/SenseVoiceSmall",
}


def test_asr_transcription(
    audio_path: str,
    model: str = "FunAudioLLM/SenseVoiceSmall",
    language: str = "auto",
    response_format: str = "json"
):
    """
    测试语音识别API
    
    Args:
        audio_path: 音频文件路径或URL
        model: ASR模型名称
        language: 指定语言（auto/zh/en/ja/yue等）
        response_format: 响应格式（json/text/verbose_json）
        
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*70}")
    print(f"🎤 语音识别测试")
    print(f"{'='*70}")
    print(f"📝 模型: {model}")
    print(f"🎵 音频: {audio_path}")
    print(f"🌐 语言: {language}")
    print(f"📋 格式: {response_format}")
    print(f"{'='*70}")
    
    # 检查API Key
    if not API_KEY or API_KEY == "your_api_key_here":
        print("❌ 错误: API Key未配置！")
        print("请在 .env 文件中设置 SILICONFLOW_API_KEY")
        print("获取地址: https://cloud.siliconflow.cn/account/ak")
        return False
    
    # 处理音频文件
    if audio_path.startswith(('http://', 'https://')):
        # 网络音频URL
        print(f"📡 使用网络音频: {audio_path[:60]}...")
        try:
            print("⏳ 正在下载音频...")
            response = requests.get(audio_path, timeout=30)
            response.raise_for_status()
            audio_data = response.content
            audio_file = ("audio.mp3", audio_data, "audio/mpeg")
        except Exception as e:
            print(f"❌ 音频下载失败: {str(e)}")
            return False
    else:
        # 本地音频文件
        if not os.path.exists(audio_path):
            print(f"❌ 错误: 音频文件不存在: {audio_path}")
            return False
        
        print(f"📁 使用本地音频: {audio_path}")
        
        # 检测文件格式
        file_ext = Path(audio_path).suffix.lower()
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
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        file_size = len(audio_data) / (1024 * 1024)
        print(f"📦 文件大小: {file_size:.2f} MB")
        
        audio_file = (Path(audio_path).name, audio_data, mime_type)
    
    # 构建请求
    url = f"{BASE_URL}/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    files = {
        'file': audio_file
    }
    
    data = {
        'model': model
    }
    
    # 添加可选参数
    if language and language != "auto":
        data['language'] = language
    
    if response_format:
        data['response_format'] = response_format
    
    print(f"\n📤 发送请求到: {url}")
    print(f"📦 请求参数: {data}")
    
    try:
        # 发送POST请求
        start_time = time.time()
        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  响应时间: {elapsed_time:.2f}秒")
        print(f"📊 状态码: {response.status_code}")
        
        # 检查响应
        if response.status_code == 200:
            # 解析结果
            if response_format == "text":
                text = response.text
                
                print(f"\n{'='*70}")
                print(f"✅ 识别成功!")
                print(f"{'='*70}")
                print(f"\n📝 识别文本:\n")
                print(text)
                print(f"\n{'='*70}")
                print(f"📊 文本长度: {len(text)} 字符")
                print(f"{'='*70}")
                
            elif response_format == "verbose_json":
                result = response.json()
                
                text = result.get('text', '')
                language_detected = result.get('language', 'unknown')
                duration = result.get('duration', 0)
                segments = result.get('segments', [])
                
                print(f"\n{'='*70}")
                print(f"✅ 识别成功!")
                print(f"{'='*70}")
                print(f"\n📝 识别文本:\n")
                print(text)
                print(f"\n{'='*70}")
                print(f"📊 详细信息:")
                print(f"   - 检测语言: {language_detected}")
                print(f"   - 音频时长: {duration:.2f}秒")
                print(f"   - 文本长度: {len(text)} 字符")
                print(f"   - 片段数量: {len(segments)}")
                
                if segments and len(segments) > 0:
                    print(f"\n⏱️  时间戳信息:")
                    for i, seg in enumerate(segments[:5], 1):  # 只显示前5个片段
                        start = seg.get('start', 0)
                        end = seg.get('end', 0)
                        seg_text = seg.get('text', '')
                        print(f"   {i}. [{start:.1f}s - {end:.1f}s] {seg_text}")
                    if len(segments) > 5:
                        print(f"   ... 还有 {len(segments) - 5} 个片段")
                
                print(f"{'='*70}")
                
            else:
                # JSON格式
                result = response.json()
                text = result.get('text', '')
                
                print(f"\n{'='*70}")
                print(f"✅ 识别成功!")
                print(f"{'='*70}")
                print(f"\n📝 识别文本:\n")
                print(text)
                print(f"\n{'='*70}")
                print(f"📊 文本长度: {len(text)} 字符")
                print(f"{'='*70}")
            
            return True
            
        elif response.status_code == 403:
            print(f"\n❌ 403 Forbidden - 权限错误")
            print(f"可能原因:")
            print(f"  1. API Key 没有访问该模型的权限")
            print(f"  2. 账户余额不足")
            print(f"\n建议:")
            print(f"  1. 检查 API Key 权限: https://cloud.siliconflow.cn/account/ak")
            print(f"  2. 查看账户余额: https://cloud.siliconflow.cn/account/billing")
            return False
            
        elif response.status_code == 404:
            print(f"\n❌ 404 Not Found - 模型不存在")
            print(f"模型名称可能不正确: {model}")
            print(f"\n支持的模型:")
            for key, value in SUPPORTED_MODELS.items():
                print(f"  - {value}")
            return False
            
        elif response.status_code == 400:
            print(f"\n❌ 400 Bad Request - 请求参数错误")
            try:
                error_detail = response.json()
                print(f"错误详情: {error_detail}")
            except:
                print(f"错误信息: {response.text}")
            print(f"\n检查项:")
            print(f"  1. 音频文件格式是否支持（mp3/wav/m4a/flac/ogg/webm）")
            print(f"  2. 音频文件是否损坏")
            print(f"  3. 文件大小是否超出限制（建议<25MB）")
            return False
            
        else:
            print(f"\n❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ 请求超时")
        print("建议: 检查网络连接或使用更小的音频文件")
        return False
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        return False


def test_multiple_languages():
    """测试多语言识别"""
    print(f"\n{'='*70}")
    print(f"🌐 多语言识别测试")
    print(f"{'='*70}")
    
    # 测试用例（需要准备对应的音频文件）
    test_cases = [
        {
            "name": "中文测试",
            "audio": "../outputs/audio/chinese_test.mp3",
            "language": "zh"
        },
        {
            "name": "英文测试",
            "audio": "../outputs/audio/english_test.mp3",
            "language": "en"
        },
        {
            "name": "自动检测",
            "audio": "../outputs/audio/mixed_test.mp3",
            "language": "auto"
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n🧪 {test['name']}")
        
        if not os.path.exists(test['audio']):
            print(f"⚠️  跳过: 音频文件不存在 ({test['audio']})")
            results.append(False)
            continue
        
        success = test_asr_transcription(
            audio_path=test['audio'],
            language=test['language']
        )
        results.append(success)
        
        # 间隔
        if test != test_cases[-1]:
            print(f"\n⏳ 等待2秒...")
            time.sleep(2)
    
    # 总结
    print(f"\n{'='*70}")
    print(f"📊 测试总结")
    print(f"{'='*70}")
    print(f"总测试数: {len(results)}")
    print(f"成功: {sum(results)} ✅")
    print(f"失败: {len(results) - sum(results)} ❌")
    print(f"{'='*70}")


def interactive_test():
    """交互式测试"""
    print("\n" + "="*70)
    print("🎤 语音识别交互式测试")
    print("="*70)
    
    # 选择模型
    print("\n可用模型:")
    for i, (key, model) in enumerate(SUPPORTED_MODELS.items(), 1):
        print(f"  {i}. {model}")
    
    model_choice = input("\n选择模型 (1-2, 默认1): ").strip() or "1"
    model_index = int(model_choice) - 1
    model = list(SUPPORTED_MODELS.values())[model_index] if 0 <= model_index < len(SUPPORTED_MODELS) else "FunAudioLLM/SenseVoiceSmall"
    
    # 输入音频
    print("\n音频来源:")
    print("  1. 本地音频文件")
    print("  2. 网络音频URL")
    
    source_type = input("\n选择音频来源 (1-2): ").strip()
    
    if source_type == "2":
        audio_path = input("请输入音频URL: ").strip()
    else:
        audio_path = input("请输入音频文件路径: ").strip()
    
    # 选择语言
    print("\n语言选择:")
    print("  1. auto - 自动检测")
    print("  2. zh - 中文")
    print("  3. en - 英文")
    print("  4. ja - 日语")
    print("  5. yue - 粤语")
    
    lang_choice = input("\n选择语言 (1-5, 默认1): ").strip() or "1"
    languages = ["auto", "zh", "en", "ja", "yue"]
    language = languages[int(lang_choice) - 1] if lang_choice.isdigit() and 1 <= int(lang_choice) <= 5 else "auto"
    
    # 响应格式
    print("\n响应格式:")
    print("  1. json - 标准JSON")
    print("  2. text - 纯文本")
    print("  3. verbose_json - 详细JSON（含时间戳）")
    
    format_choice = input("\n选择格式 (1-3, 默认1): ").strip() or "1"
    formats = ["json", "text", "verbose_json"]
    response_format = formats[int(format_choice) - 1] if format_choice.isdigit() and 1 <= int(format_choice) <= 3 else "json"
    
    # 执行测试
    test_asr_transcription(
        audio_path=audio_path,
        model=model,
        language=language,
        response_format=response_format
    )


def quick_test_with_sample():
    """使用在线样例音频快速测试"""
    print("\n" + "="*70)
    print("🚀 快速测试（使用在线样例音频）")
    print("="*70)
    
    # 使用公开的测试音频URL
    sample_audio_urls = [
        "https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav",  # 英文样例
        # 可以添加更多公开的音频URL
    ]
    
    print("\n正在使用在线样例音频进行测试...")
    
    for i, url in enumerate(sample_audio_urls, 1):
        print(f"\n测试 {i}/{len(sample_audio_urls)}")
        test_asr_transcription(
            audio_path=url,
            language="auto",
            response_format="json"
        )
        
        if i < len(sample_audio_urls):
            time.sleep(2)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🎤 硅基流动语音识别 API 测试工具")
    print("="*70)
    print("\n使用方式:")
    print("  1. python test_asr_local.py                    # 快速测试")
    print("  2. python test_asr_local.py interactive        # 交互式测试")
    print("  3. python test_asr_local.py <audio_path>       # 测试指定音频")
    print("  4. python test_asr_local.py multilang          # 多语言测试")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_test()
        elif sys.argv[1] == "multilang":
            test_multiple_languages()
        else:
            # 快速测试指定音频
            audio_path = sys.argv[1]
            language = sys.argv[2] if len(sys.argv) > 2 else "auto"
            test_asr_transcription(
                audio_path=audio_path,
                language=language
            )
    else:
        # 默认：快速测试
        quick_test_with_sample()

