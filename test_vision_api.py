"""
测试图像识别 API 并诊断 403 错误
"""
import requests
from config.settings import settings
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


def test_available_vlm_models():
    """
    测试可用的 VLM 模型列表
    """
    print("=" * 70)
    print("🔍 硅基流动 - 图像识别模型测试")
    print("=" * 70)
    
    # 硅基流动支持的常见 VLM 模型（按推荐顺序）
    vlm_models = [
        # 免费/低成本模型
        "Pro/Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "OpenGVLab/InternVL2-8B",
        
        # 高级模型（可能需要付费）
        "OpenGVLab/InternVL2-26B",
        "OpenGVLab/InternVL2-40B",
        "Pro/OpenGVLab/InternVL2-Pro",
    ]
    
    # 构建简单的测试消息（不需要真实图片，只测试权限）
    test_message = {
        "model": "",  # 会动态填充
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "描述这张图片"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://picsum.photos/200"  # 测试图片
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }
    
    url = f"{settings.SILICONFLOW_BASE_URL}/chat/completions"
    headers = settings.get_headers()
    
    available_models = []
    
    print(f"\n📝 当前 API Key: {settings.SILICONFLOW_API_KEY[:20]}...{settings.SILICONFLOW_API_KEY[-5:]}")
    print(f"🔗 API 地址: {url}")
    print("\n" + "=" * 70)
    print("开始测试各个模型的可用性...")
    print("=" * 70 + "\n")
    
    for model_name in vlm_models:
        test_message["model"] = model_name
        
        try:
            print(f"🧪 测试模型: {model_name}")
            response = requests.post(
                url,
                json=test_message,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   ✅ 可用！")
                available_models.append(model_name)
            elif response.status_code == 403:
                print(f"   ❌ 403 - 无权限访问")
            elif response.status_code == 404:
                print(f"   ⚠️  404 - 模型不存在")
            elif response.status_code == 401:
                print(f"   ❌ 401 - API Key 无效")
            else:
                print(f"   ⚠️  {response.status_code} - {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️  超时（可能可用，但响应慢）")
        except Exception as e:
            print(f"   ❌ 错误: {str(e)[:50]}")
        
        print()
    
    print("=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    
    if available_models:
        print(f"\n✅ 找到 {len(available_models)} 个可用模型：\n")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model}")
        
        print("\n" + "=" * 70)
        print("💡 建议配置")
        print("=" * 70)
        print(f"\n在 .env 文件中修改：")
        print(f"\nVLM_MODEL={available_models[0]}")
        print("\n或在代码中使用：")
        print(f"\nsettings.VLM_MODEL = \"{available_models[0]}\"")
        
    else:
        print("\n❌ 未找到可用的 VLM 模型！")
        print("\n可能的原因：")
        print("   1. API Key 没有开通图像识别权限")
        print("   2. 账户余额不足")
        print("   3. 需要在硅基流动控制台申请 VLM 权限")
        print("\n🔧 解决方法：")
        print("   1. 访问：https://cloud.siliconflow.cn/account/ak")
        print("   2. 检查 API Key 的权限范围")
        print("   3. 查看账户余额：https://cloud.siliconflow.cn/account/billing")
        print("   4. 联系客服开通 VLM 模型权限")
    
    print("\n" + "=" * 70)
    return available_models


def test_simple_text_api():
    """
    测试文本 API 是否正常（对比测试）
    """
    print("\n🧪 对比测试：文本 API（确认 API Key 基本功能）")
    print("=" * 70)
    
    try:
        url = f"{settings.SILICONFLOW_BASE_URL}/chat/completions"
        headers = settings.get_headers()
        
        payload = {
            "model": settings.TEXT_MODEL,
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 50
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print("✅ 文本 API 正常工作 - API Key 有效")
            print(f"   模型: {settings.TEXT_MODEL}")
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            print(f"   回复: {reply[:50]}...")
        else:
            print(f"❌ 文本 API 也失败了: {response.status_code}")
            print(f"   这可能是 API Key 的整体问题")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        # 验证配置
        settings.validate()
        
        # 测试文本 API（对比）
        test_simple_text_api()
        
        # 测试 VLM 模型
        available_models = test_available_vlm_models()
        
        print("\n" + "=" * 70)
        print("✨ 测试完成！")
        print("=" * 70)
        
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("\n请确保 .env 文件存在并配置了 SILICONFLOW_API_KEY")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

