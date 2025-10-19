"""
硅基流动 Vision API 本地测试脚本
支持多种视觉模型和图像分析功能
基于官方文档：https://docs.siliconflow.cn/cn/userguide/capabilities/multimodal-vision
"""
import requests
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

# 配置
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
OUTPUT_DIR = "../outputs/images"

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# 支持的视觉模型列表（根据官方文档）
SUPPORTED_MODELS = {
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",          # 推荐：稳定可用
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",    # 高性能版本
    "qwen2.5-vl-72b-pro": "Pro/Qwen/Qwen2.5-VL-72B-Instruct",  # Pro专业版
    "glm-4v": "THUDM/glm-4v-plus",                       # GLM视觉模型
    "deepseek-vl2": "deepseek-ai/DeepSeek-VL2",          # DeepSeek视觉模型
}


def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片编码为Base64
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        str: Base64编码的图片数据
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_vision_analysis(
    image_source: str,
    question: str = "请详细描述这张图片的内容",
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    detail: str = "auto",
    use_base64: bool = False
):
    """
    测试图像识别API
    
    Args:
        image_source: 图片来源（本地路径或URL）
        question: 对图片的提问
        model: 视觉模型名称
        detail: 分析详细程度 (auto/low/high)
        use_base64: 是否使用Base64编码（对本地图片）
        
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*70}")
    print(f"🖼️  图像识别测试")
    print(f"{'='*70}")
    print(f"📝 模型: {model}")
    print(f"🎨 图片: {image_source}")
    print(f"💬 提问: {question}")
    print(f"🔍 详细度: {detail}")
    print(f"{'='*70}")
    
    # 检查API Key
    if not API_KEY or API_KEY == "your_api_key_here":
        print("❌ 错误: API Key未配置！")
        print("请在 .env 文件中设置 SILICONFLOW_API_KEY")
        print("获取地址: https://cloud.siliconflow.cn/account/ak")
        return False
    
    # 处理图片URL
    if image_source.startswith(('http://', 'https://')):
        # 网络图片直接使用URL
        image_url = image_source
        print(f"📡 使用网络图片: {image_url[:60]}...")
    else:
        # 本地图片
        if not os.path.exists(image_source):
            print(f"❌ 错误: 图片文件不存在: {image_source}")
            return False
        
        if use_base64:
            # 使用Base64编码
            try:
                base64_image = encode_image_to_base64(image_source)
                image_url = f"data:image/jpeg;base64,{base64_image}"
                print(f"📦 使用Base64编码 (长度: {len(base64_image)} 字符)")
            except Exception as e:
                print(f"❌ Base64编码失败: {str(e)}")
                return False
        else:
            print("⚠️  注意: 本地图片需要使用Base64编码或上传到网络")
            print("提示: 设置 use_base64=True 自动转换")
            return False
    
    # 构建请求（根据官方文档格式）
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建消息内容（多模态格式）
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": detail
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    print(f"\n📤 发送请求到: {url}")
    print(f"📦 模型: {model}")
    print(f"🔧 详细度: {detail}")
    
    try:
        # 发送POST请求
        start_time = time.time()
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  响应时间: {elapsed_time:.2f}秒")
        print(f"📊 状态码: {response.status_code}")
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            
            # 提取回复内容
            content = result['choices'][0]['message']['content']
            
            # 提取token使用情况
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            
            print(f"\n{'='*70}")
            print(f"✅ 分析成功!")
            print(f"{'='*70}")
            print(f"\n🤖 AI 回复:\n")
            print(content)
            print(f"\n{'='*70}")
            print(f"📊 Token使用统计:")
            print(f"   - 输入 Tokens: {prompt_tokens}")
            print(f"   - 输出 Tokens: {completion_tokens}")
            print(f"   - 总计 Tokens: {total_tokens}")
            print(f"{'='*70}")
            
            return True
            
        elif response.status_code == 403:
            print(f"\n❌ 403 Forbidden - 权限错误")
            print(f"可能原因:")
            print(f"  1. API Key 没有访问该模型的权限")
            print(f"  2. 当前模型需要特殊权限")
            print(f"\n建议:")
            print(f"  1. 检查 API Key 权限: https://cloud.siliconflow.cn/account/ak")
            print(f"  2. 尝试其他模型: Qwen/Qwen2-VL-7B-Instruct")
            print(f"  3. 运行诊断脚本: python test_vision_api.py")
            return False
            
        elif response.status_code == 404:
            print(f"\n❌ 404 Not Found - 模型不存在")
            print(f"模型名称可能不正确: {model}")
            print(f"\n支持的模型列表:")
            for key, value in SUPPORTED_MODELS.items():
                print(f"  - {value}")
            return False
            
        else:
            print(f"\n❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ 请求超时")
        print("建议: 检查网络连接或增加超时时间")
        return False
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        return False


def test_multi_image_comparison(
    image1_path: str,
    image2_path: str,
    question: str = "比较这两张图片的相同点和不同点",
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
):
    """
    测试多图对比功能
    
    Args:
        image1_path: 第一张图片路径
        image2_path: 第二张图片路径
        question: 对比问题
        model: 视觉模型
        
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*70}")
    print(f"🖼️🖼️  多图对比测试")
    print(f"{'='*70}")
    
    # 检查图片是否存在
    if not os.path.exists(image1_path):
        print(f"❌ 图片1不存在: {image1_path}")
        return False
    if not os.path.exists(image2_path):
        print(f"❌ 图片2不存在: {image2_path}")
        return False
    
    # 编码图片
    try:
        base64_image1 = encode_image_to_base64(image1_path)
        base64_image2 = encode_image_to_base64(image2_path)
    except Exception as e:
        print(f"❌ 图片编码失败: {str(e)}")
        return False
    
    # 构建请求
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }
    
    print(f"📝 模型: {model}")
    print(f"🎨 图片1: {image1_path}")
    print(f"🎨 图片2: {image2_path}")
    print(f"💬 提问: {question}")
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=60)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print(f"\n✅ 对比成功! (耗时: {elapsed_time:.2f}秒)")
            print(f"\n🤖 AI 分析:\n")
            print(content)
            return True
        else:
            print(f"\n❌ 请求失败: {response.status_code}")
            print(f"错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🚀 硅基流动图像识别 API 测试套件")
    print("="*70)
    
    # 测试用例
    test_cases = [
        {
            "name": "测试1: 网络图片分析",
            "image": "https://picsum.photos/400/300",
            "question": "请描述这张图片的内容",
            "model": "Qwen/Qwen2-VL-7B-Instruct",
            "use_base64": False
        },
        {
            "name": "测试2: 本地图片分析",
            "image": "../outputs/images/test_image.jpg",  # 需要准备测试图片
            "question": "图片中有什么物体？",
            "model": "Qwen/Qwen2-VL-7B-Instruct",
            "use_base64": True
        },
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"🧪 {test['name']}")
        print(f"{'='*70}")
        
        # 跳过不存在的本地图片
        if not test['image'].startswith('http') and not os.path.exists(test['image']):
            print(f"⚠️  跳过测试: 图片不存在 ({test['image']})")
            print(f"💡 提示: 在 outputs/images/ 目录放置测试图片")
            results.append(False)
            continue
        
        success = test_vision_analysis(
            image_source=test['image'],
            question=test['question'],
            model=test['model'],
            use_base64=test.get('use_base64', False)
        )
        results.append(success)
        
        # 每次测试间隔
        if i < len(test_cases):
            print(f"\n⏳ 等待2秒...")
            time.sleep(2)
    
    # 总结
    print(f"\n{'='*70}")
    print(f"📊 测试总结")
    print(f"{'='*70}")
    print(f"总测试数: {len(results)}")
    print(f"成功: {sum(results)} ✅")
    print(f"失败: {len(results) - sum(results)} ❌")
    print(f"成功率: {sum(results)/len(results)*100:.1f}%")
    print(f"{'='*70}")


def interactive_test():
    """交互式测试"""
    print("\n" + "="*70)
    print("🎨 图像识别交互式测试")
    print("="*70)
    
    # 选择模型
    print("\n可用模型:")
    for i, (key, model) in enumerate(SUPPORTED_MODELS.items(), 1):
        print(f"  {i}. {model}")
    
    model_choice = input("\n选择模型 (1-5, 默认1): ").strip() or "1"
    model_index = int(model_choice) - 1
    model = list(SUPPORTED_MODELS.values())[model_index] if 0 <= model_index < len(SUPPORTED_MODELS) else "Qwen/Qwen2-VL-7B-Instruct"
    
    # 输入图片
    print("\n图片来源:")
    print("  1. 网络图片URL")
    print("  2. 本地图片路径")
    
    source_type = input("\n选择图片来源 (1-2): ").strip()
    
    if source_type == "1":
        image_source = input("请输入图片URL: ").strip()
        use_base64 = False
    else:
        image_source = input("请输入图片路径: ").strip()
        use_base64 = True
    
    # 输入问题
    question = input("\n请输入你的问题 (直接回车使用默认): ").strip()
    if not question:
        question = "请详细描述这张图片的内容"
    
    # 详细程度
    detail = input("\n分析详细程度 (auto/low/high, 默认auto): ").strip() or "auto"
    
    # 执行测试
    test_vision_analysis(
        image_source=image_source,
        question=question,
        model=model,
        detail=detail,
        use_base64=use_base64
    )


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🖼️  硅基流动图像识别 API 测试工具")
    print("="*70)
    print("\n使用方式:")
    print("  1. python test_vision_local.py              # 运行所有测试")
    print("  2. python test_vision_local.py interactive  # 交互式测试")
    print("  3. python test_vision_local.py <image_url>  # 快速测试指定图片")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_test()
        else:
            # 快速测试指定图片
            image_url = sys.argv[1]
            question = sys.argv[2] if len(sys.argv) > 2 else "请描述这张图片"
            test_vision_analysis(
                image_source=image_url,
                question=question,
                use_base64=not image_url.startswith('http')
            )
    else:
        run_all_tests()

