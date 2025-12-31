import json
import os
import dashscope
from dashscope import MultiModalConversation
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import traceback

# ComfyUI节点类 - 根据实际响应格式重写
class DashScopeImageGeneration:
    """阿里云百炼图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书'义本生知人机同道善思新'，右书'通云赋智乾坤启数高志远'， 横批'智启通义'，字体飘逸，在中间挂着一幅中国风的画作，内容是岳阳楼。"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False
                }),
            },
            "optional": {
                "model": (["qwen-image-plus", "qwen-image-vl-plus"], {"default": "qwen-image-plus"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "region": (["beijing", "singapore"], {"default": "beijing"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "DashScope"
    DESCRIPTION = "使用阿里云百炼API生成图像"
    
    def generate_image(self, prompt, api_key, model="qwen-image-plus", width=1024, height=1024, 
                      negative_prompt="", prompt_extend=True, watermark=False, region="beijing"):
        
        # 设置地域URL
        if region == "singapore":
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        else:
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        # 使用提供的API Key或环境变量
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            raise ValueError("API Key未提供，请在节点输入中设置或配置DASHSCOPE_API_KEY环境变量")
        
        try:
            print(f"正在生成图像，提示词: {prompt[:100]}...")
            print(f"使用模型: {model}, 尺寸: {width}x{height}, 地域: {region}")
            
            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": str(prompt)}
                    ]
                }
            ]
            
            # 调用API - 根据你的原始代码
            response = MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=messages,
                result_format='message',
                stream=False,
                watermark=watermark,
                prompt_extend=prompt_extend,
                negative_prompt=str(negative_prompt) if negative_prompt else '',
                size=f'{width}*{height}'
            )
            
            print(f"API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("API调用成功，解析响应...")
                
                # 根据你提供的响应格式解析
                response_data = response
                
                # 调试：打印响应结构
                print(f"响应类型: {type(response_data)}")
                print(f"响应属性: {dir(response_data)}")
                
                # 提取图像URL
                image_url = None
                
                # 根据你提供的响应格式解析
                if hasattr(response_data, 'output') and hasattr(response_data.output, 'choices'):
                    choices = response_data.output.choices
                    print(f"找到 {len(choices)} 个choices")
                    
                    for choice in choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            content = choice.message.content
                            print(f"content类型: {type(content)}")
                            
                            # 遍历content列表查找图像
                            if isinstance(content, list):
                                for item in content:
                                    print(f"item类型: {type(item)}")
                                    if isinstance(item, dict) and 'image' in item:
                                        image_url = item['image']
                                        print(f"找到图像URL: {image_url[:100]}...")
                                        break
                                    elif hasattr(item, 'image'):
                                        image_url = item.image
                                        print(f"找到图像URL: {image_url[:100]}...")
                                        break
                            
                            if image_url:
                                break
                
                if not image_url:
                    # 尝试其他方式查找
                    try:
                        # 将响应转换为字典查找
                        import pprint
                        response_dict = response_data.__dict__
                        # 递归查找图像URL
                        def find_image_url(data):
                            if isinstance(data, dict):
                                for k, v in data.items():
                                    if k == 'image' and isinstance(v, str) and v.startswith('http'):
                                        return v
                                    elif isinstance(v, (dict, list)):
                                        result = find_image_url(v)
                                        if result:
                                            return result
                            elif isinstance(data, list):
                                for item in data:
                                    result = find_image_url(item)
                                    if result:
                                        return result
                            return None
                        
                        image_url = find_image_url(response_dict)
                        if image_url:
                            print(f"通过递归找到图像URL: {image_url[:100]}...")
                    except Exception as e:
                        print(f"递归查找失败: {e}")
                
                if image_url:
                    print(f"开始下载图像: {image_url}")
                    
                    # 下载图像
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        
                        # 下载图像
                        response = requests.get(image_url, headers=headers, timeout=30)
                        response.raise_for_status()
                        
                        # 从字节流创建图像
                        image = Image.open(BytesIO(response.content))
                        print(f"图像下载成功: {image.size}, 模式: {image.mode}")
                        
                        # 转换为RGB模式
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                            print(f"已转换为RGB模式")
                        
                        # 检查实际尺寸
                        actual_width, actual_height = image.size
                        print(f"图像实际尺寸: {actual_width}x{actual_height}")
                        
                        # 调整大小到指定尺寸（如果需要）
                        if (actual_width, actual_height) != (width, height):
                            print(f"调整图像大小从 {actual_width}x{actual_height} 到 {width}x{height}")
                            image = image.resize((width, height), Image.Resampling.LANCZOS)
                        
                        # 转换为ComfyUI图像格式 (1, H, W, C)
                        image_np = np.array(image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_np)[None,]
                        
                        print("图像转换完成")
                        
                        # 创建返回信息
                        info_dict = {
                            "status": "success",
                            "model": model,
                            "size": f"{width}x{height}",
                            "region": region,
                            "request_id": getattr(response_data, 'request_id', 'N/A'),
                            "image_url": image_url[:100] + "..." if len(image_url) > 100 else image_url
                        }
                        
                        info = json.dumps(info_dict, ensure_ascii=False)
                        return (image_tensor, info)
                        
                    except Exception as e:
                        print(f"图像下载或处理错误: {str(e)}")
                        raise Exception(f"图像处理失败: {str(e)}")
                else:
                    # 打印响应内容用于调试
                    print("未找到图像URL，响应内容:")
                    print(json.dumps(response_data.__dict__, indent=2, ensure_ascii=False, default=str))
                    
                    raise Exception("API响应中没有找到图像URL。响应结构可能已更改。")
                    
            else:
                error_msg = f"API错误: {response.code if hasattr(response, 'code') else 'N/A'} - {response.message if hasattr(response, 'message') else '未知错误'}"
                print(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"生成图像时发生异常: {str(e)}")
            print(f"异常详情: {traceback.format_exc()}")
            raise Exception(f"生成失败: {str(e)}")


# 直接响应解析版本
class DashScopeSimpleGeneration:
    """简化版图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "一副典雅庄重的对联悬挂于厅堂之中"
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "size": (["1024x1024", "768x1024", "1024x768", "1328x1328"], {"default": "1024x1024"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "DashScope"
    DESCRIPTION = "简化版图像生成"
    
    def generate_image(self, prompt, api_key, size="1024x1024"):
        """简化版生成图像"""
        
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            raise ValueError("API Key未提供")
        
        try:
            width, height = map(int, size.split('x'))
            
            print(f"生成图像: {prompt[:50]}...")
            print(f"尺寸: {width}x{height}")
            
            # 直接使用你的原始代码
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": str(prompt)}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(
                api_key=api_key,
                model="qwen-image-plus",
                messages=messages,
                result_format='message',
                stream=False,
                watermark=False,
                prompt_extend=True,
                negative_prompt='',
                size=f'{width}*{height}'
            )
            
            if response.status_code == 200:
                print("API调用成功")
                
                # 直接按照你提供的响应格式解析
                output = response.output
                choices = output.choices
                
                # 提取图像URL
                image_url = None
                for choice in choices:
                    message = choice.message
                    content = message.content
                    
                    for item in content:
                        if isinstance(item, dict) and 'image' in item:
                            image_url = item['image']
                            break
                
                if image_url:
                    print(f"图像URL: {image_url[:100]}...")
                    
                    # 下载图像
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    img_response = requests.get(image_url, headers=headers, timeout=30)
                    
                    if img_response.status_code == 200:
                        image = Image.open(BytesIO(img_response.content))
                        
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        if image.size != (width, height):
                            image = image.resize((width, height), Image.Resampling.LANCZOS)
                        
                        # 转换为tensor
                        image_np = np.array(image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_np)[None,]
                        
                        info_dict = {
                            "status": "success",
                            "model": "qwen-image-plus",
                            "size": f"{width}x{height}",
                            "request_id": response.request_id
                        }
                        
                        info = json.dumps(info_dict, ensure_ascii=False)
                        return (image_tensor, info)
                    else:
                        raise Exception(f"下载图像失败: {img_response.status_code}")
                else:
                    raise Exception("未找到图像URL")
            else:
                raise Exception(f"API错误: {response.code} - {response.message}")
                
        except Exception as e:
            print(f"错误: {str(e)}")
            raise


# 批量生成节点
class DashScopeBatchGeneration:
    """批量生成图像"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "multiline": True, 
                    "default": "一副典雅庄重的对联悬挂于厅堂之中\n一幅美丽的山水画\n一个可爱的小猫"
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 4}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate_batch"
    CATEGORY = "DashScope"
    OUTPUT_IS_LIST = (True, False)
    
    def generate_batch(self, prompts, api_key, batch_size, width=1024, height=1024):
        """批量生成图像"""
        
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            raise ValueError("API Key未提供")
        
        # 分割提示词
        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        if not prompt_list:
            raise ValueError("请输入至少一个提示词")
        
        # 限制batch大小
        prompt_list = prompt_list[:batch_size]
        
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        images = []
        results_info = []
        
        for i, prompt in enumerate(prompt_list):
            try:
                print(f"生成第 {i+1}/{len(prompt_list)} 张图像: {prompt[:50]}...")
                
                messages = [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
                
                response = MultiModalConversation.call(
                    api_key=api_key,
                    model="qwen-image-plus",
                    messages=messages,
                    result_format='message',
                    stream=False,
                    watermark=False,
                    prompt_extend=True,
                    negative_prompt='',
                    size=f'{width}*{height}'
                )
                
                if response.status_code == 200:
                    # 提取图像URL
                    image_url = None
                    for choice in response.output.choices:
                        for item in choice.message.content:
                            if isinstance(item, dict) and 'image' in item:
                                image_url = item['image']
                                break
                        if image_url:
                            break
                    
                    if image_url:
                        # 下载图像
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        img_response = requests.get(image_url, headers=headers, timeout=30)
                        
                        if img_response.status_code == 200:
                            image = Image.open(BytesIO(img_response.content))
                            
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            if image.size != (width, height):
                                image = image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            # 转换为tensor
                            image_np = np.array(image).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_np)[None,]
                            
                            images.append(image_tensor)
                            results_info.append({
                                "index": i,
                                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                                "status": "success",
                                "request_id": response.request_id
                            })
                        else:
                            results_info.append({
                                "index": i,
                                "prompt": prompt,
                                "status": "error",
                                "error": f"下载失败: {img_response.status_code}"
                            })
                    else:
                        results_info.append({
                            "index": i,
                            "prompt": prompt,
                            "status": "error",
                            "error": "无图像URL"
                        })
                else:
                    results_info.append({
                        "index": i,
                        "prompt": prompt,
                        "status": "error",
                        "error": f"API错误: {response.code}"
                    })
                    
            except Exception as e:
                results_info.append({
                    "index": i,
                    "prompt": prompt,
                    "status": "exception",
                    "error": str(e)
                })
        
        # 合并所有成功生成的图像
        if images:
            combined_images = torch.cat(images, dim=0)
            info_dict = {
                "total": len(prompt_list),
                "success": len(images),
                "results": results_info
            }
            info = json.dumps(info_dict, ensure_ascii=False)
            return (combined_images, info)
        else:
            raise Exception(f"所有生成都失败: {json.dumps(results_info, ensure_ascii=False)}")


# 节点注册
NODE_CLASS_MAPPINGS = {
    "DashScopeImageGeneration": DashScopeImageGeneration,
    "DashScopeSimpleGeneration": DashScopeSimpleGeneration,
    "DashScopeBatchGeneration": DashScopeBatchGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashScopeImageGeneration": "DashScope 图像生成",
    "DashScopeSimpleGeneration": "DashScope 简单生成",
    "DashScopeBatchGeneration": "DashScope 批量生成",
}

print("""
✅ DashScope 节点已成功加载！
使用说明：
1. 确保已安装依赖：pip install dashscope requests Pillow numpy torch
2. 在节点中输入你的 API Key，或设置环境变量 DASHSCOPE_API_KEY

响应格式确认：
- API返回图像URL，节点会自动下载并转换为ComfyUI图像格式
- 支持批量生成
- 支持自定义尺寸

注意：
1. 图像URL有有效期，请及时使用
2. 批量生成时注意API调用频率限制
3. 生成的图像会自动调整到指定尺寸

推荐节点：
1. DashScope 简单生成：最接近原始代码的版本
2. DashScope 图像生成：功能更完整
3. DashScope 批量生成：一次生成多张图像
""")
