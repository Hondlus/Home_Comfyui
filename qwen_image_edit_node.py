import json
import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import random
from typing import Dict, List, Optional, Tuple
import folder_paths

# 导入Dashscope SDK
try:
    from dashscope import MultiModalConversation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告: dashscope 库未安装。请运行: pip install dashscope")

class QwenImageEditPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {
                    "label": "图像1 (必填)"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "图1中的女生穿着图2中的黑色裙子按图3的姿势坐下"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""  # 留空则从环境变量读取
                }),
                "num_outputs": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "image2": ("IMAGE", {
                    "label": "图像2 (可选)"
                }),
                "image3": ("IMAGE", {
                    "label": "图像3 (可选)"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低质量"
                }),
                "prompt_extend": (["true", "false"], {
                    "default": "true"
                }),
                "watermark": (["true", "false"], {
                    "default": "false"
                }),
                "region": (["beijing", "singapore"], {
                    "default": "beijing"
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "🦊 Qwen/Image Edit"
    OUTPUT_IS_LIST = (True,)
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
    
    def image_to_base64(self, image: torch.Tensor) -> str:
        """将ComfyUI图像张量转换为base64编码的URL格式"""
        # 确保图像在正确范围内
        if image.dim() == 4:
            image = image[0]
        image = image.permute(2, 0, 1)  # HWC to CHW
        
        # 转换为numpy并调整范围
        image_np = image.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 转换回PIL图像
        image_np = image_np.transpose(1, 2, 0)  # CHW to HWC
        pil_image = Image.fromarray(image_np)
        
        # 保存到内存并转换为base64
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 转换为base64 data URL
        base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def download_image(self, url: str) -> torch.Tensor:
        """从URL下载图像并转换为ComfyUI格式"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 打开图像
            image = Image.open(io.BytesIO(response.content))
            
            # 转换为RGB模式（移除alpha通道）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为torch张量
            image_tensor = torch.from_numpy(image_np)[None, ...]
            
            return image_tensor
            
        except Exception as e:
            raise Exception(f"下载图像失败: {str(e)}")
    
    def generate_images(self, image1, prompt, api_key, num_outputs, 
                       image2=None, image3=None, negative_prompt="低质量",
                       prompt_extend="true", watermark="false", region="beijing", seed=-1):
        
        # 检查dashscope是否可用
        if not DASHSCOPE_AVAILABLE:
            raise Exception("dashscope 库未安装。请运行: pip install dashscope")
        
        # 获取API密钥
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise Exception("请提供API密钥或在环境变量中设置 DASHSCOPE_API_KEY")
        
        # 设置地域URL
        if region == "singapore":
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        else:
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        # 收集所有输入的图像
        input_images = []
        if image1 is not None:
            input_images.append(image1)
        if image2 is not None:
            input_images.append(image2)
        if image3 is not None:
            input_images.append(image3)
        
        if len(input_images) == 0:
            raise Exception("至少需要提供一张输入图像")
        
        print(f"输入图像数量: {len(input_images)}")
        
        # 转换输入图像为base64格式
        content = []
        for i, image in enumerate(input_images):
            if i >= 3:  # 最多3张输入图像
                break
            try:
                image_base64 = self.image_to_base64(image)
                content.append({"image": image_base64})
                print(f"图像{i+1}转换完成")
            except Exception as e:
                raise Exception(f"转换图像{i+1}失败: {str(e)}")
        
        # 添加文本提示
        content.append({"text": prompt})
        
        # 构建消息
        messages = [{
            "role": "user",
            "content": content
        }]
        
        # 准备调用参数
        call_kwargs = {
            "api_key": api_key,
            "model": "qwen-image-edit-plus",
            "messages": messages,
            "stream": False,
            "n": num_outputs,
            "watermark": watermark == "true",
            "negative_prompt": negative_prompt,
            "prompt_extend": prompt_extend == "true",
        }
        
        # 处理seed参数：修复范围问题
        if seed != -1:
            # 确保seed在有效范围内
            if seed > 2147483647:
                print(f"警告: seed值 {seed} 超出API限制(2147483647)，自动调整为有效值")
                seed = seed % 2147483647  # 使用取模确保在范围内
            call_kwargs["seed"] = seed
        else:
            # 如果seed为-1，生成一个随机种子（在有效范围内）
            random_seed = random.randint(0, 2147483647)
            call_kwargs["seed"] = random_seed
        
        print(f"API调用参数: seed={call_kwargs.get('seed')}, num_outputs={num_outputs}, 输入图像数={len(input_images)}")
        
        # 调用API
        try:
            print(f"正在调用Qwen Image Edit Plus API，生成 {num_outputs} 张图像...")
            response = MultiModalConversation.call(**call_kwargs)
            
            if response.status_code == 200:
                print("API调用成功!")
                
                # 下载所有生成的图像
                output_images = []
                for i, content_item in enumerate(response.output.choices[0].message.content):
                    if "image" in content_item:
                        image_url = content_item["image"]
                        print(f"下载图像 {i+1}: {image_url}")
                        image_tensor = self.download_image(image_url)
                        output_images.append(image_tensor)
                
                if not output_images:
                    raise Exception("API响应中没有找到图像数据")
                
                print(f"成功下载 {len(output_images)} 张图像")
                return (output_images,)
                
            else:
                error_msg = f"API调用失败:\n"
                error_msg += f"HTTP返回码: {response.status_code}\n"
                error_msg += f"错误码: {getattr(response, 'code', 'N/A')}\n"
                error_msg += f"错误信息: {getattr(response, 'message', 'N/A')}"
                
                # 打印完整的错误信息以便调试
                print(f"完整响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
                
                raise Exception(error_msg)
                
        except Exception as e:
            # 提供更详细的错误信息
            error_details = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                try:
                    error_json = json.loads(e.response.text)
                    error_details = json.dumps(error_json, indent=2, ensure_ascii=False)
                except:
                    error_details = e.response.text
            
            raise Exception(f"生成图像时出错: {error_details}")

# 一个更简单的版本，适合大多数使用场景
class QwenImageEdit3Inputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {
                    "label": "图像1 (参考图)"
                }),
                "image2": ("IMAGE", {
                    "label": "图像2 (参考图)"
                }),
                "image3": ("IMAGE", {
                    "label": "图像3 (参考图)"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "图1中的女生穿着图2中的黑色裙子按图3的姿势坐下"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "num_outputs": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低质量"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "🦊 Qwen/Image Edit"
    OUTPUT_IS_LIST = (True,)
    
    def generate_images(self, image1, image2, image3, prompt, api_key, num_outputs, 
                       negative_prompt="低质量"):
        """简化版本，专为3张输入图像设计"""
        if not DASHSCOPE_AVAILABLE:
            raise Exception("dashscope 库未安装")
        
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise Exception("请提供API密钥")
        
        # 设置地域URL（默认北京）
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        # 转换3张输入图像
        content = []
        
        # 转换图像1
        try:
            base64_1 = self.image_to_base64(image1)
            content.append({"image": base64_1})
        except Exception as e:
            raise Exception(f"转换图像1失败: {str(e)}")
        
        # 转换图像2
        try:
            base64_2 = self.image_to_base64(image2)
            content.append({"image": base64_2})
        except Exception as e:
            raise Exception(f"转换图像2失败: {str(e)}")
        
        # 转换图像3
        try:
            base64_3 = self.image_to_base64(image3)
            content.append({"image": base64_3})
        except Exception as e:
            raise Exception(f"转换图像3失败: {str(e)}")
        
        # 添加文本提示
        content.append({"text": prompt})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        # 使用最简化的参数调用
        try:
            print(f"正在调用Qwen Image Edit Plus API，使用3张输入图像生成 {num_outputs} 张输出...")
            
            response = MultiModalConversation.call(
                api_key=api_key,
                model="qwen-image-edit-plus",
                messages=messages,
                stream=False,
                n=num_outputs,
                watermark=False,
                negative_prompt=negative_prompt,
                prompt_extend=True,
            )
            
            if response.status_code == 200:
                output_images = []
                for i, content_item in enumerate(response.output.choices[0].message.content):
                    if "image" in content_item:
                        print(f"下载输出图像 {i+1}")
                        image_tensor = self.download_image(content_item["image"])
                        output_images.append(image_tensor)
                
                if not output_images:
                    raise Exception("API响应中没有找到图像数据")
                
                print(f"成功生成 {len(output_images)} 张图像")
                return (output_images,)
            else:
                error_msg = f"API调用失败: {response.status_code}\n"
                error_msg += f"错误信息: {response.message}"
                raise Exception(error_msg)
                
        except Exception as e:
            raise Exception(f"生成图像时出错: {str(e)}")
    
    def image_to_base64(self, image: torch.Tensor) -> str:
        """将ComfyUI图像张量转换为base64"""
        if image.dim() == 4:
            image = image[0]
        image = image.permute(2, 0, 1)
        image_np = image.cpu().numpy()
        
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        image_np = image_np.transpose(1, 2, 0)
        pil_image = Image.fromarray(image_np)
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        base64_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def download_image(self, url: str) -> torch.Tensor:
        """下载图像"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None, ...]

# 支持1-3张图像的灵活版本
class QwenImageEditFlexible:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "图1中的女生穿着图2中的黑色裙子按图3的姿势坐下"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "num_outputs": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 6,
                    "step": 1
                }),
            },
            "optional": {
                "image1": ("IMAGE", {
                    "label": "图像1"
                }),
                "image2": ("IMAGE", {
                    "label": "图像2"
                }),
                "image3": ("IMAGE", {
                    "label": "图像3"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "低质量"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "🦊 Qwen/Image Edit"
    OUTPUT_IS_LIST = (True,)
    
    def generate_images(self, prompt, api_key, num_outputs, 
                       image1=None, image2=None, image3=None,
                       negative_prompt="低质量"):
        """灵活版本，支持1-3张输入图像"""
        if not DASHSCOPE_AVAILABLE:
            raise Exception("dashscope 库未安装")
        
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise Exception("请提供API密钥")
        
        # 收集有效的输入图像
        input_images = []
        image_names = []
        
        if image1 is not None:
            input_images.append(image1)
            image_names.append("图像1")
        if image2 is not None:
            input_images.append(image2)
            image_names.append("图像2")
        if image3 is not None:
            input_images.append(image3)
            image_names.append("图像3")
        
        if len(input_images) == 0:
            raise Exception("至少需要提供一张输入图像")
        
        print(f"输入图像: {len(input_images)}张 ({', '.join(image_names)})")
        
        # 转换图像
        content = []
        for i, image in enumerate(input_images):
            try:
                base64_image = self.image_to_base64(image)
                content.append({"image": base64_image})
                print(f"{image_names[i]} 转换完成")
            except Exception as e:
                raise Exception(f"转换{image_names[i]}失败: {str(e)}")
        
        # 添加文本提示
        content.append({"text": prompt})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        # 调用API
        try:
            print(f"正在调用API，生成 {num_outputs} 张图像...")
            
            response = MultiModalConversation.call(
                api_key=api_key,
                model="qwen-image-edit-plus",
                messages=messages,
                stream=False,
                n=num_outputs,
                watermark=False,
                negative_prompt=negative_prompt,
                prompt_extend=True,
            )
            
            if response.status_code == 200:
                output_images = []
                for i, content_item in enumerate(response.output.choices[0].message.content):
                    if "image" in content_item:
                        print(f"下载输出图像 {i+1}")
                        image_tensor = self.download_image(content_item["image"])
                        output_images.append(image_tensor)
                
                print(f"成功生成 {len(output_images)} 张图像")
                return (output_images,)
            else:
                raise Exception(f"API调用失败: {response.message}")
                
        except Exception as e:
            raise Exception(f"生成图像时出错: {str(e)}")
    
    def image_to_base64(self, image: torch.Tensor) -> str:
        """将ComfyUI图像张量转换为base64"""
        if image.dim() == 4:
            image = image[0]
        image = image.permute(2, 0, 1)
        image_np = image.cpu().numpy()
        
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        image_np = image_np.transpose(1, 2, 0)
        pil_image = Image.fromarray(image_np)
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        base64_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def download_image(self, url: str) -> torch.Tensor:
        """下载图像"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None, ...]

# 节点注册
NODE_CLASS_MAPPINGS = {
    "QwenImageEditPlus": QwenImageEditPlus,
    "QwenImageEdit3Inputs": QwenImageEdit3Inputs,
    "QwenImageEditFlexible": QwenImageEditFlexible,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditPlus": "🦊 Qwen Image Edit Plus",
    "QwenImageEdit3Inputs": "🦊 Qwen 3图编辑",
    "QwenImageEditFlexible": "🦊 Qwen 灵活编辑",
}

# 使用说明
"""
更新说明:
1. 现在支持3张独立的图像输入:
   - image1: 图像1 (必填)
   - image2: 图像2 (可选)
   - image3: 图像3 (可选)

2. 提供了三种节点:
   - QwenImageEditPlus: 完整功能版
   - QwenImageEdit3Inputs: 专为3张输入图像设计
   - QwenImageEditFlexible: 灵活版本，支持1-3张图像

3. 使用建议:
   - 如果需要处理正好3张图像，使用 QwenImageEdit3Inputs
   - 如果需要灵活处理1-3张图像，使用 QwenImageEditFlexible
   - 如果需要所有高级参数，使用 QwenImageEditPlus

4. 工作流示例:
   Load Image 1 → QwenImageEdit3Inputs → Save Image
   Load Image 2 ↗
   Load Image 3 ↗
"""