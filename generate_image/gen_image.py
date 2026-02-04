
# -*- coding: utf-8 -*-

# === 粘贴静音补丁开始 ===
import os
import warnings
from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging
warnings.filterwarnings("ignore")
diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()
# === 粘贴静音补丁结束 ===


import torch
import os
import gc
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL

def generate_static_image(
    base_model_path, 
    vae_path,          # 接收参数
    embedding_path,    
    prompt, 
    neg_prompt, 
    output_dir="output", 
    filename="current_preview.png"
):
    print(f"\n>> [模块调用] 正在加载文生图模型...")

    # 1. 加载外部 VAE (文件夹模式)
    # 使用 FP16 加载，避免 "Input type mismatch" 报错
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    
    # 开启这个开关，VAE 会在计算时自动临时转为 FP32，防止数值溢出变黑
    vae.config.force_upcast = True 
    
    # 2. 加载管道 (注入外部 VAE)
    pipe = StableDiffusionPipeline.from_single_file(
        base_model_path,
        vae=vae,                      # 使用我们加载好的 VAE
        torch_dtype=torch.float16, 
        load_safety_checker=False,
        local_files_only=True
    )

    # 【关键防黑图设置2】暴力移除安全过滤器,修改为false之后，可以生成违禁图
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # 注入 Embedding
    if os.path.exists(embedding_path):
        print(f">> [系统] 正在注入 Embedding: EasyNegative...")
        try:
            pipe.load_textual_inversion(embedding_path, token="easynegative")
        except Exception as e:
            print(f"!! [警告] Embedding 加载失败: {e}")




    pipe.to("cuda")
    pipe.enable_model_cpu_offload() 
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f">> [画师] 正在绘制: {prompt[:30]}...")
    
    # 生成
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        width=512,
        height=512,
        num_inference_steps=700, 
        guidance_scale=8.0
    ).images[0]
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    image.save(save_path)
    print(f">> [完成] 图片已保存: {save_path}")
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path