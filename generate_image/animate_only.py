# -*- coding: utf-8 -*-
import os
import warnings
from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging
warnings.filterwarnings("ignore")
diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

import torch
import gc
import imageio
import numpy as np # 新增依赖
from diffusers import (
    AnimateDiffPipeline, 
    StableDiffusionPipeline, 
    MotionAdapter, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
from diffusers.utils import load_image

class AnimationEngine:
    def __init__(self, base_model_path, motion_module_path, ip_adapter_path, vae_path, embedding_path):
        print(f"\n>> [模块调用] 正在初始化动画引擎...")
        
        adapter = MotionAdapter.from_pretrained(motion_module_path, torch_dtype=torch.float16, local_files_only=True)
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="linear",
            steps_offset=1, timestep_spacing="linspace"
        )
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, local_files_only=True)
        vae.config.force_upcast = True

        sd_pipe = StableDiffusionPipeline.from_single_file(
            base_model_path, vae=vae, torch_dtype=torch.float16,
            load_safety_checker=False, local_files_only=True
        )
        sd_pipe.safety_checker = None
        sd_pipe.requires_safety_checker = False

        if os.path.exists(embedding_path):
            try: sd_pipe.load_textual_inversion(embedding_path, token="easynegative")
            except: pass

        self.pipe = AnimateDiffPipeline(
            unet=sd_pipe.unet, vae=sd_pipe.vae, text_encoder=sd_pipe.text_encoder,
            tokenizer=sd_pipe.tokenizer, scheduler=scheduler,         
            motion_adapter=adapter, feature_extractor=sd_pipe.feature_extractor,
        )
        del sd_pipe
        gc.collect()
        torch.cuda.empty_cache()

        print(">> [系统] 加载 IP-Adapter...")
        self.pipe.load_ip_adapter(ip_adapter_path, subfolder="models", weight_name="ip-adapter_sd15.bin")
        self.pipe.set_ip_adapter_scale(0.7)
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()

    def run(self, image_path, action_prompt, neg_prompt, output_path, num_frames=16, fps=8):
        print(f">> [驱动] 读取参考图: {image_path}")
        reference_image = load_image(image_path)
        print(f">> [驱动] 生成动作: {action_prompt[:30]}...")
        
        generator = torch.Generator("cpu").manual_seed(torch.randint(0, 1000000, (1,)).item())
        
        output = self.pipe(
            prompt=action_prompt, negative_prompt=neg_prompt,
            ip_adapter_image=reference_image, num_frames=num_frames,
            guidance_scale=7.5, num_inference_steps=25, # 降到25步，速度更快且足够了
            generator=generator, width=512, height=512
        )
        
        frames = output.frames[0]
        
        # === 核心修改：使用 imageio 保存为高清 MP4 ===
        print(f">> [保存] 正在封装视频流: {output_path}")
        
        # 将 PIL Image 转为 numpy 数组
        np_frames = [np.array(frame) for frame in frames]
        
        # 使用 imageio 保存，quality=9 表示高质量
        imageio.mimsave(output_path, np_frames, fps=fps, quality=9, macro_block_size=None)
        
        print(f">> [成功] 动画已保存至: {output_path}")

if __name__ == "__main__":
    pass