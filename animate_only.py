# 文件名: animate_only.py
import torch
import os
import gc
from diffusers import (
    AnimateDiffPipeline, 
    StableDiffusionPipeline, 
    MotionAdapter, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
from diffusers.utils import export_to_gif, load_image

class AnimationEngine:
    # 接收 vae_path
    def __init__(self, base_model_path, motion_module_path, ip_adapter_path, vae_path, embedding_path):
        print(f"\n>> [模块调用] 正在初始化动画引擎...")
        
        adapter = MotionAdapter.from_pretrained(motion_module_path, torch_dtype=torch.float16)

        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            steps_offset=1,
            timestep_spacing="linspace"
        )

        # 1. 加载外部 VAE
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        # 【关键】防黑图开关
        vae.config.force_upcast = True

        # 2. 加载 SD (注入 VAE)
        sd_pipe = StableDiffusionPipeline.from_single_file(
            base_model_path,
            vae=vae,  # 使用外部 VAE
            torch_dtype=torch.float16,
            load_safety_checker=False 
        )

        # 【关键】暴力移除过滤器
        sd_pipe.safety_checker = None
        sd_pipe.requires_safety_checker = False

        if os.path.exists(embedding_path):
            try:
                sd_pipe.load_textual_inversion(embedding_path, token="easynegative")
            except:
                pass

        # 3. 组装 AnimateDiff
        self.pipe = AnimateDiffPipeline(
            unet=sd_pipe.unet,
            vae=sd_pipe.vae,    # 这里就是我们配置好的外部 VAE
            text_encoder=sd_pipe.text_encoder,
            tokenizer=sd_pipe.tokenizer,
            scheduler=scheduler,         
            motion_adapter=adapter,      
            feature_extractor=sd_pipe.feature_extractor
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
            prompt=action_prompt,
            negative_prompt=neg_prompt,
            ip_adapter_image=reference_image,
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=100,
            generator=generator,
            width=512, height=512
        )
        
        duration_ms = int(1000 / fps)
        frames = output.frames[0]
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0
        )
        print(f">> [成功] 动画已保存至: {output_path}")

if __name__ == "__main__":
    pass