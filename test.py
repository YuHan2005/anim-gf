# -*- coding: utf-8 -*-
import torch
from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, MotionAdapter, EulerAncestralDiscreteScheduler

# === 你的本地路径配置 ===
BASE_MODEL = r"E:\huggingface_cache\anything-v4.0"
MOTION_MODULE = r"E:\huggingface_cache\animatediff-motion-adapter-v1-5-3"

def test_level_1_basic_sd():
    print("\n========== 测试等级 1: 纯静态图片生成 (不含动画/不含IP-Adapter) ==========")
    print("目标: 验证 Anything V4 模型文件本身是否损坏，以及 VAE 是否能在 fp16 下工作。")
    
    try:
        # 使用最原始的 StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe.to("cuda")
        
        # 强制设置调度器 (关键点：防止配置冲突)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        # 生成
        print(">> 正在生成测试图 (static_test.png)...")
        image = pipe(
            prompt="1girl, smile, simple background", 
            negative_prompt="low quality", 
            num_inference_steps=20
        ).images[0]
        
        image.save("debug_level_1.png")
        print(">> [成功] Level 1 通过！请查看 debug_level_1.png 是否清晰。")
        del pipe
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f">> [失败] Level 1 报错: {e}")
        return False

def test_level_2_animation():
    print("\n========== 测试等级 2: 基础动画生成 (不含 IP-Adapter) ==========")
    print("目标: 验证 Motion Adapter 是否与 Base Model 冲突 (通常是 Scheduler 配置问题)。")
    
    try:
        adapter = MotionAdapter.from_pretrained(MOTION_MODULE, torch_dtype=torch.float16)
        pipe = AnimateDiffPipeline.from_pretrained(
            BASE_MODEL,
            motion_adapter=adapter,
            torch_dtype=torch.float16
        )
        pipe.to("cuda")
        
        # =======================================================
        # 【最大嫌疑点】AnimateDiff 必须使用 linear beta_schedule
        # 很多 SD1.5 模型默认是 scaled_linear，这会导致全是噪点
        # =======================================================
        print(">> 正在修正调度器配置...")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            beta_schedule="linear",   # <--- 强制改为 linear
            timestep_spacing="linspace",
            steps_offset=1
        )
        
        # 开启 VAE Slicing 防止显存不够
        pipe.enable_vae_slicing()

        print(">> 正在生成测试动画 (anim_test.gif)...")
        output = pipe(
            prompt="1girl, smile, waving hand, simple background",
            negative_prompt="low quality, error",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=20,
        )
        
        from diffusers.utils import export_to_gif
        export_to_gif(output.frames[0], "debug_level_2.gif")
        print(">> [成功] Level 2 通过！请查看 debug_level_2.gif 是否有噪点。")
        return True
    except Exception as e:
        print(f">> [失败] Level 2 报错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 第一步：先跑基础模型
    if test_level_1_basic_sd():
        # 第二步：如果基础模型没问题，再跑动画模型
        test_level_2_animation()
    else:
        print("\n>> 致命错误：连基础模型(Level 1)都跑不通，说明模型文件损坏或显卡环境有问题。")
        print(">> 建议：重新下载 Anything-v4.0 或检查 CUDA。")