# 文件名: main.py
# -*- coding: utf-8 -*-
import os
import sys
from gen_image import generate_static_image
from animate_only import AnimationEngine

# ===================== 🔧 终极配置 =====================
# 1. 模型路径
BASE_MODEL = r"E:\huggingface_cache\Counterfeit-V3.0\Counterfeit-V3.0_fix_fp16.safetensors"
MOTION_MODULE = r"E:\huggingface_cache\animatediff-motion-adapter-v1-5-3"
IP_ADAPTER = r"E:\huggingface_cache\IP-Adapter"

# 2. 插件路径 (指向包含 config.json 和 safetensors 的文件夹)
VAE_PATH = r"E:\huggingface_cache\vae" 
EMBEDDING_PATH = r"E:\huggingface_cache\embeddings\easynegative.safetensors"

# 3. 输出配置
OUTPUT_DIR = "output"
# ==========================================================

def main():
    print("=== 虚拟女友生成器 (外挂VAE版) ===")
    
    # 路径自检
    for p in [BASE_MODEL, MOTION_MODULE, IP_ADAPTER, VAE_PATH]:
        if not os.path.exists(p):
            print(f"❌ 严重错误: 找不到文件或目录: {p}")
            return

    current_image_path = ""
    Create_image = True
    preview_path = os.path.join(OUTPUT_DIR, "preview_wife.png")
        
    if os.path.exists(preview_path):
        Create_image = False
        user_input = input(">> [系统] 检测到已有图片，是否需要重新生成。(Y/N)? :").lower()
        if user_input == 'y':
            try:
                os.remove(preview_path)
            except OSError:
                pass 
            Create_image = True
        else:
            Create_image = False
            current_image_path = preview_path

    # 提示词
    base_prompt = "masterpiece, best quality, 1girl, silver hair, red eyes, white dress, upper body, gentle smile, looking at viewer, highres, solo, big breast, sexy expression"
    neg_prompt = (
            "easynegative, (low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), "
            "bad composition, inaccurate eyes, extra digit, fewer digits, (extra arms:1.2), "
            "bad hands, missing fingers, extra limbs, "
            "blurry, ugly, deformed, noisy, texture, jpeg artifacts, signature, watermark"
        )
    
    # --- 阶段一：抽卡 ---
    while Create_image:
        current_image_path = generate_static_image(
            base_model_path=BASE_MODEL,
            vae_path=VAE_PATH,          # 传入外部 VAE
            embedding_path=EMBEDDING_PATH, 
            prompt=base_prompt,
            neg_prompt=neg_prompt,
            output_dir=OUTPUT_DIR,
            filename="preview_wife.png"
        )
        
        print(f"\n请查看预览图: {current_image_path}")
        user_input = input(">> 满意吗？(Enter: 继续 / r: 重抽 / q: 退出): ").lower()
        
        if user_input == 'q': sys.exit()
        elif user_input == 'r': continue
        else: break 

    # --- 阶段二：设置时长 ---
    print("\n" + "-"*30)
    print(">> 请设置视频参数")
    try:
        user_frames = input(">> 请输入生成帧数 [默认为16]: ")
        num_frames = int(user_frames) if user_frames.strip() else 16
        if num_frames > 32: num_frames = 32
    except ValueError:
        num_frames = 16

    # 动画引擎初始化
    print("\n>> 正在初始化动画引擎...")
    engine = AnimationEngine(
        base_model_path=BASE_MODEL,
        motion_module_path=MOTION_MODULE,
        ip_adapter_path=IP_ADAPTER,
        vae_path=VAE_PATH,              # 传入外部 VAE
        embedding_path=EMBEDDING_PATH   
    )
    
    action_prompt = f"{base_prompt}, reading book, turning page, wind blowing hair, upper body, detailed face"
    
    final_gif_path = os.path.join(OUTPUT_DIR, f"final_wife_{num_frames}frames.gif")
    
    engine.run(
        image_path=current_image_path,
        action_prompt=action_prompt,
        neg_prompt=neg_prompt,
        output_path=final_gif_path,
        num_frames=num_frames,
        fps=8 
    )
    
    print("\n>> 全部流程结束！")

if __name__ == "__main__":
    main()