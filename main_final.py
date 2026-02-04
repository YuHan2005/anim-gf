# -*- coding: utf-8 -*-
import os
import time
import subprocess
import platform

# 导入我们之前写好的所有模块
from language.brain import AIBrain
from speech.voice import AIVoice
from lip.lipsync import LipSyncEngine
from generate_image.gen_image import generate_static_image
from generate_image.animate_only import AnimationEngine

# ================= 🔧 全局配置路径 =================
BASE_MODEL = r"E:\huggingface_cache\hassaku\hassakuSD15_v13.safetensors"
MOTION_MODULE = r"E:\huggingface_cache\animatediff-motion-adapter-v1-5-3"
IP_ADAPTER = r"E:\huggingface_cache\IP-Adapter"
VAE_PATH = r"E:\huggingface_cache\vae" 
EMBEDDING_PATH = r"E:\huggingface_cache\embeddings\easynegative.safetensors"

OUTPUT_DIR = "output_chat"
AVATAR_IMG = os.path.join(OUTPUT_DIR, "avatar_base.png")
TEMPLATE_VIDEO = os.path.join(OUTPUT_DIR, "template_idle.mp4")
AUDIO_TEMP = os.path.join(OUTPUT_DIR, "response_audio.mp3")
# ======================================================

def open_video(path):
    """跨平台打开视频文件"""
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception as e:
        print(f"打开视频失败: {e}")

def main():
    print("=== 🤖 AI 女友 (极速语音版) 启动中... ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 🚀 阶段一：视觉形象 (保留作为动态壁纸)
    # ==========================================
    need_init_visuals = not os.path.exists(TEMPLATE_VIDEO)
    
    if need_init_visuals:
        print("\n[1/3] 正在生成女友的动态形象 (仅需一次)...")
        
        # A. 生成静态图
        if not os.path.exists(AVATAR_IMG):
            # 稳重的 Prompt
            prompt = (
                "masterpiece, best quality, 1girl, solo, silver hair, red eyes, "
                "white dress, looking at viewer, shy, blushing, upper body, "
                "soft lighting, high resolution, closed mouth, smile" 
            )
            neg_prompt = "easynegative, nsfw, worst quality, low quality, open mouth"
            
            print("   >> 正在绘制女友照片...")
            generate_static_image(
                base_model_path=BASE_MODEL,
                vae_path=VAE_PATH,
                embedding_path=EMBEDDING_PATH,
                prompt=prompt,
                neg_prompt=neg_prompt,
                output_dir=OUTPUT_DIR,
                filename="avatar_base.png"
            )
        
        # B. 生成待机动画
        print("   >> 正在生成待机动作视频...")
        animator = AnimationEngine(
            base_model_path=BASE_MODEL,
            motion_module_path=MOTION_MODULE,
            ip_adapter_path=IP_ADAPTER,
            vae_path=VAE_PATH,
            embedding_path=EMBEDDING_PATH
        )
        
        # 使用稳重的动作 Prompt
        animator.run(
            image_path=AVATAR_IMG,
            action_prompt="best quality, 1girl, static pose, breathing, blinking, looking at viewer, minimal head movement",
            neg_prompt="worst quality, low quality, distortion, morphing, open mouth",
            output_path=TEMPLATE_VIDEO,
            num_frames=16, 
            fps=8
        )
        del animator
    else:
        print("\n[1/3] ✅ 形象已准备就绪！")
        # 启动时自动打开这个视频，用户可以手动设置循环播放，假装她在听
        print(">> 正在打开待机视频，请将其设置为【循环播放】...")
        open_video(TEMPLATE_VIDEO)

    # ==========================================
    # 🚀 阶段二：加载核心模块
    # ==========================================
    print("\n[2/3] 正在唤醒大脑 (Llama-3)...")
    brain = AIBrain()
    
    print("\n[3/3] 正在准备声音 (EdgeTTS)...")
    voice = AIVoice()
    
    # ❌ 删除了 Wav2Lip 加载

    # ==========================================
    # 🚀 阶段三：极速聊天循环
    # ==========================================
    print("\n" + "="*40)
    print("💖 女友已上线！(响应速度已大幅提升)")
    print("="*40)

    while True:
        user_input = input("\n👤 你: ")
        
        if user_input.lower() in ['q', 'exit']:
            break
        if user_input.lower() == 'reset':
            print(">> [指令] 删除旧形象...")
            if os.path.exists(TEMPLATE_VIDEO): os.remove(TEMPLATE_VIDEO)
            if os.path.exists(AVATAR_IMG): os.remove(AVATAR_IMG)
            break 

        # A. 思考 (秒回)
        start_time = time.time()
        reply_text = brain.chat(user_input)
        
        # 清理文本
        spoken_text = reply_text.split('*')[0].split('(')[0]
        if not spoken_text.strip(): spoken_text = "嗯..."
        
        print(f"👩 小爱: {reply_text}")

        # B. 配音 (秒回)
        # 这一步会自动播放声音
        voice.speak(spoken_text, output_file=AUDIO_TEMP)

        cost_time = time.time() - start_time
        print(f">> [耗时] 响应耗时: {cost_time:.2f}秒")
        
        # ❌ 不再调用 Wav2Lip，也不再弹窗视频

if __name__ == "__main__":
    main()