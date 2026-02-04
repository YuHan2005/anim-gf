import os
import sys
import subprocess
import torch

# === 配置路径 ===
# 你的 Wav2Lip 文件夹名字
WAV2LIP_DIR = "Wav2Lip"
# 你的模型路径 (刚才整理好的)
CHECKPOINT_PATH = r"E:\huggingface_cache\lipsync\wav2lip_gan.pth"
FACE_DETECTOR_PATH = r"E:\huggingface_cache\lipsync\sfd_face.pth" 

class LipSyncEngine:
    def __init__(self):
        print(">> [嘴型] 初始化同步引擎...")
        
        # 检查环境
        if not os.path.exists(WAV2LIP_DIR):
            raise FileNotFoundError(f"❌ 找不到 Wav2Lip 文件夹！请确保它在项目根目录下。")
        
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"❌ 找不到模型文件: {CHECKPOINT_PATH}")

    def run(self, video_path, audio_path, output_path):
        print(f"\n>> [嘴型] 开始处理: {os.path.basename(video_path)} + {os.path.basename(audio_path)}")
        

        # === 【新增】自动修复 temp 文件夹缺失的问题 ===
        if not os.path.exists("temp"):
            os.makedirs("temp")
        # ===========================================


        # 构造 Wav2Lip 的推理命令
        # 我们直接调用 Wav2Lip 自带的 inference.py
        inference_script = os.path.join(WAV2LIP_DIR, "inference.py")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 组装命令行参数
        # --checkpoint_path: 模型路径
        # --face: 输入视频
        # --audio: 输入音频
        # --outfile: 输出视频
        # --resize_factor: 1 (保持原分辨率)
        # --face_det_batch_size: 人脸检测批次 (根据显存调整，8或16)
        
        # 组装命令行参数
        cmd = [
            sys.executable, inference_script,
            "--checkpoint_path", CHECKPOINT_PATH,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", "1",
            "--nosmooth"
        ]
        
        print(">> [系统] 正在调用 Wav2Lip 内核 (这可能需要几分钟)...")
        
        try:
            # 这一步会启动子进程运行 Wav2Lip
            # 我们需要把环境变量传进去，确保它能找到 CUDA
            env = os.environ.copy()
            
            # 【关键补丁】
            # Wav2Lip 默认会找 wav2lip/models 里的 face_detection
            # 我们通过环境变量或者临时文件操作来指引它，但最简单的办法是：
            # 确保 sfd_face.pth 被 Wav2Lip 自动找到，或者修改它的代码。
            # 为了简单，我们让 Wav2Lip 自己去默认位置找，如果找不到会报错。
            # 这里我们假设用户已经按照标准流程操作。
            
            subprocess.run(cmd, check=True, env=env)
            
            print(f">> [成功] 最终视频已生成: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"!! [错误] Wav2Lip 运行失败，错误代码: {e.returncode}")
            return None
        except Exception as e:
            print(f"!! [错误] 发生未知错误: {e}")
            return None

# === 测试代码 ===
if __name__ == "__main__":
    # 这里需要你有真实的视频和音频文件才能测试
    print("请使用 main.py 进行整合测试，或手动修改这里的路径进行测试。")
    pass