import requests
import os

# ================= ⚙️ 配置区域 =================
API_URL = "http://127.0.0.1:9880"

# 请确认这个路径和文件真实存在！
REF_AUDIO_PATH = r"E:\huggingface_cache\data\vo_BZLQ001_4_hutao_10.wav"
REF_TEXT = "不只是有，甚至还在失控边缘，一旦爆发，后果不堪设想"
REF_LANG = "zh"

def debug_speak():
    print("-" * 30)
    print(f">> [诊断] 正在检查 API 状态: {API_URL}")
    
    # 1. 先检查 API 活没活着
    try:
        resp = requests.get(f"{API_URL}/")
        print(f">> [状态] API 连接成功 (HTTP {resp.status_code})")
    except Exception as e:
        print(f"!! [致命错误] 无法连接到 API，请检查黑色窗口是否开着！\n错误信息: {e}")
        return

    # 2. 构造 100% 纯正的 V2 请求
    payload = {
        "text": "你好，这是一次测试。",
        "text_lang": "zh",
        "ref_audio_path": REF_AUDIO_PATH,  # 关键参数
        "prompt_text": REF_TEXT,
        "prompt_lang": REF_LANG,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "mp3",
        "streaming_mode": False
    }

    print(f">> [诊断] 正在发送 V2 请求...")
    print(f">> [参数检查] 参考音频路径: {REF_AUDIO_PATH}")
    
    if not os.path.exists(REF_AUDIO_PATH):
        print(f"!! [警告] Python脚本发现该路径下文件不存在！请检查路径是否写错？")
    
    try:
        response = requests.post(f"{API_URL}/tts", json=payload)
        
        # 3. 打印结果
        if response.status_code == 200:
            print(f">> [成功] 恭喜！声音生成成功！问题已解决。")
            with open("success.mp3", "wb") as f:
                f.write(response.content)
        else:
            print(f"!! [失败] 服务器返回状态码: {response.status_code}")
            print("-" * 10 + " 真实报错信息 " + "-" * 10)
            # 🔥 这里会打印出真正的病因 🔥
            print(response.text) 
            print("-" * 30)
            
            # 智能分析报错
            if "GPT model weights" in response.text or "SoVITS model weights" in response.text:
                print("💡 [分析] 原因：服务器重启后，没有加载模型！")
                print("👉 解决：你需要去 WebUI 或通过 API 加载模型 (ckpt 和 pth 文件)。")
            elif "not found" in response.text:
                 print("💡 [分析] 原因：服务器找不到参考音频文件。")

    except Exception as e:
        print(f"!! [请求异常]: {e}")

if __name__ == "__main__":
    debug_speak()