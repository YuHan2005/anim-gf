import requests
import os
import json

# ================= ⚙️ 配置区域 =================
API_URL = "http://127.0.0.1:9880"

# 【请确认】这里必须是你电脑上真实存在的路径！
REF_AUDIO_PATH = r"E:\huggingface_cache\data\vo_BZLQ001_4_hutao_10.wav"
REF_TEXT = "不只是有，甚至还在失控边缘，一旦爆发，后果不堪设想"
REF_LANG = "zh"

class AIVoice:
    def __init__(self):
        print(f">> [声音] 初始化，目标API: {API_URL}")

    def speak(self, text, output_file="temp_voice.wav"):
        if not text: return None
        
        # 强制使用 WAV 格式，避免 400 错误
        payload = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": REF_AUDIO_PATH,
            "prompt_text": REF_TEXT,
            "prompt_lang": REF_LANG,
            "text_split_method": "cut5",
            "batch_size": 1,
            "media_type": "wav",    # <--- 修改了这里：改成 wav
            "streaming_mode": False
        }

        try:
            # print(">> [调试] 发送请求...")
            response = requests.post(f"{API_URL}/tts", json=payload)
            
            if response.status_code == 200:
                # 确保保存为 wav 后缀
                if output_file.endswith(".mp3"):
                    output_file = output_file.replace(".mp3", ".wav")
                    
                with open(output_file, "wb") as f:
                    f.write(response.content)
                return output_file
            else:
                print(f"!! [错误] 生成失败，状态码: {response.status_code}")
                print(f"!! [服务器返回]: {response.text}")
                return None
                
        except Exception as e:
            print(f"!! [系统错误]: {e}")
            return None

# === 本地测试代码 ===
if __name__ == "__main__":
    voice = AIVoice()
    print("正在测试生成...")
    # 测试生成 wav
    voice.speak("你好，我是胡桃，这就去把mp3改成wav。", "test_final.wav")
    print("测试结束，请去听一下 test_final.wav")