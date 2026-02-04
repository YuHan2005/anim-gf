import streamlit as st
import base64
import os
import time
import re  # 正则库
from language.brain import AIBrain
from speech.voice import AIVoice

# ================= 🔧 配置路径 =================
OUTPUT_DIR = "output_chat"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TEMPLATE_VIDEO = os.path.join(OUTPUT_DIR, "template_idle.mp4")
AVATAR_PATH = os.path.join(OUTPUT_DIR, "avatar_base.png")

# 用户头像
USER_AVATAR_URL = "https://api.dicebear.com/7.x/adventurer/svg?seed=Felix"

# ================= 🎨 页面设置 =================
st.set_page_config(
    page_title="我的女友: 小爱",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 🛠️ 工具函数 =================
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def clean_text_for_speech(text):
    """
    清洗文本：去除 *动作描写* 和 (心理活动)，只保留想说的话
    """
    # 1. 去除 *...* 之间的内容 (兼容旧格式)
    text = re.sub(r'\*.*?\*', '', text)
    # 2. 去除 (...) 英文括号
    text = re.sub(r'\(.*?\)', '', text)
    # 3. 去除 （...） 中文括号 (防止AI偶尔不听话)
    text = re.sub(r'（.*?）', '', text)
    
    # 4. 去除多余空格
    return text.strip()

# 预加载头像
if os.path.exists(AVATAR_PATH):
    img_b64 = get_img_as_base64(AVATAR_PATH)
    AI_AVATAR_HTML = f"data:image/png;base64,{img_b64}"
else:
    AI_AVATAR_HTML = "https://api.dicebear.com/7.x/avataaars/svg?seed=Coco"

# ================= 💄 CSS 样式 =================
st.markdown("""
<style>
    .stApp { background-color: #f5f5f5; }
    .chat-row { display: flex; align-items: flex-start; margin-bottom: 20px; width: 100%; }
    .avatar { width: 50px; height: 50px; border-radius: 6px; object-fit: cover; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
    .bubble { padding: 10px 14px; border-radius: 6px; position: relative; max-width: 70%; word-wrap: break-word; font-size: 16px; line-height: 1.6; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    .row-ai { justify-content: flex-start; }
    .bubble-ai { background-color: #ffffff; color: #000; margin-left: 12px; border: 1px solid #ededed; }
    .bubble-ai::before { content: ""; position: absolute; left: -6px; top: 16px; width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 6px solid #ffffff; }
    .row-user { justify-content: flex-end; }
    .bubble-user { background-color: #95ec69; color: #000; margin-right: 12px; }
    .bubble-user::before { content: ""; position: absolute; right: -6px; top: 16px; width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 6px solid #95ec69; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= 🧠 核心加载 =================
@st.cache_resource
def load_brain():
    return AIBrain()

@st.cache_resource
def load_voice():
    return AIVoice()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 📱 侧边栏 =================
with st.sidebar:
    st.title("💖 你的女友")
    if os.path.exists(TEMPLATE_VIDEO):
        st.video(TEMPLATE_VIDEO, autoplay=True, loop=True, muted=True)
    elif os.path.exists(AVATAR_PATH):
        st.image(AVATAR_PATH)
    st.markdown("---")
    if st.button("🗑️ 清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ================= 💬 聊天界面 =================
st.header("💬 甜蜜对话")
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""<div class="chat-row row-user"><div class="bubble bubble-user">{msg["content"]}</div><img class="avatar" src="{USER_AVATAR_URL}"></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="chat-row row-ai"><img class="avatar" src="{AI_AVATAR_HTML}"><div class="bubble bubble-ai">{msg["content"]}</div></div>""", unsafe_allow_html=True)
            if "audio" in msg and msg["audio"]:
                 st.audio(msg["audio"], format="audio/wav")

# ================= ⚡ 交互逻辑 =================
if user_input := st.chat_input("想对她说点什么..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with chat_container:
        message_placeholder = st.empty()
        # 先显示思考中
        message_placeholder.markdown(f"""<div class="chat-row row-ai"><img class="avatar" src="{AI_AVATAR_HTML}"><div class="bubble bubble-ai" style="color:gray;"><i>(正在思考...)</i></div></div>""", unsafe_allow_html=True)
        
        # 1. 生成文字
        brain = load_brain()
        full_response = brain.chat(st.session_state.messages[-1]["content"])
        
        # 立即显示文字结果
        message_placeholder.markdown(f"""<div class="chat-row row-ai"><img class="avatar" src="{AI_AVATAR_HTML}"><div class="bubble bubble-ai">{full_response}</div></div>""", unsafe_allow_html=True)
        
        # 2. 处理语音
        voice = load_voice()
        
        # 清洗动作描述 (支持中英文括号)
        spoken_text = clean_text_for_speech(full_response)
        
        if not spoken_text: 
            spoken_text = "嗯~" 
            
        # 生成动态文件名
        # 1. 生成文件名时，直接用 .wav
        temp_filename = os.path.join(OUTPUT_DIR, f"audio_{int(time.time())}.wav")
        
        # 2. 调用生成，并接收返回的“真实路径” (voice.py 可能会修正路径，所以要接住返回值)
        real_filepath = voice.speak(spoken_text, output_file=temp_filename)
        
        # 3. 读取音频 (读取 voice.speak 返回的那个真实路径)
        audio_data = None
        # 只有当文件路径存在，且文件真的在硬盘上时才读取
        if real_filepath and os.path.exists(real_filepath):
            with open(real_filepath, "rb") as f:
                audio_data = f.read()
            try:
                os.remove(real_filepath) # 读完即焚
            except:
                pass
        else:
            print(f"!! [调试] 网页端没找到音频文件: {temp_filename}")
        
        # 4. 保存到历史
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "audio": audio_data 
        })
        
        message_placeholder.empty()
        st.rerun()