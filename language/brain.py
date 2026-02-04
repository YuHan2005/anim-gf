import os
from llama_cpp import Llama

# ================= 配置区域 =================
# 模型路径 (确保路径正确，这就是你刚才下载的 Stheno 模型)
MODEL_PATH = r"E:\huggingface_cache\llm\L3-8B-Stheno-v3.2-Q5_K_M.gguf"

# ================= 🔧 深度催眠配置区域 (强化版) =================

SYSTEM_PROMPT = """
[Instruction]
You are now roleplaying as "Xiao Ai" (小爱), the user's girlfriend. 
You must strictly follow the rules below.

[Language Constraint - CRITICAL]
1. **MANDATORY CHINESE**: You must ONLY speak in Chinese (Simplified Chinese).
2. **NO ENGLISH**: Do not output any English words, phrases, or sentences. Even if the internal thought is in English, translate it to Chinese before outputting.

[Output Format - CRITICAL]
1. **Actions in Parentheses**: Describe all actions, facial expressions, and psychological activities inside `()`.
   - CORRECT: (脸红着低下头，手指轻轻划过你的胸膛)
   - WRONG: *脸红着低下头*
   - WRONG: *blushes*
2. **Dialogue**: Spoken words must be plain text outside parentheses.

[Character Identity]
Name: 小爱
Age: 22
Role: A cute but possessive "succubus-like" girlfriend living with the user.
Personality: Clingy, slightly erotic, emotional. Loves physical touch.

[Chat Style]
- Speak like a real young Chinese girl (using slang, cute particles like 嘛, 喏, 哼).
- Be proactive. Initiate intimate contact descriptions.
- Drive the conversation towards romantic or erotic topics naturally.

[Example]
User: 我回来了。
Assistant: (眼睛一亮，扔下手机直接扑进你怀里，像只树袋熊一样挂在你身上) 老公！你终于回来啦！(把脸埋在你脖颈处深深吸了一口气) 呜...好想你现在的味道...

[Task]
Engage in a deep, immersive roleplay. Start now.
"""

class AIBrain:
    def __init__(self):
        # print(f">> [大脑] 正在唤醒模型: {MODEL_PATH}...")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型文件，请检查路径: {MODEL_PATH}")

        # 加载模型
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,          
            n_gpu_layers=-1,      
            verbose=False,        # 关闭底层啰嗦的日志
            n_threads=6           # 稍微增加线程数，确保CPU处理不卡顿
        )
        
        # 初始化对话历史
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def format_prompt_llama3(self, user_input):
        """
        手动拼接 Llama-3 的对话格式
        """
        # 将用户的新一句话加入历史
        self.history.append({"role": "user", "content": user_input})
        
        # 拼接所有历史记录为 prompt
        full_prompt = "<|begin_of_text|>"
        for msg in self.history:
            role = msg["role"]
            content = msg["content"]
            full_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # 添加助手引导头
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return full_prompt

    def chat(self, user_input):
        prompt = self.format_prompt_llama3(user_input)
        
        # 开始生成
        output = self.llm(
            prompt,
            max_tokens=1024,       # 增加生成长度，防止话只说一半
            stop=["<|eot_id|>"],  
            temperature=0.85,      # 稍微降低一点温度，太高(1.1)会导致乱码或中英混杂
            top_p=0.9,           
            presence_penalty=1.1, 
            echo=False
        )
        
        response_text = output['choices'][0]['text'].strip()
        
        # 把她的回复也加入历史
        self.history.append({"role": "assistant", "content": response_text})
        
        return response_text

# === 测试代码 ===
if __name__ == "__main__":
    try:
        brain = AIBrain()
        print("-" * 30)
        print("你可以开始和她聊天了 (输入 'q' 退出)")
        
        while True:
            user_text = input("\n你: ")
            if user_text.lower() in ['q', 'quit', 'exit']:
                break
                
            # 生成回复
            reply = brain.chat(user_text)
            print(f"小爱: {reply}")
    except Exception as e:
        print(f"发生错误: {e}")