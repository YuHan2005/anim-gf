# 🛠️ 环境准备

你需要两个独立的组件配合运行：

1. **主项目环境 (Conda)**
   * 环境名称：`anim-gf`
   * 主要依赖：`streamlit`, `llama-cpp-python`, `requests`

2. **语音后端 (GPT-SoVITS 整合包)**
   * 版本：`GPT-SoVITS-v2pro`
   * 路径示例：`E:\GPT-SoVITS-v2pro-20250604-nvidia50`

---

# 🚀 启动指南 (必读)

**由于显存 (RTX 3060 6GB) 有限，为了防止显存爆炸 (OOM)，我们需要采用 “CPU 推理语音，GPU 推理文字” 的策略。**

请务必严格按照以下顺序启动两个黑色窗口：

## 第一步：启动语音服务 (GPT-SoVITS)
**目标**：强制使用 CPU 运行语音服务，把显卡留给文字模型。

1. 进入 **GPT-SoVITS 整合包文件夹**。
2. 在地址栏输入 `cmd` 打开命令行。
3. 依次执行以下命令（**不要直接双击 go-webui.bat**，那是给 GPU 用的）：

```cmd
REM 1. 屏蔽显卡，强制使用 CPU
set CUDA_VISIBLE_DEVICES=-1

REM 2. 启动 API 服务 (注意使用 runtime 下的 python)
.\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880


✅ 成功标志：看到 Uvicorn running on http://127.0.0.1:9880。