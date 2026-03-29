# Voice to Text - 基于 Whisper 的语音转文字项目

一个基于 Whisper Large-V3-Turbo 模型的本地语音转文字应用，专为 Apple Silicon 优化，提供 Web 界面和命令行两种使用方式。

## ✨ 特性

- 🎯 **高准确率** - 使用 Whisper Large-V3-Turbo 模型
- 🚀 **Apple Silicon 优化** - 支持 MPS 加速，速度提升显著
- 🌐 **Web 界面** - 简洁易用的网页操作界面
- 💻 **命令行支持** - 支持批量处理音频文件
- 🔤 **智能标点** - 集成 CT-Transformer 标点恢复模型
- 📝 **多格式输出** - 同时生成 TXT 和 Markdown 文件
- 🔄 **繁简转换** - 自动将繁体中文转换为简体

## 📋 环境要求

- Python 3.9+
- 推荐：Apple Silicon Mac (M1/M2/M3/M4) 或 NVIDIA GPU

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Theo-Stats/voice-to-text.git
cd voice-to-text
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install torch transformers librosa flask flask-cors opencc onnxruntime huggingface_hub
```

### 4. 下载模型

首次运行需要下载 Whisper 模型（约 1.5GB）：

```bash
# 使用国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3-turbo')"
```

### 5. 下载标点恢复模型

```bash
cd CT-Transformer-punctuation
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('iic/speech_cttransformer_cn_punc', 'punc.onnx', local_dir='cttpunctuator/src/onnx')"
cd ..
```

## 📖 使用方法

### 方式一：Web 界面

```bash
source venv/bin/activate
python app.py
```

然后在浏览器打开 http://127.0.0.1:5000

### 方式二：命令行

1. 将音频文件放入 `input` 文件夹
2. 运行：

```bash
source venv/bin/activate
python transcribe.py
```

3. 在 `output` 文件夹查看结果

## 🎬 示例

### 输入音频

一段 11 秒的中文语音测试文件，内容为绕口令和诗句：

> 🎵 [示例音频.mp3](input/示例音频.mp3)

### 转录结果

```
吃葡萄不吐葡萄皮儿，不吃葡萄到吐葡萄皮儿。床前明月光，一阵美人香，不知春梦里，网字硬邦邦。
```

**转录耗时：** 6.0 秒（音频时长 11 秒）

**性能：** 实时率 (RTF) = 0.55，即转录速度约为音频时长的 1.8 倍

### 输出文件

| 格式 | 文件 | 说明 |
|------|------|------|
| TXT | [示例音频.txt](output/示例音频.txt) | 纯文本格式，包含分段信息 |
| Markdown | [示例音频.md](output/示例音频.md) | Markdown 格式，适合文档嵌入 |

## 📁 项目结构

```
voice-to-text/
├── app.py                    # Flask Web 应用
├── transcribe.py             # 命令行脚本
├── index.html                # Web 前端界面
├── README.md                 # 使用说明
├── 技术文档.md                # 技术文档
├── LICENSE                   # MIT 开源协议
│
├── input/                    # 输入文件夹
│   └── 示例音频.mp3           # 示例音频文件
│
├── output/                   # 输出文件夹
│   ├── 示例音频.txt           # TXT 格式输出
│   └── 示例音频.md            # Markdown 格式输出
│
└── CT-Transformer-punctuation/  # 标点恢复模型
```

## 🔧 支持的音频格式

- M4A
- MP3
- WAV
- WebM
- MP4
- MPEG
- MPGA

## ⚙️ 配置说明

### 切片时长

默认 30 秒切片，可在代码中修改 `CHUNK_LENGTH_S` 变量：

```python
CHUNK_LENGTH_S = 30  # 可调整为 10-60 秒
```

### 语言设置

默认识别中文，如需识别其他语言，修改：

```python
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
# 将 "zh" 改为 "en" (英文)、"ja" (日文) 等
```

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别模型
- [FunASR CT-Transformer](https://github.com/alibaba-damo-academy/FunASR) - 标点恢复模型
- [Hugging Face Transformers](https://huggingface.co/) - 模型框架

## 📄 开源协议

本项目采用 [MIT](LICENSE) 协议开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

如果这个项目对你有帮助，欢迎 ⭐ Star 支持！