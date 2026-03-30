# Voice to Text - Whisper-based Speech-to-Text Project

[中文](README.md) | **English**

A local speech-to-text application based on Whisper Large-V3-Turbo, optimized for Apple Silicon, with both Web UI and command-line interfaces.

## ✨ Features

- 🎯 **High Accuracy** - Powered by Whisper Large-V3-Turbo model
- 🚀 **Apple Silicon Optimized** - MPS acceleration for significant speed boost
- 🌐 **Web Interface** - Clean and intuitive web UI
- 💻 **CLI Support** - Batch processing via command line
- 🔤 **Smart Punctuation** - Integrated CT-Transformer punctuation restoration
- 📝 **Multiple Output Formats** - Generates both TXT and Markdown files
- 🔄 **Traditional/Simplified Conversion** - Auto-converts Traditional Chinese to Simplified

## 📋 Requirements

- Python 3.9+
- Recommended: Apple Silicon Mac (M1/M2/M3/M4) or NVIDIA GPU

## 🚀 Quick Start

### 1. Clone the Project

```bash
git clone https://github.com/Theo-Stats/voice-to-text.git
cd voice-to-text
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install torch transformers librosa flask flask-cors opencc onnxruntime huggingface_hub
```

### 4. Download Models

First run requires downloading the Whisper model (~1.5GB):

```bash
# Use mirror for faster download (optional)
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import WhisperForConditionalGeneration; WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3-turbo')"
```

### 5. Download Punctuation Model

```bash
cd CT-Transformer-punctuation
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('iic/speech_cttransformer_cn_punc', 'punc.onnx', local_dir='cttpunctuator/src/onnx')"
cd ..
```

## 📖 Usage

### Option 1: Web Interface

```bash
source venv/bin/activate
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

### Option 2: Command Line

1. Place audio files in the `input` folder
2. Run:

```bash
source venv/bin/activate
python transcribe.py
```

3. Check results in the `output` folder

## 🎬 Example

### Input Audio

An 11-second Chinese speech test file containing tongue twisters and poetry:

> 🎵 [sample_audio.mp3](input/示例音频.mp3)

### Transcription Result

```
吃葡萄不吐葡萄皮儿，不吃葡萄到吐葡萄皮儿。床前明月光，一阵美人香，不知春梦里，网字硬邦邦。
```

**Processing Time:** 6.0 seconds (Audio duration: 11 seconds)

**Performance:** RTF (Real-Time Factor) = 0.55, meaning transcription is ~1.8x faster than audio duration

### Output Files

| Format | File | Description |
|--------|------|-------------|
| TXT | [示例音频.txt](output/示例音频.txt) | Plain text with segment information |
| Markdown | [示例音频.md](output/示例音频.md) | Markdown format for documentation |

## 📁 Project Structure

```
voice-to-text/
├── app.py                    # Flask Web application
├── transcribe.py             # Command-line script
├── index.html                # Web frontend
├── README.md                 # Documentation (Chinese)
├── README_EN.md              # Documentation (English)
├── 技术文档.md                # Technical documentation
├── LICENSE                   # MIT License
│
├── input/                    # Input folder
│   └── 示例音频.mp3           # Sample audio file
│
├── output/                   # Output folder
│   ├── 示例音频.txt           # TXT output
│   └── 示例音频.md            # Markdown output
│
└── CT-Transformer-punctuation/  # Punctuation restoration module
```

## 🔧 Supported Audio Formats

- M4A
- MP3
- WAV
- WebM
- MP4
- MPEG
- MPGA

## ⚙️ Configuration

### Chunk Duration

Default is 30 seconds. Modify `CHUNK_LENGTH_S` in the code:

```python
CHUNK_LENGTH_S = 30  # Adjust between 10-60 seconds
```

### Language Settings

Default is Chinese. For other languages:

```python
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
# Change "zh" to "en" (English), "ja" (Japanese), etc.
```

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [FunASR CT-Transformer](https://github.com/alibaba-damo-academy/FunASR) - Punctuation restoration model
- [Hugging Face Transformers](https://huggingface.co/) - Model framework

## 📄 License

This project is open-sourced under the [MIT](LICENSE) license.

## 🤝 Contributing

Issues and Pull Requests are welcome!

---

If this project helps you, please consider giving it a ⭐ Star!