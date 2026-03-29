#!/usr/bin/env python3
"""
语音转文字工具 - 基于 Whisper Large-V3-Turbo
支持批量处理 input 文件夹中的音频文件
"""

import torch
import time
import os
import sys
import glob
from datetime import timedelta
from pathlib import Path

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
CHUNK_LENGTH_S = 30

sys.path.insert(0, str(Path(__file__).parent / "CT-Transformer-punctuation"))

try:
    from opencc import OpenCC
    cc = OpenCC('t2s')
    def to_simplified(text):
        return cc.convert(text)
except ImportError:
    def to_simplified(text):
        return text

try:
    from cttPunctuator import CttPunctuator
    punctuator = CttPunctuator()
    print("✅ 标点恢复模型加载成功")
except Exception as e:
    print(f"⚠️ 标点恢复模型加载失败: {e}")
    punctuator = None

def check_device():
    if torch.backends.mps.is_available():
        print("✅ 检测到 Apple Silicon MPS 加速")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ 检测到 CUDA GPU 加速")
        return "cuda"
    else:
        print("⚠️ 使用 CPU 运行")
        return "cpu"

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_audio_files():
    extensions = ["*.m4a", "*.mp3", "*.wav", "*.webm", "*.mp4", "*.mpeg", "*.mpga"]
    files = []
    for ext in extensions:
        files.extend(INPUT_DIR.glob(ext))
    return sorted(files)

def transcribe_file(audio_path, model, processor, device):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
    
    audio_name = audio_path.stem
    print(f"\n{'='*60}")
    print(f"📁 处理文件: {audio_path.name}")
    print(f"{'='*60}")
    
    start_total = time.time()
    
    print("\n🎯 开始转录...")
    
    audio, sr = librosa.load(str(audio_path), sr=16000)
    
    chunk_samples = CHUNK_LENGTH_S * sr
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    
    all_text = []
    segments = []
    
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    
    for i in range(total_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, len(audio))
        chunk = audio[start_sample:end_sample]
        
        progress = (i + 1) / total_chunks * 100
        print(f"\r   转录进度: {progress:.1f}% ({i+1}/{total_chunks} 片段)", end="", flush=True)
        
        input_features = processor(
            chunk, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        if device != "cpu":
            input_features = input_features.half()
        
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )
        
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        if text.strip():
            all_text.append(text)
            segments.append({
                "start": start_sample / sr,
                "end": end_sample / sr,
                "text": text.strip()
            })
    
    print()
    
    transcribe_time = time.time() - start_total
    full_text = " ".join(all_text)
    
    full_text = to_simplified(full_text)
    
    if punctuator:
        try:
            full_text = punctuator.punctuate(full_text)[0]
            print("✅ 标点恢复完成")
        except Exception as e:
            print(f"⚠️ 标点恢复失败: {e}")
    
    def smart_format_text(text):
        import re
        
        paragraphs = []
        sentences = re.split(r'([。！？\n])', text)
        
        current_para = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ''
            
            if sentence:
                current_para.append(sentence + delimiter)
                
                if len(''.join(current_para)) > 100:
                    paragraphs.append(''.join(current_para))
                    current_para = []
        
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            current_para.append(sentences[-1].strip())
        
        if current_para:
            paragraphs.append(''.join(current_para))
        
        return '\n\n'.join(paragraphs)
    
    formatted_text = smart_format_text(full_text)
    
    base_name = audio_name
    txt_path = OUTPUT_DIR / f"{base_name}.txt"
    md_path = OUTPUT_DIR / f"{base_name}.md"
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("语音转文字结果 - Whisper Large-V3-Turbo\n")
        f.write(f"源文件: {audio_path.name}\n")
        f.write(f"转录耗时: {transcribe_time:.1f} 秒\n")
        f.write(f"音频时长: {len(audio)/sr:.1f} 秒\n")
        f.write("=" * 60 + "\n\n")
        f.write("【完整文本（带断句）】\n")
        f.write("-" * 60 + "\n")
        f.write(formatted_text)
        f.write("\n\n")
        f.write("【原始连续文本】\n")
        f.write("-" * 60 + "\n")
        f.write(full_text)
        f.write("\n\n")
        f.write("【分段文本（按时间）】\n")
        f.write("-" * 60 + "\n")
        for segment in segments:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"]
            f.write(f"[{start} - {end}] {text}\n")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {audio_name}\n\n")
        f.write(f"> 模型: Whisper Large-V3-Turbo | ")
        f.write(f"耗时: {transcribe_time:.1f}秒 | ")
        f.write(f"时长: {len(audio)/sr:.1f}秒\n\n")
        f.write("---\n\n")
        f.write("## 完整文本\n\n")
        f.write(formatted_text + "\n\n")
        f.write("---\n\n")
        f.write("## 分段文本\n\n")
        for segment in segments:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"]
            f.write(f"**[{start} - {end}]**\n\n{text}\n\n")
    
    print(f"\n✅ 转录完成! 耗时: {transcribe_time:.1f} 秒")
    print(f"📄 TXT: {txt_path.name}")
    print(f"📄 MD:  {md_path.name}")
    
    return full_text

def main():
    print("=" * 60)
    print("🎤 语音转文字工具 - Whisper Large-V3-Turbo")
    print("=" * 60)
    
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    audio_files = get_audio_files()
    
    if not audio_files:
        print(f"\n❌ input 文件夹中没有找到音频文件")
        print(f"   支持的格式: m4a, mp3, wav, webm, mp4, mpeg")
        print(f"   请将音频文件放入: {INPUT_DIR}")
        return
    
    print(f"\n📂 找到 {len(audio_files)} 个音频文件:")
    for f in audio_files:
        print(f"   - {f.name}")
    
    device = check_device()
    
    print("\n⏳ 正在加载 large-v3-turbo 模型...")
    start_load = time.time()
    
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    model_name = "openai/whisper-large-v3-turbo"
    
    processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        local_files_only=True
    ).to(device)
    
    load_time = time.time() - start_load
    print(f"✅ 模型加载完成! 耗时: {load_time:.1f} 秒")
    
    import librosa
    
    for audio_file in audio_files:
        transcribe_file(audio_file, model, processor, device)
    
    print("\n" + "=" * 60)
    print("🎉 所有文件转录完成!")
    print(f"📂 结果保存在: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()