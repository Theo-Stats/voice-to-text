#!/usr/bin/env python3
"""
语音转文字 Web 应用
启动方式: python app.py
访问地址: http://127.0.0.1:5000
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "CT-Transformer-punctuation"))

import torch
import time
import threading
from datetime import timedelta
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='.')
CORS(app)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CHUNK_LENGTH_S = 30

ALLOWED_EXTENSIONS = {'m4a', 'mp3', 'wav', 'webm', 'mp4', 'mpeg', 'mpga'}

model = None
processor = None
device = None
punctuator = None
transcribe_status = {
    "is_running": False,
    "current_file": None,
    "progress": 0,
    "total_files": 0,
    "completed_files": 0,
    "message": ""
}

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
except Exception as e:
    print(f"Warning: Could not load punctuator: {e}")
    punctuator = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_device():
    global device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def load_model():
    global model, processor, device
    if model is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        model_name = "openai/whisper-large-v3-turbo"
        device = check_device()
        processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            local_files_only=True
        ).to(device)
    return model, processor, device

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def transcribe_file_internal(audio_path, status_callback=None):
    import librosa
    
    model, processor, device = load_model()
    
    audio_name = audio_path.stem
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
        
        if status_callback:
            progress = (i + 1) / total_chunks * 100
            status_callback(progress, i + 1, total_chunks)
        
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
    
    full_text = " ".join(all_text)
    
    full_text = to_simplified(full_text)
    
    if punctuator:
        try:
            full_text = punctuator.punctuate(full_text)[0]
        except Exception as e:
            print(f"Punctuation restoration failed: {e}")
    
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
    
    txt_path = OUTPUT_DIR / f"{audio_name}.txt"
    md_path = OUTPUT_DIR / f"{audio_name}.md"
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("语音转文字结果 - Whisper Large-V3-Turbo\n")
        f.write(f"源文件: {audio_path.name}\n")
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
            f.write(f"[{start} - {end}] {segment['text']}\n")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {audio_name}\n\n")
        f.write(f"> 模型: Whisper Large-V3-Turbo | 时长: {len(audio)/sr:.1f}秒\n\n")
        f.write("---\n\n")
        f.write("## 完整文本\n\n")
        f.write(formatted_text + "\n\n")
        f.write("---\n\n")
        f.write("## 分段文本\n\n")
        for segment in segments:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            f.write(f"**[{start} - {end}]**\n\n{segment['text']}\n\n")
    
    return full_text, txt_path, md_path

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(transcribe_status)

@app.route('/api/files', methods=['GET'])
def list_files():
    input_files = []
    for ext in ALLOWED_EXTENSIONS:
        for f in INPUT_DIR.glob(f"*.{ext}"):
            input_files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(f.stat().st_mtime))
            })
    
    output_files = []
    for f in OUTPUT_DIR.glob("*.txt"):
        md_exists = (OUTPUT_DIR / f"{f.stem}.md").exists()
        output_files.append({
            "name": f.stem,
            "txt": f.name,
            "md": f"{f.stem}.md" if md_exists else None
        })
    
    return jsonify({
        "input_files": sorted(input_files, key=lambda x: x['modified'], reverse=True),
        "output_files": sorted(output_files, key=lambda x: x['name'])
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件格式，支持: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    filename = file.filename
    if not filename or filename.strip() == '':
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename) or f"audio_{int(time.time())}.{file.filename.rsplit('.', 1)[1]}"
    
    file.save(INPUT_DIR / filename)
    
    return jsonify({"message": "上传成功", "filename": filename})

@app.route('/api/transcribe', methods=['POST'])
def start_transcribe():
    global transcribe_status
    
    if transcribe_status["is_running"]:
        return jsonify({"error": "已有转录任务在运行中"}), 400
    
    data = request.json or {}
    filename = data.get('filename')
    
    if filename:
        audio_path = INPUT_DIR / filename
        if not audio_path.exists():
            return jsonify({"error": "文件不存在"}), 404
        files_to_transcribe = [audio_path]
    else:
        files_to_transcribe = []
        for ext in ALLOWED_EXTENSIONS:
            files_to_transcribe.extend(INPUT_DIR.glob(f"*.{ext}"))
    
    if not files_to_transcribe:
        return jsonify({"error": "没有找到可转录的文件"}), 400
    
    def run_transcribe():
        global transcribe_status
        transcribe_status["is_running"] = True
        transcribe_status["total_files"] = len(files_to_transcribe)
        transcribe_status["completed_files"] = 0
        
        for audio_path in files_to_transcribe:
            transcribe_status["current_file"] = audio_path.name
            transcribe_status["message"] = f"正在转录: {audio_path.name}"
            
            def update_progress(progress, current, total):
                transcribe_status["progress"] = progress
                transcribe_status["message"] = f"转录 {audio_path.name}: {progress:.0f}% ({current}/{total})"
            
            try:
                transcribe_file_internal(audio_path, update_progress)
                transcribe_status["completed_files"] += 1
            except Exception as e:
                transcribe_status["message"] = f"错误: {str(e)}"
        
        transcribe_status["is_running"] = False
        transcribe_status["current_file"] = None
        transcribe_status["progress"] = 100
        transcribe_status["message"] = "转录完成!"
    
    thread = threading.Thread(target=run_transcribe)
    thread.start()
    
    return jsonify({"message": "转录已开始"})

@app.route('/api/download/<filename>')
def download_file(filename):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(file_path, as_attachment=True)

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    file_path = INPUT_DIR / filename
    if file_path.exists():
        file_path.unlink()
    return jsonify({"message": "删除成功"})

@app.route('/api/delete-output/<filename>', methods=['DELETE'])
def delete_output_file(filename):
    stem = filename.replace('.txt', '').replace('.md', '')
    txt_path = OUTPUT_DIR / f"{stem}.txt"
    md_path = OUTPUT_DIR / f"{stem}.md"
    if txt_path.exists():
        txt_path.unlink()
    if md_path.exists():
        md_path.unlink()
    return jsonify({"message": "删除成功"})

if __name__ == '__main__':
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 50)
    print("🎤 语音转文字 Web 应用")
    print("=" * 50)
    print(f"📁 输入目录: {INPUT_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"🌐 访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    print("\n⏳ 首次使用会自动加载模型，请稍候...")
    
    load_model()
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)