# CT-Transformer-punctuation

中文标点恢复模型，基于 FunASR 项目。

## 模型下载

首次使用需要下载 ONNX 模型文件：

```bash
cd CT-Transformer-punctuation
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('iic/speech_cttransformer_cn_punc', 'punc.onnx', local_dir='cttpunctuator/src/onnx')"
```

## 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里巴巴语音识别项目
- [CT-Transformer-punctuation](https://github.com/lovemefan/CT-Transformer-punctuation) - 标点恢复模型封装