import argparse
import json
import torch
import numpy as np
import pyloudnorm as pyln
import librosa
from pathlib import Path
from copy import deepcopy
from einops import rearrange

from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model # 自定义Wav2Vec2模型

class AudioEmbeddingExtractor:
    def __init__(self, audio_encoder, feature_extractor, device='cpu'):
        self.audio_encoder = audio_encoder
        self.feature_extractor = feature_extractor
        self.device = device
        self.sr = 16000

    def loudness_norm(self, audio_array, sr=16000, lufs=-23):  # 音频信号响度标准化
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio  # 归一化音频数组

    def load_audio(self, audio_path):
        human_speech_array, sr = librosa.load(audio_path, sr=self.sr)  # 音频重采样
        human_speech_array = self.loudness_norm(human_speech_array, sr)  # 音频响度归一化
        return human_speech_array  # 返回音频数组

    def extract_embedding(self, speech_array, video_length_frames=81): # 固定视频帧数
        audio_duration = video_length_frames / 25  # 假设音频时长 3.24s
        audio_feature = np.squeeze(
            self.feature_extractor(speech_array, sampling_rate=self.sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)
        # audio encoder
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(audio_duration), output_hidden_states=True)  # 对特征序列进行线性插值，保证长度和视频帧数对齐
        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")  # [seq_len,num_layers,D]
        audio_emb = audio_emb.cpu().detach()
        return audio_emb

class AudioEmbeddingWriter:
    def __init__(self, extractor, output_dir):
        self.extractor = extractor
        self.output_dir = Path(output_dir) # data/audio/audio_embeddings
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_audio(self, audio_path):
        audio_path = Path(audio_path)
        audio_name = audio_path.stem
        speech = self.extractor.load_audio(audio_path)
        emb = self.extractor.extract_embedding(speech)
        out_path = self.output_dir / f"{audio_name}.pt"
        torch.save(emb, out_path)
        return str(out_path)

class AudioDatasetPreprocessor:
    def __init__(self, embedding_writer):
        self.embedding_writer = embedding_writer

    def create_traindata(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            for person, audio_path in item['cond_audio'].items():
                emb_path = self.embedding_writer.process_audio(audio_path)
                item['cond_audio'][person] = emb_path
        out_path = Path(input_json).with_name(f"{Path(input_json).stem}_updated.json")
        json.dump(data, open(out_path, "w"), indent=4, ensure_ascii=False)
        return str(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav2vec", type=str, required=True, help="The path to the wav2vec checkpoint directory.")
    parser.add_argument("--audioemb_output", type=str, required=True, help="The path to save the audio embedding.")
    parser.add_argument("--input_json", type=str, required=True, help="The original json path.")
    args = parser.parse_args()

    device = "cpu"
    audio_encoder = Wav2Vec2Model.from_pretrained(args.wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()  # 冻结特征提取器的参数
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec, local_files_only=True) # 官方 Hugging Face 特征提取器

    extractor = AudioEmbeddingExtractor(
        audio_encoder,
        feature_extractor,
        device
    )
    writer = AudioEmbeddingWriter(
        extractor,
        args.audioemb_output
    )

    dataset = AudioDatasetPreprocessor(writer)
    dataset.create_traindata(args.input_json)


if __name__ == "__main__":
    main()
# python processor_audio.py --wav2vec weights/chinese-wav2vec2-base --audioemb_output data/audio_embeddings --input_json data/train.json
