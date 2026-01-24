import argparse
import os
import json
import torch
import numpy as np
import subprocess
import pyloudnorm as pyln
import librosa
from pathlib import Path
from copy import deepcopy
from einops import rearrange

from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model # 自定义Wav2Vec2模型

BASE_DIR = Path(__file__).resolve().parent

class AudioProcessor:
    def __init__(self, wav2vec, device='cpu'):
        self.wav2vec = wav2vec
        self.device = device
        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()  # 冻结特征提取器的参数
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)  # 官方 Hugging Face 特征提取器
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

    def get_embedding(self, speech_array, video_length_frames=81): # 固定视频帧数
        audio_duration = video_length_frames / 25  # 假设音频时长 3.24s
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=self.sr).input_values
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

class DatasetPreprocessor:
    def __init__(self, audioemb_output, wav2vec, device="cpu"):
        self.audio_processor = AudioProcessor(wav2vec, device)
        self.audioemb_output = Path(audioemb_output) # data/audio/audio_embeddings
        self.audioemb_output.mkdir(parents=True, exist_ok=True)

    def process_audio(self, audio_path):
        audio_path = Path(audio_path)
        audio_name = audio_path.stem
        speech_array = self.audio_processor.load_audio(audio_path)
        audio_emb = self.audio_processor.get_embedding(speech_array)
        emb_path = self.audioemb_output / f"{audio_name}.pt"
        torch.save(audio_emb, emb_path)
        return str(emb_path)

    def load_original_json(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        return original_data

    def save_updated_json(self, updated_data, input_json):
        input_json = Path(input_json)
        new_path = input_json.with_name(f"{input_json.stem}_updated{input_json.suffix}")
        with open(new_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=4)
        return str(new_path)

    def create_traindata(self, input_json):
        original_data = self.load_original_json(input_json)
        updated_data = deepcopy(original_data)
        for item in updated_data:
            for person_key, audio_path in item['cond_audio'].items():
                emb_path = self.process_audio(audio_path)
                item['cond_audio'][person_key] = emb_path
        updated_json_path = self.save_updated_json(updated_data, input_json)
        print("Finish creating audio traindata")
        return str(updated_json_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav2vec", type=str, required=True, help="The path to the wav2vec checkpoint directory.")
    parser.add_argument("--audioemb_output", type=str, required=True, help="The path to save the audio embedding.")
    parser.add_argument("--input_json", type=str, required=True, help="The original json path.")
    args = parser.parse_args()
    preprocessor = DatasetPreprocessor(
        audioemb_output=args.audioemb_output,
        wav2vec=args.wav2vec,
        device="cpu"
    )
    updated_json = preprocessor.create_traindata(args.input_json)
    print(f"Updated json saved to: {updated_json}")

if __name__ == "__main__":
    main()
# python processor_audio.py --wav2vec weights/chinese-wav2vec2-base --audioemb_output data/audio_embeddings --input_json data/train.json


