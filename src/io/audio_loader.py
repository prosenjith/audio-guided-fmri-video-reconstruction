import torchaudio
import torch

def load_audio_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.mean(dim=0)  # mono (T,)