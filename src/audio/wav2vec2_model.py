import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def load_wav2vec2(model_name: str, device: str):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()
    return processor, model