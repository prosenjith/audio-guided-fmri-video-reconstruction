import os, sys, yaml, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.audio.wav2vec2_model import load_wav2vec2
from src.pipeline.audio_processor import process_audio

CFG_PATH = os.path.join(ROOT, "configs", "audio_embedding.yaml")
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

IN_DIR = cfg["paths"]["input_audio_dir"]
OUT_DIR = cfg["paths"]["output_embed_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor, model = load_wav2vec2(cfg["model"]["name"], device)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
files = [os.path.join(IN_DIR, f) for f in sorted(os.listdir(IN_DIR))
         if os.path.splitext(f.lower())[1] in AUDIO_EXTS]

print(f"Found {len(files)} audio files.")

for p in files:
    process_audio(p, OUT_DIR, processor, model, cfg, device)

print("✅ Audio embedding pipeline completed.")