import os, sys, yaml, torch
import pandas as pd
from torchvision import transforms
from lpips import LPIPS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.video.quantitative import evaluate
from src.video.plotting import plot_all

with open(os.path.join(ROOT, "configs", "eval_video_quantitave.yaml")) as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(cfg["video"]["resize"]),
    transforms.ToTensor()
])

lpips_fn = LPIPS(net="vgg").to(device)

csv_path = cfg["paths"]["output_csv"]
if os.path.exists(csv_path):
    print(f"⏩ Skipping evaluation — CSV exists: {csv_path}")
    df = pd.read_csv(csv_path)
else:
    df = evaluate(cfg, transform, lpips_fn, device)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved metrics to {csv_path}")

plot_all(df, cfg["paths"]["summary_dir"])
