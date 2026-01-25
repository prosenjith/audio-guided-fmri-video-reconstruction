import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from tqdm import tqdm
from src.motion.optical_flow import compute_optical_flow

VIDEO_DIR = "/content/drive/MyDrive/Research/data/stimuli/videos"
OUT_DIR = "/content/drive/MyDrive/Research/data/generated/motion_targets"
os.makedirs(OUT_DIR, exist_ok=True)

video_ids = [f"seg{i}" for i in range(1,19)] + [f"test{i}" for i in range(1,6)]

for vid in tqdm(video_ids):
    out = os.path.join(OUT_DIR, f"{vid}_motion.pt")
    if os.path.exists(out):
        print(f"⏩ Skipping {vid} (already exists)")
        continue
    path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
    if not os.path.exists(path):
        continue
    flow = compute_optical_flow(path)
    if flow is not None:
        torch.save(flow, out)
