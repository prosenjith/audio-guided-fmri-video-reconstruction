import os, sys, yaml, torch, json
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.fusion.dataset import FMRI_AudioFusionDataset
from src.fusion.model import CrossAttentionFusion

# --------------------------------------------------
# Load config
# --------------------------------------------------
with open(os.path.join(ROOT, "configs", "fusion.yaml")) as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Dataset / Loader
# --------------------------------------------------
dataset = FMRI_AudioFusionDataset(
    cfg["paths"]["csv"],
    cfg["paths"]["fmri_root"],
    cfg["paths"]["audio_root"],
    cfg["sequence_length"]
)

loader = DataLoader(
    dataset,
    batch_size=cfg["batch_size"],
    shuffle=False
)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = CrossAttentionFusion(**cfg["model"]).to(device).eval()

out_root = cfg["paths"]["output_root"]
os.makedirs(out_root, exist_ok=True)

# --------------------------------------------------
# Fusion loop (FIXED)
# --------------------------------------------------
with torch.no_grad():
    for fmri, audio, subs, segs in tqdm(loader, desc="Fusing"):
        fmri, audio = fmri.to(device), audio.to(device)
        fused, _ = model(fmri, audio)

        for i in range(len(subs)):
            out_dir = os.path.join(out_root, subs[i])
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(
                out_dir,
                f"{segs[i]}_fused_embeddings.pt"
            )

            # ✅ SKIP if already exists
            if os.path.exists(out_path):
                print(f"⏩ Skipping {subs[i]}/{segs[i]} (already fused)")
                continue

            torch.save(fused[i].cpu(), out_path)

            # metadata (same as Colab)
            meta = {
                "subject": subs[i],
                "segment": segs[i],
                "n_tr": int(fused[i].shape[0]),
                "embedding_dim": int(fused[i].shape[1]),
                "model": "CrossAttentionFusion",
                "status": "success"
            }

            with open(out_path.replace("_embeddings.pt", "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"✅ Saved {subs[i]}/{segs[i]}")

print("✅ Fusion pipeline completed.")