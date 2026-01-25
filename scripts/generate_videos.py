import os
import sys
import yaml
import torch
import imageio
from PIL import Image
from tqdm import tqdm

# ============================================================
# Project root for Colab / VS Code
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ============================================================
# Imports from codebase
# ============================================================
from src.motion.model import MotionDecoder, load_encoder_only
from src.video.generation import load_svd, generate_video

# ============================================================
# Load config
# ============================================================
with open(os.path.join(ROOT, "configs", "video_reconstruction.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Initialize Stage-4 MotionDecoders
# ============================================================
fmri_model = MotionDecoder(
    d_model=cfg["dims"]["fmri"],
    d_motion=cfg["dims"]["motion_latent"],
    n_layers=2,
    n_heads=5
)

fusion_model = MotionDecoder(
    d_model=cfg["dims"]["fusion"],
    d_motion=cfg["dims"]["motion_latent"],
    n_layers=2,
    n_heads=8
)

# ============================================================
# Load encoder weights only (Stage-3 → Stage-4)
# ============================================================
load_encoder_only(
    fmri_model,
    os.path.join(cfg["paths"]["model_dir"],
                 "motion_decoder_fmri_only_allsubj.pth")
)

load_encoder_only(
    fusion_model,
    os.path.join(cfg["paths"]["model_dir"],
                 "motion_decoder_fusion_allsubj.pth")
)

fmri_model.eval()
fusion_model.eval()

print("✅ MotionDecoders loaded (encoder-only transfer).")

# ============================================================
# Load Stable Video Diffusion
# ============================================================
pipe = load_svd(device, cfg["paths"]["svd_cache"])
print("✅ Stable Video Diffusion loaded.")

# ============================================================
# Stage-4 Video Generation
# ============================================================
for seg in tqdm(cfg["test_segments"], desc="🎞️ Generating videos"):
    video_path = os.path.join(cfg["paths"]["stimuli_dir"], f"{seg}.mp4")
    if not os.path.exists(video_path):
        print(f"⚠️ Missing {seg}.mp4 — skipping.")
        continue

    reader = imageio.get_reader(video_path, "ffmpeg")
    first_frame = Image.fromarray(reader.get_data(0))
    reader.close()

    for subj in cfg["subjects"]:
        out_dir = os.path.join(cfg["paths"]["output_dir"], subj)
        os.makedirs(out_dir, exist_ok=True)

        # ====================================================
        # 🧠 fMRI-only reconstruction
        # ====================================================
        out_fmri = os.path.join(out_dir, f"{seg}_fmri_recon.mp4")
        if not os.path.exists(out_fmri):
            fmri_path = os.path.join(
                cfg["paths"]["fmri_root"], subj, seg, f"{seg}_avg_embeddings.pt"
            )
            if os.path.exists(fmri_path):
                fmri = torch.load(fmri_path).float().to(device)
                with torch.no_grad():
                    motion = fmri_model(fmri.unsqueeze(0)).squeeze(0)

                frames = generate_video(
                    pipe,
                    first_frame,
                    motion,
                    cfg["video"]["num_frames"],
                    cfg["video"]["fps"],
                    device
                )

                imageio.mimsave(out_fmri, frames, fps=cfg["video"]["fps"])
                print(f"✅ Saved {out_fmri}")
        else:
            print(f"⏩ Skipping {subj}/{seg}_fmri_recon.mp4")

        # ====================================================
        # 🎧 Fusion reconstruction
        # ====================================================
        out_fusion = os.path.join(out_dir, f"{seg}_fusion_recon.mp4")
        if not os.path.exists(out_fusion):
            fusion_path = os.path.join(
                cfg["paths"]["fusion_root"], subj, f"{seg}_fused_embeddings.pt"
            )
            if os.path.exists(fusion_path):
                fusion = torch.load(fusion_path).float().to(device)
                with torch.no_grad():
                    motion = fusion_model(fusion.unsqueeze(0)).squeeze(0)

                frames = generate_video(
                    pipe,
                    first_frame,
                    motion,
                    cfg["video"]["num_frames"],
                    cfg["video"]["fps"],
                    device
                )

                imageio.mimsave(out_fusion, frames, fps=cfg["video"]["fps"])
                print(f"✅ Saved {out_fusion}")
        else:
            print(f"⏩ Skipping {subj}/{seg}_fusion_recon.mp4")

print("\n🎯 Stage-4 video reconstruction complete.")