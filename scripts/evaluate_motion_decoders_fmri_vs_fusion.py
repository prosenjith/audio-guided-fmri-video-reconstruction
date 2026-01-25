import os
import sys
import yaml
import torch
import pandas as pd

# ============================================================
# Fix PYTHONPATH so `src` is importable
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.motion.model import MotionDecoder
from src.motion.evaluation_motion import evaluate_motion_decoders

# ============================================================
# Load config
# ============================================================
with open(os.path.join(ROOT_DIR, "configs", "eval_motion_fmri_vs_fusion.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

OUT_CSV = cfg["paths"]["output_csv"]

# ============================================================
# ✅ Skip if evaluation already exists
# ============================================================
if os.path.exists(OUT_CSV):
    print(f"⏩ Evaluation already exists: {OUT_CSV}")
    print("Skipping evaluation.\n")
    df = pd.read_csv(OUT_CSV)
    print(df)
    sys.exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Load models
# ============================================================
fmri_model = MotionDecoder(
    d_model=cfg["dims"]["fmri"],
    d_motion=2,
    n_layers=2,
    n_heads=cfg["model"]["fmri"]["n_heads"]
).to(device)

fusion_model = MotionDecoder(
    d_model=cfg["dims"]["fusion"],
    d_motion=2,
    n_layers=2,
    n_heads=cfg["model"]["fusion"]["n_heads"]
).to(device)

fmri_model.load_state_dict(
    torch.load(
        os.path.join(cfg["paths"]["model_root"],
                     "motion_decoder_fmri_only_allsubj.pth"),
        map_location=device
    )
)

fusion_model.load_state_dict(
    torch.load(
        os.path.join(cfg["paths"]["model_root"],
                     "motion_decoder_fusion_allsubj.pth"),
        map_location=device
    )
)

fmri_model.eval()
fusion_model.eval()

# ============================================================
# Run evaluation
# ============================================================
df = evaluate_motion_decoders(
    fmri_model,
    fusion_model,
    cfg["paths"]["fmri_root"],
    cfg["paths"]["fusion_root"],
    cfg["paths"]["motion_root"],
    cfg["test_segments"],
    cfg["tr_sec"],
    cfg["fps"],
    cfg["dims"]["fmri"],
    cfg["dims"]["fusion"],
    device
)

# ============================================================
# Save results
# ============================================================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(df)
print("✅ Evaluation complete.")