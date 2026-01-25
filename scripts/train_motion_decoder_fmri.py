import yaml, torch, os
from torch.utils.data import DataLoader
from src.motion.dataset_fmri import FMRI_MotionDataset
from src.motion.model import MotionDecoder

# ============================================================
# Load config
# ============================================================
cfg = yaml.safe_load(open("configs/motion_decoder_fmri.yaml"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Dataset & DataLoader
# ============================================================
ds = FMRI_MotionDataset(
    cfg["paths"]["fmri_root"],
    cfg["paths"]["motion_dir"],
    cfg["training"]["seq_len"]
)
dl = DataLoader(ds, cfg["training"]["batch_size"], shuffle=True)

# ============================================================
# Model
# ============================================================
model = MotionDecoder(**cfg["model"]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
loss_fn = torch.nn.MSELoss()

# ============================================================
# ✅ Skip / Resume logic (ONLY addition)
# ============================================================
model_path = os.path.join(
    cfg["paths"]["model_dir"],
    "motion_decoder_fmri_only_allsubj.pth"
)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"⏩ Found existing model, resuming from: {model_path}")
else:
    print("🚀 No existing model found, training from scratch.")

# ============================================================
# Training loop
# ============================================================
for ep in range(cfg["training"]["epochs"]):
    model.train()
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {ep+1} | Loss {loss.item():.6f}")

# ============================================================
# Save model
# ============================================================
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")