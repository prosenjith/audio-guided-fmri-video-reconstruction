import yaml, torch, os
from torch.utils.data import DataLoader
from src.motion.dataset_fusion import Fusion_MotionDataset
from src.motion.model import MotionDecoder

cfg = yaml.safe_load(open("configs/motion_decoder_fusion.yaml"))
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = Fusion_MotionDataset(
    cfg["paths"]["fusion_root"],
    cfg["paths"]["motion_dir"],
    cfg["training"]["seq_len"]
)
dl = DataLoader(ds, cfg["training"]["batch_size"], shuffle=True)

model = MotionDecoder(**cfg["model"]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
loss_fn = torch.nn.MSELoss()

ckpt = os.path.join(cfg["paths"]["model_dir"], "motion_decoder_fusion_allsubj.pth")
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt))
    print("🔄 Resuming training from saved checkpoint.")

for ep in range(cfg["training"]["epochs"]):
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {ep+1} | Loss {loss.item():.6f}")

torch.save(model.state_dict(), ckpt)
