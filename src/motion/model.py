import torch
import torch.nn as nn
from collections import OrderedDict

class MotionDecoder(nn.Module):
    def __init__(self, d_model, d_motion, n_layers, n_heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_model, d_motion)

    def forward(self, x):
        return self.fc(self.encoder(x))


def load_encoder_only(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    new_state = OrderedDict()

    for k, v in state.items():
        if k.startswith("encoder."):
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    print(f"✅ Loaded encoder weights from {checkpoint_path}")
    if missing:
        print(f"ℹ️ Missing keys (expected): {missing}")
    if unexpected:
        print(f"ℹ️ Unexpected keys (ignored): {unexpected}")

