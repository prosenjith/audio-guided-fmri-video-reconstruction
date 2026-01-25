import torch.nn as nn

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
