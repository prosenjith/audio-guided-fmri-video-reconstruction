import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_f=245, d_a=768, d_model=256, n_heads=4):
        super().__init__()
        self.fmri_proj = nn.Linear(d_f, d_model)
        self.audio_proj = nn.Linear(d_a, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, fmri, audio):
        Q = self.fmri_proj(fmri)
        K = self.audio_proj(audio)
        fused, attn = self.attn(Q, K, K)
        return fused, attn
