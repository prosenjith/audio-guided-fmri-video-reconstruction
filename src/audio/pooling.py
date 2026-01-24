import torch

def pool_windows(H, frame_hz, win_sec, hop_sec):
    if H.numel() == 0:
        return H

    win = int(win_sec * frame_hz)
    hop = int(hop_sec * frame_hz)

    pooled = []
    for s in range(0, H.size(0), hop):
        seg = H[s:s+win]
        if seg.numel() == 0:
            break
        pooled.append(seg.mean(dim=0, keepdim=True))
        if s + win >= H.size(0):
            break

    return torch.cat(pooled, dim=0)