import torch
import numpy as np

def align_w2v2_to_TR(meta, W, win_sec, hop_sec, TR=2.0):
    T = int(meta["n_tr"])
    N, D = W.shape
    out = torch.zeros(T, D)

    r = TR / hop_sec
    r_int = int(round(r))

    if abs(r - r_int) < 1e-6 and r_int >= 1:
        for t in range(T):
            s, e = t * r_int, (t + 1) * r_int
            if s >= N: break
            out[t] = W[s:min(e, N)].mean(dim=0)
        return out

    centers = (torch.arange(N) * hop_sec + win_sec / 2).numpy()
    tr_centers = (np.arange(T) * TR + TR / 2)
    idx = np.clip(np.searchsorted(centers, tr_centers), 0, N - 1)
    return W[idx]