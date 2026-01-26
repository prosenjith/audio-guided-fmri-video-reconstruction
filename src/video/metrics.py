import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_metrics(gt, gen, lpips_fn, device):
    T = min(len(gt), len(gen))
    gt, gen = gt[:T], gen[:T]

    gt_np  = gt.permute(0,2,3,1).cpu().numpy()
    gen_np = gen.permute(0,2,3,1).cpu().numpy()

    ssim_vals, psnr_vals, lpips_vals = [], [], []

    for i in range(T):
        ssim_vals.append(ssim(gt_np[i], gen_np[i], channel_axis=-1, data_range=1.0))
        psnr_vals.append(psnr(gt_np[i], gen_np[i], data_range=1.0))

        with torch.no_grad():
            lpips_vals.append(
                lpips_fn(gt[i:i+1].to(device), gen[i:i+1].to(device)).mean().item()
            )

    return np.mean(ssim_vals), np.mean(psnr_vals), np.mean(lpips_vals)
