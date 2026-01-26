import os
import pandas as pd
from tqdm import tqdm
from .io import read_video_frames
from .metrics import compute_metrics

def evaluate(cfg, transform, lpips_fn, device):
    records = []

    for subj in cfg["subjects"]:
        subj_dir = os.path.join(cfg["paths"]["gen_dir"], subj)
        if not os.path.exists(subj_dir):
            continue

        for seg in tqdm(cfg["test_segments"], desc=f"Evaluating {subj}"):
            gt_path = os.path.join(cfg["paths"]["gt_dir"], f"{seg}.mp4")
            if not os.path.exists(gt_path):
                continue

            gt_frames = read_video_frames(gt_path, transform)

            for mode in ["fmri_only", "fusion"]:
                gen_path = os.path.join(subj_dir, f"{seg}_{mode.replace('_only','')}_recon.mp4")
                if not os.path.exists(gen_path):
                    continue

                gen_frames = read_video_frames(gen_path, transform)
                s,p,l = compute_metrics(gt_frames, gen_frames, lpips_fn, device)

                records.append([subj, seg, mode, s, p, l])

    return pd.DataFrame(records, columns=["subject","segment","mode","SSIM","PSNR","LPIPS"])
