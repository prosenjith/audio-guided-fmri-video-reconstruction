import os
import torch
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from src.motion.metrics import mse, corr


def evaluate_motion_decoders(
    fmri_model,
    fusion_model,
    fmri_root,
    fusion_root,
    motion_dir,
    test_segments,
    tr_sec,
    fps,
    target_d_fmri,
    target_d_fusion,
    device
):
    frames_per_tr = int(tr_sec * fps)
    results = []

    proj_fmri = torch.nn.Linear(target_d_fmri, target_d_fusion).to(device)
    proj_fmri.eval()

    for seg in tqdm(test_segments, desc="Evaluating test segments"):
        motion_path = os.path.join(motion_dir, f"{seg}_motion.pt")
        if not os.path.exists(motion_path):
            continue

        motion_gt = torch.load(motion_path).float().to(device)
        n_trs = motion_gt.shape[0] // frames_per_tr
        if n_trs == 0:
            continue

        motion_gt = motion_gt[: n_trs * frames_per_tr] \
                        .reshape(n_trs, frames_per_tr, -1) \
                        .mean(dim=1)

        fmri_preds, fusion_preds = [], []

        for subj in ["subject1", "subject2", "subject3"]:
            # ---- fMRI ----
            fmri_path = os.path.join(fmri_root, subj, seg, f"{seg}_avg_embeddings.pt")
            if os.path.exists(fmri_path):
                fmri = torch.load(fmri_path).float().to(device)
                fmri = fmri[:, :target_d_fmri]

                with torch.no_grad():
                    try:
                        pred = fmri_model(fmri.unsqueeze(0)).squeeze(0)
                    except RuntimeError:
                        pred = fmri_model(proj_fmri(fmri).unsqueeze(0)).squeeze(0)

                fmri_preds.append(pred[: motion_gt.shape[0]])

            # ---- Fusion ----
            fusion_path = os.path.join(fusion_root, subj, f"{seg}_fused_embeddings.pt")
            if os.path.exists(fusion_path):
                fusion = torch.load(fusion_path).float().to(device)
                fusion = fusion[:, :target_d_fusion]

                with torch.no_grad():
                    pred = fusion_model(fusion.unsqueeze(0)).squeeze(0)

                fusion_preds.append(pred[: motion_gt.shape[0]])

        if fmri_preds and fusion_preds:
            fmri_pred = torch.stack(fmri_preds).mean(0)
            fusion_pred = torch.stack(fusion_preds).mean(0)

            results.append({
                "segment": seg,
                "mse_fmri": mse(fmri_pred, motion_gt),
                "mse_fusion": mse(fusion_pred, motion_gt),
                "corr_fmri": corr(fmri_pred[:, 0], motion_gt[:, 0]),
                "corr_fusion": corr(fusion_pred[:, 0], motion_gt[:, 0]),
            })

    return pd.DataFrame(results)
