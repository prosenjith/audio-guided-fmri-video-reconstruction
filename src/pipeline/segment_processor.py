import os
import torch
from src.io.nifti_loader import load_fmri_from_nifti
from src.fmri.pca_embedding import compute_embeddings
from src.fmri.merge import merge_runs


def process_segment(
    seg_dir: str,
    subject_name: str,
    output_root: str,
    use_mni: bool,
    normalize_per_run: bool,
    merge: bool,
    n_components: int,
    batch_size: int
):
    seg_name = os.path.basename(seg_dir)
    out_dir = os.path.join(output_root, subject_name, seg_name)
    os.makedirs(out_dir, exist_ok=True)

    data_dir = os.path.join(seg_dir, "mni" if use_mni else "raw")
    nii_files = [f for f in os.listdir(data_dir) if f.endswith(".nii.gz")]

    emb_paths = []

    for nii in nii_files:
        voxels, T = load_fmri_from_nifti(
            os.path.join(data_dir, nii),
            normalize_per_run
        )

        Z, ev = compute_embeddings(
            voxels,
            n_components,
            batch_size
        )

        out_path = os.path.join(out_dir, f"{nii}_embeddings.pt")
        torch.save(Z, out_path)
        emb_paths.append(out_path)

        print(f"Saved {out_path} | TR={T} | EV={ev:.3f}")

    if merge and len(emb_paths) > 1:
        avg = merge_runs(emb_paths)
        torch.save(
            avg,
            os.path.join(out_dir, f"{seg_name}_avg_embeddings.pt")
        )