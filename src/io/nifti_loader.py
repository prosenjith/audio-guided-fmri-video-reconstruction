import nibabel as nib
import numpy as np

def load_fmri_from_nifti(
    nifti_path: str,
    normalize_per_run: bool = True
):
    img = nib.load(nifti_path)
    data = img.get_fdata()

    T = data.shape[-1]
    voxels = data.reshape(-1, T).T  # (TR, voxels)

    if normalize_per_run:
        voxels = (voxels - voxels.mean(0)) / (voxels.std(0) + 1e-6)

    return voxels.astype(np.float32), T