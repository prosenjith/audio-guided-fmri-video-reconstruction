import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA


def compute_embeddings(
    voxels: np.ndarray,
    n_components: int,
    batch_size: int
):
    max_components = min(n_components, voxels.shape[0])

    if max_components < n_components:
        print(
            f"Reducing PCA components {n_components} → {max_components} "
            f"(limited by TR={voxels.shape[0]})"
        )

    safe_batch = max(batch_size, max_components)
    pca = IncrementalPCA(n_components=max_components)

    for i in tqdm(
        range(0, len(voxels), safe_batch),
        desc="Fitting PCA",
        leave=False
    ):
        pca.partial_fit(voxels[i:i + safe_batch])

    Z = np.zeros((len(voxels), max_components), dtype=np.float32)

    for i in tqdm(
        range(0, len(voxels), safe_batch),
        desc="Transforming",
        leave=False
    ):
        Z[i:i + safe_batch] = pca.transform(voxels[i:i + safe_batch])

    explained_var = float(np.sum(pca.explained_variance_ratio_))

    return torch.from_numpy(Z), explained_var