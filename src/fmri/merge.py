import torch

def merge_runs(embedding_paths):
    tensors = [torch.load(p) for p in embedding_paths]
    min_len = min(t.shape[0] for t in tensors)
    stacked = torch.stack([t[:min_len] for t in tensors], dim=0)
    return stacked.mean(dim=0)