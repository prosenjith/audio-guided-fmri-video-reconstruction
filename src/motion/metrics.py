import torch
from scipy.stats import pearsonr

def mse(pred, gt):
    return torch.mean((pred - gt) ** 2).item()

def corr(pred, gt):
    return pearsonr(pred.cpu().numpy(), gt.cpu().numpy())[0]
