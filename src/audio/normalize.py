def zscore(W):
    return (W - W.mean(0, keepdim=True)) / (W.std(0, keepdim=True) + 1e-6)