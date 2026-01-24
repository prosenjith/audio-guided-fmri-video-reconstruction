import torch

@torch.no_grad()
def extract_frames_chunked(
    wav,
    processor,
    model,
    frame_hz: float,
    chunk_sec: float,
    overlap_sec: float,
    device: str
):
    sr = 16000
    n = wav.numel()
    chunk = int(chunk_sec * sr)
    step = int((chunk_sec - overlap_sec) * sr)
    drop = int(overlap_sec * frame_hz)

    frames = []
    first = True
    for s in range(0, n, step):
        e = min(n, s + chunk)
        seg = wav[s:e]
        if seg.numel() == 0:
            break

        inp = processor(seg.numpy(), sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp).last_hidden_state.squeeze(0).cpu()

        if not first and drop > 0 and out.size(0) > drop:
            out = out[drop:]

        frames.append(out)
        first = False
        if e >= n:
            break

    return torch.cat(frames, dim=0) if frames else torch.zeros(0, model.config.hidden_size)