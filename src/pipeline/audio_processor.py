import os, json, torch
from src.io.audio_loader import load_audio_16k
from src.audio.wav2vec2_frames import extract_frames_chunked
from src.audio.pooling import pool_windows
from src.audio.normalize import zscore

def process_audio(
    audio_path,
    out_dir,
    processor,
    model,
    cfg,
    device
):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    emb_path = os.path.join(out_dir, f"{base}_w2v2_{int(cfg['windowing']['win_sec'])}s.pt")
    meta_path = emb_path.replace(".pt", "_meta.json")

    if os.path.exists(emb_path) and os.path.exists(meta_path):
        print(f"⏩ Skip {base}")
        return

    wav = load_audio_16k(audio_path)

    H = extract_frames_chunked(
        wav,
        processor,
        model,
        cfg["model"]["frame_hz"],
        cfg["chunking"]["chunk_sec"],
        cfg["chunking"]["overlap_sec"],
        device
    )

    W = pool_windows(
        H,
        cfg["model"]["frame_hz"],
        cfg["windowing"]["win_sec"],
        cfg["windowing"]["hop_sec"]
    )

    if cfg["normalization"]["zscore"] and W.numel() > 0:
        W = zscore(W)

    torch.save(W.float(), emb_path)

    with open(meta_path, "w") as f:
        json.dump({
            "audio": audio_path,
            "model": cfg["model"]["name"],
            "n_windows": int(W.size(0)),
            "dims": int(W.size(1)),
        }, f, indent=2)

    print(f"Saved {base}: {tuple(W.shape)}")