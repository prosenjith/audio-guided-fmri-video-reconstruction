import torch
import numpy as np
from diffusers import StableVideoDiffusionPipeline

def load_svd(device, cache_dir):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    ).to(device)

    pipe.enable_sequential_cpu_offload()
    return pipe


def generate_video(pipe, image, motion_features, num_frames, fps, device):
    motion_mean = motion_features.mean().item()
    motion_std = motion_features.std().item()
    seed = int(abs(motion_mean * 1e5 + motion_std * 1e4)) % (2**31)

    gen = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        image=image,
        num_frames=num_frames,
        decode_chunk_size=4,
        generator=gen
    )
    return out.frames[0]
