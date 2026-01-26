import imageio
import torch
from PIL import Image
from torchvision import transforms

def read_video_frames(path, transform, max_frames=None):
    reader = imageio.get_reader(path, "ffmpeg")
    frames = []

    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break
        frames.append(transform(Image.fromarray(frame)))

    reader.close()
    return torch.stack(frames)
