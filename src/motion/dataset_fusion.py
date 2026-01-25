import os, torch
from torch.utils.data import Dataset

class Fusion_MotionDataset(Dataset):
    def __init__(self, fusion_root, motion_dir, seq_len):
        self.samples = []
        self.seq_len = seq_len

        for subj in sorted(os.listdir(fusion_root)):
            subj_dir = os.path.join(fusion_root, subj)
            if not os.path.isdir(subj_dir):
                continue

            for f in os.listdir(subj_dir):
                if not f.endswith("_fused_embeddings.pt"):
                    continue

                seg = f.replace("_fused_embeddings.pt", "")
                fusion_path = os.path.join(subj_dir, f)
                motion_path = os.path.join(motion_dir, f"{seg}_motion.pt")

                if os.path.exists(motion_path):
                    self.samples.append((fusion_path, motion_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fusion_path, motion_path = self.samples[idx]
        fusion = torch.load(fusion_path).float()
        motion = torch.load(motion_path).float()
        T = min(self.seq_len, fusion.shape[0], motion.shape[0])
        return fusion[:T], motion[:T]
