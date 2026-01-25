import os, torch
from torch.utils.data import Dataset

class FMRI_MotionDataset(Dataset):
    def __init__(self, fmri_root, motion_dir, seq_len):
        self.samples = []
        self.seq_len = seq_len

        for subj in sorted(os.listdir(fmri_root)):
            subj_dir = os.path.join(fmri_root, subj)
            if not os.path.isdir(subj_dir):
                continue

            for seg in os.listdir(subj_dir):
                seg_dir = os.path.join(subj_dir, seg)
                fmri_path = os.path.join(seg_dir, f"{seg}_avg_embeddings.pt")
                motion_path = os.path.join(motion_dir, f"{seg}_motion.pt")

                if os.path.exists(fmri_path) and os.path.exists(motion_path):
                    self.samples.append((fmri_path, motion_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fmri_path, motion_path = self.samples[idx]
        fmri = torch.load(fmri_path).float()
        motion = torch.load(motion_path).float()
        T = min(self.seq_len, fmri.shape[0], motion.shape[0])
        return fmri[:T], motion[:T]
