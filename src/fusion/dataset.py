import os, json, torch, pandas as pd
from torch.utils.data import Dataset
from src.fusion.align import align_w2v2_to_TR

class FMRI_AudioFusionDataset(Dataset):
    def __init__(self, csv_path, fmri_root, audio_root, seq_len=16):
        self.df = pd.read_csv(csv_path)
        self.fmri_root = fmri_root
        self.audio_root = audio_root
        self.seq_len = seq_len
        self.video_ids = set(self.df["video_id"].astype(str))
        self.samples = []

        for subj in sorted(os.listdir(fmri_root)):
            subj_dir = os.path.join(fmri_root, subj)
            if not os.path.isdir(subj_dir):
                continue

            for seg in os.listdir(subj_dir):
                if seg not in self.video_ids:
                    continue

                seg_dir = os.path.join(subj_dir, seg)
                avg_pt = os.path.join(seg_dir, f"{seg}_avg_embeddings.pt")
                meta_json = os.path.join(seg_dir, f"{seg}_meta.json")

                if os.path.exists(avg_pt) and os.path.exists(meta_json):
                    self.samples.append((subj, seg, avg_pt, meta_json))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, seg, fmri_path, meta_path = self.samples[idx]
        fmri_meta = json.load(open(meta_path))
        fmri = torch.load(fmri_path).float()

        # fix dim to 245
        fmri = fmri[:, :245] if fmri.shape[1] >= 245 else \
               torch.cat([fmri, torch.zeros(fmri.shape[0], 245 - fmri.shape[1])], 1)

        audio_path = os.path.join(self.audio_root, f"{seg}_full_w2v2_2s.pt")
        audio_meta = json.load(open(audio_path.replace(".pt", "_meta.json")))
        W = torch.load(audio_path).float()

        audio_aligned = align_w2v2_to_TR(
            fmri_meta, W,
            audio_meta["secs_per_window"],
            audio_meta["hop_sec"]
        )

        T = min(self.seq_len, fmri.shape[0], audio_aligned.shape[0])
        return fmri[:T], audio_aligned[:T], subj, seg