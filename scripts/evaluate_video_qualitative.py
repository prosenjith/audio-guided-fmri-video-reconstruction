import os
import sys
import yaml
import imageio
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.video.qualitative import sample_frames, create_timeline_figure

cfg = yaml.safe_load(open("configs/eval_video_qualitative.yaml"))

GT_DIR = cfg["paths"]["gt_dir"]
GEN_DIR = cfg["paths"]["generated_dir"]
OUT_DIR = cfg["paths"]["timeline_dir"]

os.makedirs(OUT_DIR, exist_ok=True)

fps = cfg["video"]["fps_gt"]
seg_dur = cfg["video"]["segment_duration_sec"]
gt_frames_count = fps * seg_dur

records = []

for subj in cfg["subjects"]:
    subj_gen = os.path.join(GEN_DIR, subj)
    subj_out = os.path.join(OUT_DIR, subj)
    os.makedirs(subj_out, exist_ok=True)

    for seg in tqdm(cfg["test_segments"], desc=f"{subj}"):
        save_path = os.path.join(subj_out, f"{seg}_timeline.png")
        if os.path.exists(save_path):
            print(f"⏩ Skipping {subj}/{seg} (timeline exists)")
            continue

        gt_path = os.path.join(GT_DIR, f"{seg}.mp4")
        fmri_path = os.path.join(subj_gen, f"{seg}_fmri_recon.mp4")
        fusion_path = os.path.join(subj_gen, f"{seg}_fusion_recon.mp4")

        if not all(os.path.exists(p) for p in [gt_path, fmri_path, fusion_path]):
            continue

        reader = imageio.get_reader(gt_path, "ffmpeg")
        gt_frames = []
        for i, frame in enumerate(reader):
            if i < gt_frames_count:
                gt_frames.append(Image.fromarray(frame))
            else:
                break
        reader.close()

        n = cfg["video"]["num_samples"]
        gt_samples = [gt_frames[int(i*(len(gt_frames)-1)/(n-1))] for i in range(n)]
        fmri_samples = sample_frames(fmri_path, n)
        fusion_samples = sample_frames(fusion_path, n)

        create_timeline_figure(gt_samples, fmri_samples, fusion_samples,
                               save_path, seg_dur)

        records.append({
            "subject": subj,
            "segment": seg,
            "path": save_path,
            "motion_alignment": "",
            "temporal_smoothness": "",
            "object_coherence": "",
            "fusion_better": ""
        })

df = pd.DataFrame(records)
df.to_csv(cfg["paths"]["summary_csv"], index=False)
print("✅ Qualitative timelines generated.")
