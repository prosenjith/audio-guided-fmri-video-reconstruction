import os
import sys
import yaml

# ============================================================
# Ensure repository root is on PYTHONPATH
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ============================================================
# Imports from src/
# ============================================================
from src.pipeline.segment_finder import find_seg_dirs
from src.pipeline.segment_processor import process_segment

# ============================================================
# Load configuration
# ============================================================
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "fmri_embedding.yaml")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

FMRI_ROOT = cfg["paths"]["fmri_root"]
OUT_ROOT = cfg["paths"]["output_root"]

print("==============================================")
print("Running fMRI Embedding Pipeline")
print(f"FMRI_ROOT : {FMRI_ROOT}")
print(f"OUT_ROOT  : {OUT_ROOT}")
print("==============================================\n")

# ============================================================
# Iterate over subjects
# ============================================================
for subject in sorted(os.listdir(FMRI_ROOT)):
    subject_root = os.path.join(
        FMRI_ROOT,
        subject,
        "video_fmri_dataset",
        subject,
        "fmri"
    )

    if not os.path.isdir(subject_root):
        continue

    print(f"\n🔹 Processing subject: {subject}")

    # --------------------------------------------------------
    # Recursively find all segment directories
    # --------------------------------------------------------
    seg_dirs = find_seg_dirs(subject_root)

    if not seg_dirs:
        print("   ⚠️ No segments found")
        continue

    # --------------------------------------------------------
    # Process each segment
    # --------------------------------------------------------
    for seg_dir in seg_dirs:
        seg_name = os.path.basename(seg_dir)

        out_dir = os.path.join(OUT_ROOT, subject, seg_name)
        avg_out = os.path.join(out_dir, f"{seg_name}_avg_embeddings.pt")

        # -------------------------
        # Skip if already processed
        # -------------------------
        if os.path.exists(avg_out):
            print(f"   ⏩ Skipping {seg_name} (already processed)")
            continue

        print(f"   ▶ Segment: {seg_name}")

        try:
            process_segment(
                seg_dir=seg_dir,
                subject_name=subject,
                output_root=OUT_ROOT,
                use_mni=cfg["use_mni"],
                normalize_per_run=cfg["normalize_per_run"],
                merge=cfg["merge_runs"],
                n_components=cfg["n_components"],
                batch_size=cfg["batch_size"],
            )
        except Exception as e:
            print(f"   ❌ Error processing {subject}/{seg_name}: {e}")

print("\n✅ fMRI embedding pipeline completed.")