import os
import imageio
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, clear_output, Image as IPyImage
import ipywidgets as widgets


def sample_frames(video_path, n_samples):
    reader = imageio.get_reader(video_path, "ffmpeg")
    total = reader.count_frames()
    indices = [int(i * (total - 1) / (n_samples - 1)) for i in range(n_samples)]
    frames = [Image.fromarray(reader.get_data(idx)) for idx in indices]
    reader.close()
    return frames


def create_timeline_figure(gt_frames, fmri_frames, fusion_frames,
                           save_path, seg_duration):
    num_samples = len(gt_frames)
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 6))
    rows = [
        ("Ground Truth", gt_frames),
        ("fMRI-only", fmri_frames),
        ("Fusion", fusion_frames)
    ]

    for r, (label, frames) in enumerate(rows):
        for c, img in enumerate(frames):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(
                    f"t={c * seg_duration / (num_samples - 1):.1f}s",
                    fontsize=10
                )
        axes[r, 0].set_ylabel(label, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def launch_annotation_ui(df, csv_path):
    import os
    from IPython.display import display, clear_output, Image as IPyImage
    import ipywidgets as widgets

    required_cols = [
        "subject",
        "segment",
        "path",
        "motion_alignment",
        "temporal_smoothness",
        "object_coherence",
        "fusion_better",
    ]

    # --- Safety check ---
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # --- Find remaining rows to annotate ---
    pending = df[
        df["fusion_better"].isna() | (df["fusion_better"] == "")
    ]

    if pending.empty:
        print("✅ All qualitative evaluations are already complete.")
        return

    current_index = [pending.index.min()]

    # --- Widgets ---
    motion_box = widgets.ToggleButtons(options=["✓", "x"], description="Motion:")
    smooth_box = widgets.ToggleButtons(options=["✓", "x"], description="Smooth:")
    shape_box  = widgets.ToggleButtons(options=["✓", "x"], description="Coherence:")
    better_box = widgets.ToggleButtons(options=["✓", "x"], description="Fusion better:")

    next_btn = widgets.Button(description="Next ▶", button_style="success")
    skip_btn = widgets.Button(description="⏩ Skip", button_style="warning")

    controls = widgets.VBox([
        motion_box,
        smooth_box,
        shape_box,
        better_box,
        widgets.HBox([next_btn, skip_btn])
    ])

    # --- Helpers ---
    def save():
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

    def show(idx):
        clear_output(wait=True)

        if idx >= len(df):
            print("✅ All evaluations complete.")
            return

        row = df.loc[idx]
        print(f"🔹 {row.subject} — {row.segment}")
        display(IPyImage(filename=row.path))
        display(controls)

    def on_next(_):
        idx = current_index[0]
        df.loc[idx, "motion_alignment"] = motion_box.value
        df.loc[idx, "temporal_smoothness"] = smooth_box.value
        df.loc[idx, "object_coherence"] = shape_box.value
        df.loc[idx, "fusion_better"] = better_box.value

        save()

        # Move to next pending
        remaining = df[
            df["fusion_better"].isna() | (df["fusion_better"] == "")
        ]

        if remaining.empty:
            clear_output(wait=True)
            print("✅ All qualitative evaluations completed.")
            return

        current_index[0] = remaining.index.min()
        show(current_index[0])

    def on_skip(_):
        remaining = df[
            df["fusion_better"].isna() | (df["fusion_better"] == "")
        ]

        if remaining.empty:
            clear_output(wait=True)
            print("✅ All qualitative evaluations completed.")
            return

        current_index[0] = remaining.index.min()
        show(current_index[0])

    next_btn.on_click(on_next)
    skip_btn.on_click(on_skip)

    show(current_index[0])

    motion_box  = widgets.ToggleButtons(options=["✓", "✗"], description="Motion Align:")
    smooth_box  = widgets.ToggleButtons(options=["✓", "✗"], description="Smoothness:")
    shape_box   = widgets.ToggleButtons(options=["✓", "✗"], description="Coherence:")
    better_box  = widgets.ToggleButtons(options=["✓", "✗"], description="Fusion Better:")
    next_btn    = widgets.Button(description="Next ▶", button_style="success")
    skip_btn    = widgets.Button(description="⏩ Skip", button_style="warning")

    controls = widgets.VBox([
        motion_box, smooth_box, shape_box, better_box,
        widgets.HBox([next_btn, skip_btn])
    ])

    current_index = [df[(df["fusion_better"] == "") | (df["fusion_better"].isna())].index.min() or 0]

    def save_progress():
        df.to_csv(csv_path, index=False)
        print(f"💾 Progress saved → {csv_path}")

    def show_image(idx):
        clear_output(wait=True)
        if idx >= len(df):
            print("✅ All evaluations complete.")
            return
        row = df.loc[idx]
        print(f"🔹 {row.subject} — {row.segment} ({idx+1}/{len(df)})")
        display(IPyImage(filename=row.path))
        display(controls)

    def on_next_click(b):
        idx = current_index[0]
        df.loc[idx, "motion_alignment"] = motion_box.value
        df.loc[idx, "temporal_smoothness"] = smooth_box.value
        df.loc[idx, "object_coherence"] = shape_box.value
        df.loc[idx, "fusion_better"] = better_box.value
        save_progress()
        current_index[0] += 1
        show_image(current_index[0])

    def on_skip_click(b):
        current_index[0] += 1
        show_image(current_index[0])

    next_btn.on_click(on_next_click)
    skip_btn.on_click(on_skip_click)

    show_image(current_index[0])
