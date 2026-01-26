import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Subject-wise and global summaries
# ============================================================
def summarize_metrics(df):
    """
    Returns:
      - subject-wise mean
      - global mean
      - global mean ± std
    """
    subject_summary = (
        df.groupby(["subject", "mode"])[["SSIM", "PSNR", "LPIPS"]]
        .mean()
        .reset_index()
    )

    global_mean = (
        df.groupby("mode")[["SSIM", "PSNR", "LPIPS"]]
        .mean()
        .reset_index()
    )

    global_stats = (
        df.groupby("mode")[["SSIM", "PSNR", "LPIPS"]]
        .agg(["mean", "std"])
    )

    return subject_summary, global_mean, global_stats


# ============================================================
# 2. Basic comparison plots
# ============================================================
def plot_basic_comparison(df):
    melted = df.melt(
        id_vars=["subject", "mode"],
        value_vars=["SSIM", "PSNR", "LPIPS"],
        var_name="Metric",
        value_name="Value"
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=melted,
        x="Metric",
        y="Value",
        hue="mode",
        ci="sd",
        capsize=0.1,
        palette=["#3b82f6", "#22c55e"]
    )
    plt.title("Evaluation Metrics — fMRI-only vs Fusion")
    plt.ylabel("Mean ± SD")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Subject-wise bar plots
# ============================================================
def plot_subject_wise(df):
    summary = (
        df.groupby(["subject", "mode"])[["SSIM", "PSNR", "LPIPS"]]
        .mean()
        .reset_index()
    )

    melted = summary.melt(
        id_vars=["subject", "mode"],
        var_name="Metric",
        value_name="Value"
    )

    sns.catplot(
        data=melted,
        x="Metric",
        y="Value",
        hue="mode",
        col="subject",
        kind="bar",
        palette=["#3b82f6", "#22c55e"],
        height=4,
        aspect=0.9
    )
    plt.subplots_adjust(top=0.85)
    plt.suptitle("Per-Subject Metric Comparison")
    plt.show()


# ============================================================
# 4. Zoomed metric plots (paper figures)
# ============================================================
def plot_zoomed_metrics(df):
    summary = df.groupby("mode")[["SSIM", "PSNR", "LPIPS"]].mean().reset_index()
    melted = summary.melt(id_vars="mode", var_name="Metric", value_name="Value")

    y_lims = {
        "SSIM": (0.45, 0.65),
        "PSNR": (20.5, 23.5),
        "LPIPS": (0.12, 0.19)
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric in zip(axes, ["SSIM", "PSNR", "LPIPS"]):
        data = melted[melted["Metric"] == metric]
        sns.barplot(
            data=data,
            x="Metric",
            y="Value",
            hue="mode",
            palette=["#3b82f6", "#22c55e"],
            ax=ax,
            width=0.5
        )
        ax.set_ylim(y_lims[metric])
        ax.set_title(f"{metric} Comparison")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.suptitle("Zoomed Metric Comparison (Paper Figure)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Improvement analysis (Fusion vs fMRI-only)
# ============================================================
def plot_improvement(df):
    mean_metrics = df.groupby("mode")[["SSIM", "PSNR", "LPIPS"]].mean()

    improvements = (
        (mean_metrics.loc["fusion"] - mean_metrics.loc["fmri_only"])
        / mean_metrics.loc["fmri_only"] * 100
    )

    imp_df = improvements.reset_index()
    imp_df.columns = ["Metric", "Improvement (%)"]

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=imp_df,
        x="Metric",
        y="Improvement (%)",
        palette=["#3b82f6", "#22c55e", "#f97316"],
        width=0.5
    )

    for p in plt.gca().patches:
        plt.text(
            p.get_x() + p.get_width()/2.,
            p.get_height(),
            f"{p.get_height():+.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Relative Improvement of Fusion over fMRI-only")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Radar plot summary
# ============================================================
def plot_radar(df):
    metrics = ["SSIM", "PSNR", "LPIPS"]
    avg_fmri   = [df[df["mode"]=="fmri_only"][m].mean() for m in metrics]
    avg_fusion = [df[df["mode"]=="fusion"][m].mean() for m in metrics]

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    avg_fmri += avg_fmri[:1]
    avg_fusion += avg_fusion[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, avg_fmri, "o-", linewidth=2, label="fMRI-only")
    ax.fill(angles, avg_fmri, alpha=0.25)

    ax.plot(angles, avg_fusion, "o-", linewidth=2, label="Fusion")
    ax.fill(angles, avg_fusion, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_title("Overall Performance Summary")
    ax.legend(loc="upper right")

    plt.show()


# ============================================================
# 7. Master function (used by evaluate_videos.py)
# ============================================================
def plot_all(df, output_dir=None):
    plot_basic_comparison(df)
    plot_subject_wise(df)
    plot_zoomed_metrics(df)
    plot_improvement(df)
    plot_radar(df)