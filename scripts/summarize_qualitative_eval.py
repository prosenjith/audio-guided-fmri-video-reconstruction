import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV = "configs/eval_video_qualitative.yaml"

df = pd.read_csv(
    "/content/drive/MyDrive/Research/evaluation/qualitative/qualitative_summary_notes.csv"
)

summary = (
    df.groupby("subject")["fusion_better"]
    .apply(lambda x: round((x == "✓").mean() * 100, 2))
    .reset_index(name="% Fusion Better")
)

overall = round((df["fusion_better"] == "✓").mean() * 100, 2)

print(summary)
print(f"\n🌍 Overall Fusion Improvement: {overall}%")

plt.figure(figsize=(6,4))
sns.barplot(data=summary, x="subject", y="% Fusion Better", palette="Greens")
plt.ylim(0, 100)
plt.title("Subject-wise Qualitative Fusion Improvement")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
