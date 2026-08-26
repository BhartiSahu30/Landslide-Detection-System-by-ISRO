import matplotlib.pyplot as plt
import numpy as np
import os

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "IoU"
]

rf = [
    99.13,
    3.97,
    1.60,
    2.28,
    1.15
]

cnn = [
    97.13,
    8.78,
    37.50,
    14.22,
    7.66
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    rf,
    width,
    label="Random Forest"
)

plt.bar(
    x + width / 2,
    cnn,
    width,
    label="Lightweight CNN"
)

plt.xticks(
    x,
    metrics
)

plt.ylabel("Score (%)")

plt.xlabel("Evaluation Metric")

plt.title(
    "Random Forest vs Lightweight CNN\n"
    "Landslide Segmentation Performance"
)

plt.ylim(
    0,
    105
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

os.makedirs(
    "outputs/visualizations",
    exist_ok=True
)

plt.savefig(
    "outputs/visualizations/model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Performance chart saved successfully."
)