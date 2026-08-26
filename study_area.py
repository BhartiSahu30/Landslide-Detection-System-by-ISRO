import os
import matplotlib.pyplot as plt

os.makedirs("outputs/visualizations", exist_ok=True)

# Approximate location of Kodagu District, Karnataka
latitude = 12.3375
longitude = 75.8069

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    longitude,
    latitude,
    s=250,
    marker="*",
    label="Kodagu Study Area"
)

ax.set_title(
    "Selected Study Area for Landslide Assessment\n"
    "Kodagu District, Karnataka, India",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.text(
    longitude + 0.05,
    latitude + 0.05,
    "Kodagu District\nWestern Ghats",
    fontsize=12
)

ax.grid(True, alpha=0.3)

ax.legend()

plt.tight_layout()

output = "outputs/visualizations/kodagu_study_area.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Study area map saved to:", output)