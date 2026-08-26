import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "data/Landslide4Sense/images/train/image_6.h5"
MASK_PATH = "data/Landslide4Sense/annotations/train/mask_6.h5"

RF_PATH = "outputs/predictions/rf_test_predictions.npy"
CNN_PATH = "outputs/predictions/cnn_test_predictions.npy"

OUTPUT_DIR = "outputs/visualizations"

PATCH_SIZE = 16


# ============================================================
# LOAD H5
# ============================================================

def load_h5(path):

    with h5py.File(path, "r") as f:

        key = list(f.keys())[0]

        return np.array(f[key])


# ============================================================
# PERCENTILE CONTRAST STRETCH
# ============================================================

def stretch_band(band):

    low = np.percentile(
        band,
        2
    )

    high = np.percentile(
        band,
        98
    )

    band = (
        band - low
    ) / (
        high - low + 1e-8
    )

    return np.clip(
        band,
        0,
        1
    )


# ============================================================
# CREATE RGB
# ============================================================

def create_rgb(image):

    # Landslide4Sense:
    # B1 = index 0
    # B2 = index 1
    # B3 = index 2
    # B4 = index 3
    #
    # For natural-color Sentinel-2:
    # Blue  = B2
    # Green = B3
    # Red   = B4

    blue = image[:, :, 1]
    green = image[:, :, 2]
    red = image[:, :, 3]

    red = stretch_band(red)
    green = stretch_band(green)
    blue = stretch_band(blue)

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    return rgb


# ============================================================
# RECONSTRUCT PATCH PREDICTION
# ============================================================

def reconstruct_prediction(
    predictions,
    image_index
):

    start = image_index * 64
    end = start + 64

    patches = predictions[
        start:end
    ]

    result = np.zeros(
        (128, 128),
        dtype=np.uint8
    )

    index = 0

    for row in range(
        0,
        128,
        PATCH_SIZE
    ):

        for col in range(
            0,
            128,
            PATCH_SIZE
        ):

            patch = patches[index]

            if patch.ndim == 3:

                patch = patch[:, :, 0]

            result[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ] = patch

            index += 1

    return result


# ============================================================
# LANDSLIDE OVERLAY
# ============================================================

def create_overlay(
    ax,
    rgb,
    mask,
    title
):

    ax.imshow(
        rgb,
        interpolation="nearest"
    )

    overlay = np.ma.masked_where(
        mask == 0,
        mask
    )

    ax.imshow(
        overlay,
        cmap=ListedColormap(
            ["red"]
        ),
        alpha=0.65,
        interpolation="nearest"
    )

    # Draw landslide boundary

    if np.any(mask == 1):

        ax.contour(
            mask,
            levels=[0.5],
            linewidths=1.5
        )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("off")


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "Loading satellite image..."
    )

    image = load_h5(
        IMAGE_PATH
    )

    mask = load_h5(
        MASK_PATH
    )

    rgb = create_rgb(
        image
    )

    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    print(
        "Loading model predictions..."
    )

    rf_predictions = np.load(
        RF_PATH
    )

    cnn_predictions = np.load(
        CNN_PATH
    )

    # Image 6 is the second test image.
    #
    # Test images:
    # image 3 -> index 0
    # image 6 -> index 1
    # image 9 -> index 2

    rf_map = reconstruct_prediction(
        rf_predictions,
        image_index=1
    )

    cnn_map = reconstruct_prediction(
        cnn_predictions,
        image_index=1
    )

    # --------------------------------------------------------
    # CREATE PROFESSIONAL FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10)
    )

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    axes[0, 0].imshow(
        rgb,
        interpolation="nearest"
    )

    axes[0, 0].set_title(
        "Sentinel-2 RGB Satellite Image",
        fontsize=13,
        fontweight="bold"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    create_overlay(
        axes[0, 1],
        rgb,
        mask,
        "Ground Truth Landslide"
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    create_overlay(
        axes[1, 0],
        rgb,
        rf_map,
        "Random Forest Prediction"
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    create_overlay(
        axes[1, 1],
        rgb,
        cnn_map,
        "CNN Prediction"
    )

    # --------------------------------------------------------
    # FIGURE TITLE
    # --------------------------------------------------------

    fig.suptitle(
        "Satellite-Based Landslide Detection — Test Image 6",
        fontsize=16,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.02,
        "Red overlay = detected landslide area",
        ha="center",
        fontsize=11
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 0.95]
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "landslide_detection_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nProfessional visualization saved:"
    )

    print(
        output_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()