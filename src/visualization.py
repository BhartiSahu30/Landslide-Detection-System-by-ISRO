import os
import numpy as np
import matplotlib.pyplot as plt


def normalize_rgb(image):
    """
    Create an RGB visualization from the first
    three selected visible bands.
    """

    rgb = image[:, :, :3].copy()

    rgb_min = rgb.min(
        axis=(0, 1),
        keepdims=True
    )

    rgb_max = rgb.max(
        axis=(0, 1),
        keepdims=True
    )

    rgb = (
        rgb - rgb_min
    ) / (
        rgb_max - rgb_min + 1e-8
    )

    return np.clip(
        rgb,
        0,
        1
    )


def plot_mask(mask, title="Landslide Mask"):
    """
    Display a binary landslide mask.
    """

    plt.figure(figsize=(7, 7))

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_overlay(
    rgb,
    prediction_map,
    title="Landslide Detection"
):
    """
    Overlay predicted landslide regions
    on satellite imagery.
    """

    plt.figure(figsize=(8, 8))

    plt.imshow(rgb)

    masked_prediction = np.ma.masked_where(
        prediction_map == 0,
        prediction_map
    )

    plt.imshow(
        masked_prediction,
        cmap="Reds",
        alpha=0.5
    )

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    ground_truth,
    rf_prediction,
    cnn_prediction,
    save_path=None
):
    """
    Compare ground truth, Random Forest and CNN
    prediction maps.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        ground_truth,
        cmap="gray"
    )

    axes[0].set_title(
        "Ground Truth"
    )

    axes[0].axis("off")

    axes[1].imshow(
        rf_prediction,
        cmap="gray"
    )

    axes[1].set_title(
        "Random Forest"
    )

    axes[1].axis("off")

    axes[2].imshow(
        cnn_prediction,
        cmap="gray"
    )

    axes[2].set_title(
        "CNN Segmentation"
    )

    axes[2].axis("off")

    plt.suptitle(
        "Landslide Detection Comparison",
        fontsize=15
    )

    plt.tight_layout()

    if save_path is not None:

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def save_overlay(
    rgb,
    prediction_map,
    save_path,
    title="Landslide Detection"
):
    """
    Save an overlay visualization.
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(rgb)

    masked_prediction = np.ma.masked_where(
        prediction_map == 0,
        prediction_map
    )

    plt.imshow(
        masked_prediction,
        cmap="Reds",
        alpha=0.5
    )

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
