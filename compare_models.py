import os
import glob
import h5py
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "data/Landslide4Sense/images/train"
MASK_DIR = "data/Landslide4Sense/annotations/train"

TRAIN_IDS = [1, 2, 4, 5, 7, 8, 10]
TEST_IDS = [3, 6, 9]

PATCH_SIZE = 16


# ============================================================
# NUMERIC SORTING
# ============================================================

def numeric_sort(paths):

    return sorted(
        paths,
        key=lambda x: int(
            os.path.basename(x)
            .split("_")[1]
            .split(".")[0]
        )
    )


# ============================================================
# LOAD H5 MASK
# ============================================================

def load_h5_mask(path):

    with h5py.File(path, "r") as f:

        key = list(f.keys())[0]

        mask = np.array(f[key])

    return mask.astype(np.uint8)


# ============================================================
# CREATE MASK PATCHES
# ============================================================

def create_mask_patches(mask):

    patches = []

    height, width = mask.shape

    for row in range(
        0,
        height,
        PATCH_SIZE
    ):

        for col in range(
            0,
            width,
            PATCH_SIZE
        ):

            mask_patch = mask[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            if (
                mask_patch.shape[0] != PATCH_SIZE
                or
                mask_patch.shape[1] != PATCH_SIZE
            ):
                continue

            patches.append(mask_patch)

    return np.array(
        patches,
        dtype=np.uint8
    )


# ============================================================
# LOAD TEST GROUND TRUTH
# ============================================================

def load_test_ground_truth():

    mask_paths = numeric_sort(
        glob.glob(
            os.path.join(
                MASK_DIR,
                "*.h5"
            )
        )
    )

    test_masks = []

    for mask_path in mask_paths:

        image_id = int(
            os.path.basename(mask_path)
            .split("_")[1]
            .split(".")[0]
        )

        if image_id in TEST_IDS:

            mask = load_h5_mask(
                mask_path
            )

            patches = create_mask_patches(
                mask
            )

            test_masks.append(
                patches
            )

            print(
                f"Test image {image_id}: "
                f"mask patches = {patches.shape}, "
                f"landslide pixels = {np.sum(mask == 1)}"
            )

    y_test = np.concatenate(
        test_masks,
        axis=0
    )

    return y_test


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    name,
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    iou = jaccard_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    actual_positive = np.sum(
        y_true == 1
    )

    predicted_positive = np.sum(
        y_pred == 1
    )

    total_pixels = len(
        y_true
    )

    actual_coverage = (
        actual_positive /
        total_pixels *
        100
    )

    predicted_coverage = (
        predicted_positive /
        total_pixels *
        100
    )

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"IoU      : {iou:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print(
        "\nActual landslide pixels:",
        actual_positive
    )

    print(
        "Predicted landslide pixels:",
        predicted_positive
    )

    print(
        f"Actual coverage: "
        f"{actual_coverage:.2f}%"
    )

    print(
        f"Predicted coverage: "
        f"{predicted_coverage:.2f}%"
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "IoU": iou
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RANDOM FOREST vs CNN")
    print("LANDSLIDE SEGMENTATION COMPARISON")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------

    print("\nLoading test ground truth...")

    y_test = load_test_ground_truth()

    y_true = (
        y_test
        .reshape(-1)
        .astype(np.uint8)
    )

    print(
        "\nGround truth shape:",
        y_true.shape
    )

    print(
        "Actual landslide pixels:",
        np.sum(y_true)
    )

    # --------------------------------------------------------
    # LOAD RF PREDICTIONS
    # --------------------------------------------------------

    rf_predictions = np.load(
        "outputs/predictions/rf_test_predictions.npy"
    )

    rf_predictions = (
        rf_predictions
        .reshape(-1)
        .astype(np.uint8)
    )

    print(
        "\nRF prediction shape:",
        rf_predictions.shape
    )

    # --------------------------------------------------------
    # LOAD CNN PREDICTIONS
    # --------------------------------------------------------

    cnn_predictions = np.load(
        "outputs/predictions/cnn_test_predictions.npy"
    )

    cnn_predictions = (
        cnn_predictions
        .reshape(-1)
        .astype(np.uint8)
    )

    print(
        "CNN prediction shape:",
        cnn_predictions.shape
    )

    # --------------------------------------------------------
    # VERIFY SAME NUMBER OF PIXELS
    # --------------------------------------------------------

    if len(y_true) != len(rf_predictions):

        raise ValueError(
            "RF predictions and ground truth "
            "have different numbers of pixels."
        )

    if len(y_true) != len(cnn_predictions):

        raise ValueError(
            "CNN predictions and ground truth "
            "have different numbers of pixels."
        )

    print(
        "\nPixel count verification: PASSED"
    )

    # --------------------------------------------------------
    # RF METRICS
    # --------------------------------------------------------

    rf_results = calculate_metrics(
        "RANDOM FOREST RESULTS",
        y_true,
        rf_predictions
    )

    # --------------------------------------------------------
    # CNN METRICS
    # --------------------------------------------------------

    cnn_results = calculate_metrics(
        "CNN RESULTS",
        y_true,
        cnn_predictions
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 75)
    print("RANDOM FOREST vs CNN")
    print("=" * 75)

    print(
        f"{'Metric':<15}"
        f"{'Random Forest':<20}"
        f"{'CNN':<20}"
    )

    print("-" * 75)

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "IoU"
    ]:

        print(
            f"{metric:<15}"
            f"{rf_results[metric]:<20.4f}"
            f"{cnn_results[metric]:<20.4f}"
        )

    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()