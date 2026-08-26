import os
import numpy as np

from sklearn.model_selection import train_test_split

from src.preprocessing import (
    load_h5_image,
    load_h5_mask
)

from src.features import (
    extract_spectral_features
)

from src.random_forest import (
    prepare_pixel_data,
    split_pixel_data,
    train_random_forest,
    predict_random_forest
)

from src.evaluation import (
    calculate_metrics,
    print_metrics,
    calculate_coverage
)

from src.visualization import (
    normalize_rgb,
    save_overlay
)

from src.cnn import (
    build_segmentation_model
)


# ============================================================
# PATHS
# ============================================================

IMAGE_PATH = "data/Landslide4Sense/images/train/image_1.h5"
MASK_PATH = "data/Landslide4Sense/annotations/train/mask_1.h5"

FIGURE_DIR = "outputs/figures"
MODEL_DIR = "outputs/models"


# ============================================================
# CREATE IMAGE PATCHES
# ============================================================

def create_image_patches(
    features,
    mask,
    patch_size=16
):
    """
    Divide the satellite image into non-overlapping patches.

    Input:
        features -> (128, 128, 6)
        mask     -> (128, 128)

    Output:
        X_patches -> (64, 16, 16, 6)
        y_patches -> (64, 16, 16)
    """

    X_patches = []
    y_patches = []

    height, width, channels = features.shape

    for row in range(0, height, patch_size):

        for col in range(0, width, patch_size):

            feature_patch = features[
                row:row + patch_size,
                col:col + patch_size,
                :
            ]

            mask_patch = mask[
                row:row + patch_size,
                col:col + patch_size
            ]

            X_patches.append(feature_patch)
            y_patches.append(mask_patch)

    return (
        np.array(X_patches),
        np.array(y_patches)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LANDSLIDE DETECTION USING SATELLITE IMAGERY")
    print("=" * 60)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1] Loading Landslide4Sense data...")

    image = load_h5_image(
        IMAGE_PATH
    )

    mask = load_h5_mask(
        MASK_PATH
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Number of bands:",
        image.shape[-1]
    )

    print(
        "Mask classes:",
        np.unique(mask)
    )

    # ========================================================
    # 2. EXTRACT SPECTRAL FEATURES
    # ========================================================

    print("\n[2] Extracting spectral features...")

    features = extract_spectral_features(
        image
    )

    print(
        "Feature shape:",
        features.shape
    )

    print(
        "Features: Blue, Green, Red, NIR, NDVI, NDWI"
    )

    # ========================================================
    # 3. RANDOM FOREST
    # ========================================================

    print("\n[3] Preparing Random Forest data...")

    X, y = prepare_pixel_data(
        features,
        mask
    )

    print(
        "Pixel feature matrix:",
        X.shape
    )

    print(
        "Pixel labels:",
        y.shape
    )

    X_train, X_test, y_train, y_test = split_pixel_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(
        "Training pixels:",
        X_train.shape[0]
    )

    print(
        "Testing pixels:",
        X_test.shape[0]
    )

    print("\nTraining Random Forest...")

    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=100,
        random_state=42
    )

    # --------------------------------------------------------
    # RF TEST PREDICTION
    # --------------------------------------------------------

    y_test_pred = rf_model.predict(
        X_test
    )

    rf_metrics = calculate_metrics(
        y_test,
        y_test_pred
    )

    print_metrics(
        "Random Forest Test Results",
        rf_metrics
    )

    # --------------------------------------------------------
    # RF FULL IMAGE PREDICTION
    # --------------------------------------------------------

    print(
        "\nGenerating full-image Random Forest map..."
    )

    rf_prediction_map = predict_random_forest(
        rf_model,
        features
    )

    # ========================================================
    # 4. RANDOM FOREST COVERAGE
    # ========================================================

    print("\n[4] Landslide coverage...")

    gt_pixels, gt_percentage = calculate_coverage(
        mask
    )

    rf_pixels, rf_percentage = calculate_coverage(
        rf_prediction_map
    )

    print(
        f"Ground Truth: {gt_pixels} pixels "
        f"({gt_percentage:.2f}%)"
    )

    print(
        f"Random Forest: {rf_pixels} pixels "
        f"({rf_percentage:.2f}%)"
    )

    # ========================================================
    # 5. CNN PATCH PREPARATION
    # ========================================================

    print("\n[5] Preparing CNN patches...")

    X_patches, y_patches = create_image_patches(
        features,
        mask,
        patch_size=16
    )

    print(
        "CNN patch shape:",
        X_patches.shape
    )

    print(
        "CNN mask shape:",
        y_patches.shape
    )

    # ========================================================
    # 6. STRATIFIED CNN TRAIN/TEST SPLIT
    # ========================================================

    print("\n[6] Preparing CNN training data...")

    # A patch is considered a landslide patch if
    # it contains at least one landslide pixel.

    patch_labels = (
        np.max(
            y_patches,
            axis=(1, 2)
        ) > 0
    ).astype(np.uint8)

    print(
        "Patch class distribution:"
    )

    print(
        "Non-landslide patches:",
        np.sum(patch_labels == 0)
    )

    print(
        "Landslide patches:",
        np.sum(patch_labels == 1)
    )

    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

    (
        X_seg_train,
        X_seg_test,
        y_seg_train,
        y_seg_test
    ) = train_test_split(
        X_patches,
        y_patches,
        test_size=0.25,
        random_state=42,
        stratify=patch_labels
    )

    print(
        "Training patches:",
        X_seg_train.shape
    )

    print(
        "Testing patches:",
        X_seg_test.shape
    )

    # --------------------------------------------------------
    # VERIFY LANDSLIDE PIXELS
    # --------------------------------------------------------

    train_positive_pixels = np.sum(
        y_seg_train == 1
    )

    test_positive_pixels = np.sum(
        y_seg_test == 1
    )

    print(
        "Training landslide pixels:",
        train_positive_pixels
    )

    print(
        "Testing landslide pixels:",
        test_positive_pixels
    )

    # Safety check
    if test_positive_pixels == 0:

        raise ValueError(
            "ERROR: Test set contains zero landslide pixels. "
            "CNN evaluation cannot be performed correctly."
        )

    # ========================================================
    # 7. CNN NORMALIZATION
    # ========================================================

    print("\n[7] Normalizing CNN input...")

    seg_mean = X_seg_train.mean(
        axis=(0, 1, 2),
        keepdims=True
    )

    seg_std = X_seg_train.std(
        axis=(0, 1, 2),
        keepdims=True
    )

    seg_std = np.where(
        seg_std < 1e-8,
        1.0,
        seg_std
    )

    X_seg_train_norm = (
        X_seg_train - seg_mean
    ) / seg_std

    X_seg_test_norm = (
        X_seg_test - seg_mean
    ) / seg_std

    print(
        "Normalized training shape:",
        X_seg_train_norm.shape
    )

    print(
        "Normalized testing shape:",
        X_seg_test_norm.shape
    )

    print(
        "Training mean after normalization:",
        X_seg_train_norm.mean()
    )

    print(
        "Training standard deviation:",
        X_seg_train_norm.std()
    )

    # ========================================================
    # 8. BUILD CNN
    # ========================================================

    print(
        "\n[8] Building CNN segmentation model..."
    )

    cnn_model = build_segmentation_model(
        input_shape=(16, 16, 6)
    )

    cnn_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    cnn_model.summary()

    # ========================================================
    # 8A. CLASS WEIGHTING
    # ========================================================

    print(
        "\nCalculating pixel class weights..."
    )

    train_mask_pixels = (
        y_seg_train.flatten()
    )

    negative_pixels = np.sum(
        train_mask_pixels == 0
    )

    positive_pixels = np.sum(
        train_mask_pixels == 1
    )

    if positive_pixels == 0:

        raise ValueError(
            "ERROR: Training set contains zero "
            "landslide pixels."
        )

    positive_class_weight = (
        negative_pixels /
        positive_pixels
    )

    print(
        "Negative pixels:",
        negative_pixels
    )

    print(
        "Positive pixels:",
        positive_pixels
    )

    print(
        "Positive class weight:",
        positive_class_weight
    )

    # --------------------------------------------------------
    # PIXEL WEIGHTS
    # --------------------------------------------------------

    sample_weights = np.where(
        y_seg_train == 1,
        positive_class_weight,
        1.0
    )

    # ========================================================
    # CNN TRAINING
    # ========================================================

    print(
        "\nTraining CNN with class weighting..."
    )

    history = cnn_model.fit(
        X_seg_train_norm,
        y_seg_train[..., np.newaxis],
        sample_weight=sample_weights,
        validation_data=(
            X_seg_test_norm,
            y_seg_test[..., np.newaxis]
        ),
        epochs=30,
        batch_size=8,
        verbose=1
    )

    print(
        "\nCNN training completed!"
    )

    # ========================================================
    # 9. CNN TEST PREDICTION
    # ========================================================

    print(
        "\n[9] Evaluating CNN on test patches..."
    )

    cnn_probabilities = cnn_model.predict(
        X_seg_test_norm,
        verbose=0
    )

    print(
        "CNN probability shape:",
        cnn_probabilities.shape
    )

    print(
        "CNN probability minimum:",
        cnn_probabilities.min()
    )

    print(
        "CNN probability maximum:",
        cnn_probabilities.max()
    )

    print(
        "CNN probability mean:",
        cnn_probabilities.mean()
    )

    # Convert probability to binary mask
    cnn_predictions = (
        cnn_probabilities >= 0.5
    ).astype(np.uint8)

    print(
        "CNN prediction shape:",
        cnn_predictions.shape
    )

    print(
        "CNN prediction values:",
        np.unique(cnn_predictions)
    )

    # ========================================================
    # 10. CNN METRICS
    # ========================================================

    print(
        "\n[10] Calculating CNN metrics..."
    )

    y_true_cnn = (
        y_seg_test[..., np.newaxis]
        .astype(np.uint8)
    )

    y_pred_cnn = (
        cnn_predictions
        .astype(np.uint8)
    )

    y_true_flat = (
        y_true_cnn.flatten()
    )

    y_pred_flat = (
        y_pred_cnn.flatten()
    )

    cnn_metrics = calculate_metrics(
        y_true_flat,
        y_pred_flat
    )

    print_metrics(
        "CNN Segmentation Test Results",
        cnn_metrics
    )

    # ========================================================
    # 11. CNN DIAGNOSTICS
    # ========================================================

    print(
        "\n[11] Diagnosing CNN predictions..."
    )

    actual_landslide_pixels = np.sum(
        y_true_flat == 1
    )

    predicted_landslide_pixels = np.sum(
        y_pred_flat == 1
    )

    actual_non_landslide_pixels = np.sum(
        y_true_flat == 0
    )

    predicted_non_landslide_pixels = np.sum(
        y_pred_flat == 0
    )

    print(
        "Actual landslide pixels:",
        actual_landslide_pixels
    )

    print(
        "Predicted landslide pixels:",
        predicted_landslide_pixels
    )

    print(
        "Actual non-landslide pixels:",
        actual_non_landslide_pixels
    )

    print(
        "Predicted non-landslide pixels:",
        predicted_non_landslide_pixels
    )

    print(
        "Unique actual labels:",
        np.unique(y_true_flat)
    )

    print(
        "Unique predicted labels:",
        np.unique(y_pred_flat)
    )

    # ========================================================
    # 12. CNN TEST COVERAGE
    # ========================================================

    cnn_test_pixels = (
        np.sum(y_pred_flat == 1)
    )

    total_test_pixels = (
        y_pred_flat.size
    )

    cnn_test_percentage = (
        cnn_test_pixels /
        total_test_pixels
    ) * 100

    actual_test_pixels = (
        np.sum(y_true_flat == 1)
    )

    actual_test_percentage = (
        actual_test_pixels /
        total_test_pixels
    ) * 100

    print(
        "\nCNN test predicted coverage:",
        f"{cnn_test_percentage:.2f}%"
    )

    print(
        "CNN test ground-truth coverage:",
        f"{actual_test_percentage:.2f}%"
    )

    # ========================================================
    # 13. SAVE CNN MODEL
    # ========================================================

    cnn_model_path = os.path.join(
        MODEL_DIR,
        "cnn_landslide_model.keras"
    )

    cnn_model.save(
        cnn_model_path
    )

    print(
        "\nCNN model saved:",
        cnn_model_path
    )

    # ========================================================
    # 14. RANDOM FOREST VISUALIZATION
    # ========================================================

    print(
        "\n[14] Creating Random Forest visualization..."
    )

    rgb = normalize_rgb(
        image
    )

    image_file = os.path.join(
        FIGURE_DIR,
        "rf_landslide_overlay.png"
    )

    save_overlay(
        rgb,
        rf_prediction_map,
        image_file,
        title="Random Forest Landslide Detection"
    )

    print(
        "Saved:",
        image_file
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PROJECT PIPELINE COMPLETED")
    print("=" * 60)

    print(
        f"Ground truth coverage : "
        f"{gt_percentage:.2f}%"
    )

    print(
        f"RF predicted coverage : "
        f"{rf_percentage:.2f}%"
    )

    print(
        f"CNN test coverage     : "
        f"{cnn_test_percentage:.2f}%"
    )

    print(
        "\nOutputs:"
    )

    print(
        "- RF visualization:",
        image_file
    )

    print(
        "- CNN model:",
        cnn_model_path
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()