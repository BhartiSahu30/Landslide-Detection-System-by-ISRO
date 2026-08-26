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

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "data/Landslide4Sense/images/train"
MASK_DIR = "data/Landslide4Sense/annotations/train"

TRAIN_IDS = [1, 2, 4, 5, 7, 8, 10]
TEST_IDS = [3, 6, 9]

PATCH_SIZE = 16

EPOCHS = 30
BATCH_SIZE = 8

np.random.seed(42)
tf.random.set_seed(42)


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
# LOAD H5 IMAGE
# ============================================================

def load_h5_image(path):

    with h5py.File(path, "r") as f:

        key = list(f.keys())[0]

        image = np.array(f[key])

    return image


# ============================================================
# LOAD H5 MASK
# ============================================================

def load_h5_mask(path):

    with h5py.File(path, "r") as f:

        key = list(f.keys())[0]

        mask = np.array(f[key])

    return mask.astype(np.uint8)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    blue = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)

    ndvi = (
        (nir - red) /
        (nir + red + 1e-8)
    )

    ndwi = (
        (green - nir) /
        (green + nir + 1e-8)
    )

    features = np.stack(
        [
            blue,
            green,
            red,
            nir,
            ndvi,
            ndwi
        ],
        axis=-1
    )

    return features


# ============================================================
# CREATE PATCHES
# ============================================================

def create_patches(features, mask):

    X_patches = []
    y_patches = []

    height, width, channels = features.shape

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

            feature_patch = features[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
                :
            ]

            mask_patch = mask[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ]

            if (
                feature_patch.shape[0] != PATCH_SIZE
                or
                feature_patch.shape[1] != PATCH_SIZE
            ):
                continue

            X_patches.append(
                feature_patch
            )

            y_patches.append(
                mask_patch
            )

    return (
        np.array(
            X_patches,
            dtype=np.float32
        ),
        np.array(
            y_patches,
            dtype=np.float32
        )
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(X_train, X_test):

    mean = X_train.mean(
        axis=(0, 1, 2),
        keepdims=True
    )

    std = X_train.std(
        axis=(0, 1, 2),
        keepdims=True
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std
    )

    X_train = (
        X_train - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    return X_train, X_test


# ============================================================
# BUILD CNN
# ============================================================

def build_model():

    model = Sequential([

        Input(
            shape=(PATCH_SIZE, PATCH_SIZE, 6)
        ),

        Conv2D(
            16,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        Conv2D(
            16,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        Conv2D(
            1,
            (1, 1),
            padding="same",
            activation="sigmoid"
        )

    ])

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MULTI-IMAGE CNN LANDSLIDE SEGMENTATION")
    print("=" * 60)

    # --------------------------------------------------------
    # FIND DATA
    # --------------------------------------------------------

    image_paths = numeric_sort(
        glob.glob(
            os.path.join(
                IMAGE_DIR,
                "*.h5"
            )
        )
    )

    mask_paths = numeric_sort(
        glob.glob(
            os.path.join(
                MASK_DIR,
                "*.h5"
            )
        )
    )

    train_X = []
    train_y = []

    test_X = []
    test_y = []

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    for image_path, mask_path in zip(
        image_paths,
        mask_paths
    ):

        image_id = int(
            os.path.basename(image_path)
            .split("_")[1]
            .split(".")[0]
        )

        image = load_h5_image(
            image_path
        )

        mask = load_h5_mask(
            mask_path
        )

        features = extract_features(
            image
        )

        X_patches, y_patches = create_patches(
            features,
            mask
        )

        print(
            f"image_{image_id}: "
            f"patches={X_patches.shape}, "
            f"landslide_pixels={np.sum(mask == 1)}"
        )

        if image_id in TRAIN_IDS:

            train_X.append(
                X_patches
            )

            train_y.append(
                y_patches
            )

        elif image_id in TEST_IDS:

            test_X.append(
                X_patches
            )

            test_y.append(
                y_patches
            )

    # --------------------------------------------------------
    # COMBINE DATA
    # --------------------------------------------------------

    X_train = np.concatenate(
        train_X,
        axis=0
    )

    y_train = np.concatenate(
        train_y,
        axis=0
    )

    X_test = np.concatenate(
        test_X,
        axis=0
    )

    y_test = np.concatenate(
        test_y,
        axis=0
    )

    print("\nTRAINING DATA")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print("\nTEST DATA")

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print("\nNormalizing data...")

    X_train, X_test = normalize_data(
        X_train,
        X_test
    )

    print(
        "Training mean:",
        X_train.mean()
    )

    print(
        "Training std:",
        X_train.std()
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    positive_pixels = np.sum(
        y_train == 1
    )

    negative_pixels = np.sum(
        y_train == 0
    )

    positive_weight = (
        negative_pixels /
        max(positive_pixels, 1)
    )

    print("\nPIXEL CLASS DISTRIBUTION")

    print(
        "Positive:",
        positive_pixels
    )

    print(
        "Negative:",
        negative_pixels
    )

    print(
        "Positive class weight:",
        positive_weight
    )

    # --------------------------------------------------------
    # BUILD CNN
    # --------------------------------------------------------

    print("\nBuilding CNN...")

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # SAMPLE WEIGHTS
    # --------------------------------------------------------

    sample_weights = np.where(
        y_train == 1,
        positive_weight,
        1.0
    )

    # --------------------------------------------------------
    # TRAIN CNN
    # --------------------------------------------------------

    print("\nTraining CNN...")

    history = model.fit(
        X_train,
        y_train[..., np.newaxis],
        sample_weight=sample_weights,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1
    )

    print("\nCNN training completed!")

    # ========================================================
    # SAVE CNN MODEL
    # ========================================================

    os.makedirs(
        "outputs/models",
        exist_ok=True
    )

    cnn_model_path = (
        "outputs/models/cnn_landslide.keras"
    )

    model.save(
        cnn_model_path
    )

    print(
        "CNN model saved:",
        cnn_model_path
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    print("\nPredicting test patches...")

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.uint8)

    # --------------------------------------------------------
    # PREPARE LABELS
    # --------------------------------------------------------

    y_true = (
        y_test
        .reshape(-1)
        .astype(np.uint8)
    )

    y_pred = (
        predictions
        .reshape(-1)
        .astype(np.uint8)
    )

    print(
        "\nPrediction shape:",
        predictions.shape
    )

    print(
        "Unique predictions:",
        np.unique(predictions)
    )

    print(
        "Predicted landslide pixels:",
        np.sum(predictions)
    )

    # ========================================================
    # METRICS
    # ========================================================

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

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CNN SEGMENTATION TEST RESULTS")
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

    # ========================================================
    # LANDSLIDE COVERAGE
    # ========================================================

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
    print("CNN LANDSLIDE COVERAGE")
    print("=" * 60)

    print(
        "Actual landslide pixels:",
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

    # ========================================================
    # PROBABILITY STATISTICS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CNN PROBABILITY STATISTICS")
    print("=" * 60)

    print(
        "Minimum:",
        probabilities.min()
    )

    print(
        "Maximum:",
        probabilities.max()
    )

    print(
        "Mean:",
        probabilities.mean()
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    os.makedirs(
        "outputs/predictions",
        exist_ok=True
    )

    np.save(
        "outputs/predictions/cnn_test_predictions.npy",
        predictions
    )

    np.save(
        "outputs/predictions/cnn_test_probabilities.npy",
        probabilities
    )

    print(
        "\nCNN predictions saved."
    )

    print(
        "CNN probabilities saved."
    )

    print("\nCNN PIPELINE COMPLETED")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()