import os
import glob
import h5py
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "data/Landslide4Sense/images/train"
MASK_DIR = "data/Landslide4Sense/annotations/train"

MODEL_DIR = "outputs/models"
PREDICTION_DIR = "outputs/predictions"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)


# ============================================================
# LOAD H5 DATA
# ============================================================

def load_h5(path):

    with h5py.File(path, "r") as f:

        key = list(f.keys())[0]

        data = np.array(f[key])

    return data


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(image):

    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    nir = image[:, :, 3]

    ndvi = (nir - red) / (nir + red + 1e-8)

    ndwi = (green - nir) / (green + nir + 1e-8)

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
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MULTI-IMAGE LANDSLIDE DETECTION")
    print("=" * 60)

    image_files = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.h5"))
    )

    mask_files = sorted(
        glob.glob(os.path.join(MASK_DIR, "*.h5"))
    )

    print("Images:", len(image_files))
    print("Masks :", len(mask_files))

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_ids = [1, 2, 4, 5, 7, 8, 10]

    test_ids = [3, 6, 9]

    print("\nTRAINING IMAGES:")
    print(train_ids)

    print("TESTING IMAGES:")
    print(test_ids)

    X_train_list = []
    y_train_list = []

    X_test_list = []
    y_test_list = []

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for image_path in image_files:

        filename = os.path.basename(image_path)

        image_id = int(
            filename.replace("image_", "").replace(".h5", "")
        )

        mask_path = os.path.join(
            MASK_DIR,
            f"mask_{image_id}.h5"
        )

        if not os.path.exists(mask_path):

            print("Missing mask:", mask_path)

            continue

        image = load_h5(image_path)
        mask = load_h5(mask_path)

        features = extract_features(image)

        X = features.reshape(-1, 6)

        y = mask.reshape(-1)

        print(
            f"\nimage_{image_id}.h5 "
            f"X: {X.shape} "
            f"Landslide pixels: {np.sum(y == 1)}"
        )

        if image_id in train_ids:

            X_train_list.append(X)
            y_train_list.append(y)

        elif image_id in test_ids:

            X_test_list.append(X)
            y_test_list.append(y)

    # --------------------------------------------------------
    # COMBINE DATA
    # --------------------------------------------------------

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("Training pixels :", X_train.shape)
    print("Testing pixels  :", X_test.shape)

    print(
        "Training positive pixels:",
        np.sum(y_train == 1)
    )

    print(
        "Training negative pixels:",
        np.sum(y_train == 0)
    )

    print(
        "Testing positive pixels:",
        np.sum(y_test == 1)
    )

    print(
        "Testing negative pixels:",
        np.sum(y_test == 0)
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    print("Random Forest training completed!")

    # --------------------------------------------------------
    # SAVE RANDOM FOREST
    # --------------------------------------------------------

    rf_model_path = os.path.join(
        MODEL_DIR,
        "random_forest_landslide.pkl"
    )

    joblib.dump(rf, rf_model_path)

    print(
        "Random Forest model saved:",
        rf_model_path
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nPredicting test pixels...")

    y_pred = rf.predict(X_test)

    # Save predictions

    prediction_path = os.path.join(
        PREDICTION_DIR,
        "rf_test_predictions.npy"
    )

    np.save(
        prediction_path,
        y_pred
    )

    print(
        "RF predictions saved:",
        prediction_path
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # IoU

    intersection = np.sum(
        (y_test == 1) &
        (y_pred == 1)
    )

    union = np.sum(
        (y_test == 1) |
        (y_pred == 1)
    )

    if union == 0:
        iou = 0

    else:
        iou = intersection / union

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RANDOM FOREST TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"IoU      : {iou:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    actual_landslide = np.sum(
        y_test == 1
    )

    predicted_landslide = np.sum(
        y_pred == 1
    )

    total_pixels = len(y_test)

    actual_coverage = (
        actual_landslide /
        total_pixels *
        100
    )

    predicted_coverage = (
        predicted_landslide /
        total_pixels *
        100
    )

    print("\n")
    print("=" * 60)
    print("LANDSLIDE COVERAGE")
    print("=" * 60)

    print(
        "Actual landslide pixels    :",
        actual_landslide
    )

    print(
        "Predicted landslide pixels :",
        predicted_landslide
    )

    print(
        f"Actual coverage            : "
        f"{actual_coverage:.2f}%"
    )

    print(
        f"Predicted coverage         : "
        f"{predicted_coverage:.2f}%"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    feature_names = [
        "Blue",
        "Green",
        "Red",
        "NIR",
        "NDVI",
        "NDWI"
    ]

    for name, importance in zip(
        feature_names,
        rf.feature_importances_
    ):

        print(
            f"{name:6s}: {importance:.4f}"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RANDOM FOREST PIPELINE COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()