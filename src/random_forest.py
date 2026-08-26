import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def prepare_pixel_data(features, mask):
    """
    Convert image features and pixel mask into
    a machine-learning dataset.
    """

    X = features.reshape(
        -1,
        features.shape[-1]
    )

    y = mask.reshape(-1)

    return X, y


def split_pixel_data(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """
    Split pixels into training and testing subsets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    random_state=42
):
    """
    Train Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def predict_random_forest(
    model,
    features
):
    """
    Generate a complete 2D prediction map.
    """

    original_shape = features.shape[:2]

    X = features.reshape(
        -1,
        features.shape[-1]
    )

    predictions = model.predict(X)

    prediction_map = predictions.reshape(
        original_shape
    ).astype(np.uint8)

    return prediction_map
