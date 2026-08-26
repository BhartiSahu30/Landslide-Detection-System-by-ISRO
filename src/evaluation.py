import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate standard binary classification
    and segmentation metrics.
    """

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    results = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "iou": jaccard_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }

    return results


def print_metrics(name, metrics):
    """
    Print model evaluation results.
    """

    print(f"\n{name}")
    print("=" * len(name))

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1-score : {metrics['f1']:.4f}"
    )

    print(
        f"IoU      : {metrics['iou']:.4f}"
    )


def calculate_coverage(prediction_map):
    """
    Calculate the percentage of pixels
    predicted as landslide.
    """

    total_pixels = prediction_map.size

    landslide_pixels = np.sum(
        prediction_map == 1
    )

    percentage = (
        landslide_pixels /
        total_pixels
    ) * 100

    return landslide_pixels, percentage


def calculate_confusion_matrix(y_true, y_pred):
    """
    Return the binary confusion matrix.
    """

    return confusion_matrix(
        np.asarray(y_true).flatten(),
        np.asarray(y_pred).flatten()
    )
