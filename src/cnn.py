import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_segmentation_model(input_shape=(16, 16, 6)):
    """
    Build the lightweight CNN segmentation model
    used in the Landslide4Sense experiment.

    Input:
        16 x 16 satellite patch with 6 features.

    Output:
        16 x 16 pixel-level landslide probability mask.
    """

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.Conv2D(
            1,
            (1, 1),
            activation="sigmoid",
            padding="same"
        )
    ])

    return model


def calculate_positive_class_weight(masks):
    """
    Calculate positive-class weight for landslide pixels.

    This compensates for the strong imbalance between
    non-landslide and landslide pixels.
    """

    masks = np.asarray(masks)

    negative_pixels = np.sum(masks == 0)
    positive_pixels = np.sum(masks == 1)

    if positive_pixels == 0:
        return 1.0

    return negative_pixels / positive_pixels


def weighted_binary_crossentropy(positive_weight):
    """
    Weighted binary cross-entropy loss for
    imbalanced landslide segmentation.
    """

    def loss(y_true, y_pred):

        y_true = tf.cast(
            y_true,
            tf.float32
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        loss_value = -(
            positive_weight * y_true *
            tf.math.log(y_pred)
            +
            (1.0 - y_true) *
            tf.math.log(1.0 - y_pred)
        )

        return tf.reduce_mean(
            loss_value
        )

    return loss


def compile_segmentation_model(
    model,
    positive_weight
):
    """
    Compile the CNN using Adam optimizer
    and weighted binary cross-entropy.
    """

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss=weighted_binary_crossentropy(
            positive_weight
        ),
        metrics=["accuracy"]
    )

    return model


def normalize_patches(X_train, X_test):
    """
    Normalize training and testing patches using
    training-set statistics only.
    """

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

    X_train_norm = (
        X_train - mean
    ) / std

    X_test_norm = (
        X_test - mean
    ) / std

    return (
        X_train_norm,
        X_test_norm,
        mean,
        std
    )


def predict_segmentation(
    model,
    patches,
    threshold=0.5
):
    """
    Predict pixel-level landslide probabilities
    and convert them to binary masks.
    """

    probabilities = model.predict(
        patches,
        verbose=0
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    return predictions, probabilities


def reconstruct_prediction_map(
    patch_predictions,
    image_height=128,
    image_width=128,
    patch_size=16
):
    """
    Reconstruct the complete 128x128 prediction
    map from 16x16 non-overlapping patches.
    """

    prediction_map = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    num_patches_h = (
        image_height // patch_size
    )

    num_patches_w = (
        image_width // patch_size
    )

    patch_index = 0

    for i in range(num_patches_h):

        for j in range(num_patches_w):

            row_start = i * patch_size
            row_end = row_start + patch_size

            col_start = j * patch_size
            col_end = col_start + patch_size

            prediction_map[
                row_start:row_end,
                col_start:col_end
            ] = patch_predictions[
                patch_index, :, :, 0
            ]

            patch_index += 1

    return prediction_map
