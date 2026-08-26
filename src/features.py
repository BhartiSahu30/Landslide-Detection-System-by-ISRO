import numpy as np


def calculate_ndvi(red, nir):
    """
    Calculate Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)
    """
    return (nir - red) / (nir + red + 1e-8)


def calculate_ndwi(green, nir):
    """
    Calculate Normalized Difference Water Index.

    NDWI = (Green - NIR) / (Green + NIR)
    """
    return (green - nir) / (green + nir + 1e-8)


def extract_spectral_features(image):
    """
    Extract selected spectral bands and derived indices.

    Expected input:
        image shape = (H, W, 14)

    Selected bands:
        B2 = Blue
        B3 = Green
        B4 = Red
        B8 = NIR

    Returns:
        features shape = (H, W, 6)
    """

    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    nir = image[:, :, 3]

    ndvi = calculate_ndvi(red, nir)
    ndwi = calculate_ndwi(green, nir)

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
