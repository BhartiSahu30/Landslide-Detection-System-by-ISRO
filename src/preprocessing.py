import h5py
import numpy as np


def load_h5_image(image_path):
    """
    Load a Landslide4Sense satellite image from an HDF5 file.
    """
    with h5py.File(image_path, "r") as f:
        image = f["img"][:]

    return image


def load_h5_mask(mask_path):
    """
    Load a Landslide4Sense ground-truth mask from an HDF5 file.
    """
    with h5py.File(mask_path, "r") as f:
        mask = f["mask"][:]

    return mask


def normalize_band(band):
    """
    Min-max normalize a single image band.
    """
    band_min = np.min(band)
    band_max = np.max(band)

    if band_max - band_min == 0:
        return np.zeros_like(band)

    return (band - band_min) / (band_max - band_min)
