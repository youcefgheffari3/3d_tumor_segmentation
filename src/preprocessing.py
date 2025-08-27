# preprocessing.py
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, IMG_SIZE


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normalize MRI scan intensities to [0, 1].
    """
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def resize_image(image: np.ndarray, target_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Resize 3D image to target size using interpolation.
    """
    factors = [t / s for t, s in zip(target_size, image.shape)]
    return zoom(image, factors, order=1)  # linear interpolation


def preprocess_scan(scan_path: str, save_path: str):
    """
    Load, preprocess, and save one MRI scan.
    """
    # Load NIfTI file
    scan = nib.load(scan_path).get_fdata()

    # Normalize intensity
    scan = normalize_intensity(scan)

    # Resize to target shape
    scan = resize_image(scan, IMG_SIZE)

    # Save as numpy file
    np.save(save_path, scan)
    print(f"Saved preprocessed scan to {save_path}")


def preprocess_dataset():
    """
    Preprocess all MRI scans in RAW_DATA_DIR and save them into PREPROCESSED_DATA_DIR.
    """
    if not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            scan_path = os.path.join(RAW_DATA_DIR, filename)
            save_name = filename.replace(".nii.gz", ".npy").replace(".nii", ".npy")
            save_path = os.path.join(PREPROCESSED_DATA_DIR, save_name)

            preprocess_scan(scan_path, save_path)


if __name__ == "__main__":
    preprocess_dataset()
