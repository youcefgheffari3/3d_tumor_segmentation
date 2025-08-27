# data_loader.py

import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Config


class MRIDataset(Dataset):
    """
    Custom Dataset for 3D MRI scans and segmentation masks
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load MRI volume and mask
        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image (0–1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)

        # Convert to float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # Add channel dimension (C, D, H, W) for PyTorch
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image, mask = self.transform(image, mask)

        return torch.tensor(image), torch.tensor(mask)


def get_loaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Returns PyTorch dataloaders for training & validation
    """
    train_dataset = MRIDataset(Config.TRAIN_IMAGES, Config.TRAIN_MASKS)
    val_dataset = MRIDataset(Config.VAL_IMAGES, Config.VAL_MASKS)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader = get_loaders()
    for imgs, masks in train_loader:
        print("Image batch shape:", imgs.shape)
        print("Mask batch shape:", masks.shape)
        break
