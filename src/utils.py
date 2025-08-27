# utils.py
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Compute Dice Similarity Coefficient (DSC) between prediction and ground truth.
    Both arrays should be binary masks (0 and 1).
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def save_checkpoint(state: dict, filename: str):
    """
    Save model checkpoint.
    Args:
        state: dictionary containing model and optimizer state
        filename: path to save
    """
    torch.save(state, filename)
    print(f"[INFO] Checkpoint saved at {filename}")


def load_checkpoint(filename: str, model: torch.nn.Module, optimizer=None):
    """
    Load model checkpoint.
    Args:
        filename: checkpoint path
        model: model object
        optimizer: optimizer object (optional)
    """
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[INFO] Checkpoint loaded from {filename}")
    return model, optimizer
