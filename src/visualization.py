# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt


def show_slice(mri_slice, mask_slice=None, pred_slice=None, save_path=None, title="MRI Slice"):
    """
    Display an MRI slice with optional ground-truth and predicted masks.

    Args:
        mri_slice (numpy array): 2D MRI slice.
        mask_slice (numpy array, optional): 2D ground-truth segmentation mask.
        pred_slice (numpy array, optional): 2D predicted segmentation mask.
        save_path (str, optional): Path to save the figure. If None, will display instead.
        title (str): Title of the figure.
    """
    fig, axs = plt.subplots(1, 3 if (mask_slice is not None or pred_slice is not None) else 1, figsize=(12, 4))

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # MRI
    axs[0].imshow(mri_slice, cmap="gray")
    axs[0].set_title("MRI")
    axs[0].axis("off")

    # Ground truth mask
    if mask_slice is not None:
        axs[1].imshow(mri_slice, cmap="gray")
        axs[1].imshow(mask_slice, cmap="Reds", alpha=0.5)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

    # Prediction mask
    if pred_slice is not None:
        idx = 2 if mask_slice is not None else 1
        axs[idx].imshow(mri_slice, cmap="gray")
        axs[idx].imshow(pred_slice, cmap="Blues", alpha=0.5)
        axs[idx].set_title("Prediction")
        axs[idx].axis("off")

    plt.suptitle(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' lists.
        save_path (str, optional): Path to save the figure. If None, will display instead.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
