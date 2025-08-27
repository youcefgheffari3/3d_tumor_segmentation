# evaluate.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from config import CONFIG
from data_loader import BrainTumorDataset
from model import UNet3D


def dice_coefficient(pred, target, epsilon=1e-6):
    """Compute Dice coefficient between predicted and target masks."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)


def evaluate(model, dataloader, device):
    model.eval()
    dice_scores = []
    accuracies = []
    ious = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Dice
            dice = dice_coefficient(preds, masks)
            dice_scores.append(dice.item())

            # Accuracy
            correct = (preds == masks).sum().item()
            total = masks.numel()
            accuracies.append(correct / total)

            # IoU
            intersection = (preds & masks).float().sum((1, 2, 3, 4))
            union = (preds | masks).float().sum((1, 2, 3, 4))
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.extend(iou.cpu().numpy())

    return {
        "Dice": np.mean(dice_scores),
        "Accuracy": np.mean(accuracies),
        "IoU": np.mean(ious)
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Loader
    test_dataset = BrainTumorDataset(CONFIG["test_images"], CONFIG["test_masks"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(CONFIG["save_path"], map_location=device))

    # Evaluation
    metrics = evaluate(model, test_loader, device)
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
