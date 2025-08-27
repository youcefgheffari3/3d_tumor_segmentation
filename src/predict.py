# predict.py
import torch
import nibabel as nib
import numpy as np
import os
from model import UNet3D
from config import MODEL_SAVE_PATH, DEVICE, NUM_CLASSES
from preprocessing import normalize_volume, resize_volume

def load_model(model_path=MODEL_SAVE_PATH):
    """Load trained 3D U-Net model."""
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def predict_single_volume(model, volume_path):
    """Run segmentation prediction on a single 3D MRI volume."""
    # Load NIfTI file
    img = nib.load(volume_path)
    volume = img.get_fdata()

    # Preprocess: normalize + resize
    volume = normalize_volume(volume)
    volume = resize_volume(volume, target_shape=(128, 128, 128))

    # Convert to tensor
    volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    # Shape: (1, 1, D, H, W)

    with torch.no_grad():
        output = model(volume_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()  # shape: (1, D, H, W)

    return pred[0], img.affine

def calculate_tumor_volume(pred_mask, voxel_spacing=(1, 1, 1)):
    """Calculate tumor volume in cubic millimeters from predicted mask."""
    tumor_voxels = np.sum(pred_mask == 1)  # class "1" = tumor
    voxel_volume = np.prod(voxel_spacing)  # mm^3 per voxel
    tumor_volume = tumor_voxels * voxel_volume
    return tumor_volume

def save_prediction(pred_mask, affine, output_path):
    """Save prediction mask as NIfTI file."""
    pred_img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine)
    nib.save(pred_img, output_path)
    print(f"[INFO] Prediction saved at: {output_path}")

if __name__ == "__main__":
    # Example usage
    test_volume_path = "data/test/volume.nii.gz"   # <- put your MRI file here
    output_mask_path = "outputs/pred_mask.nii.gz"

    os.makedirs("outputs", exist_ok=True)

    model = load_model()
    pred_mask, affine = predict_single_volume(model, test_volume_path)

    # Save prediction
    save_prediction(pred_mask, affine, output_mask_path)

    # Calculate tumor volume (assuming voxel size of 1mm^3, update if metadata differs)
    tumor_volume = calculate_tumor_volume(pred_mask, voxel_spacing=(1,1,1))
    print(f"[RESULT] Estimated Tumor Volume: {tumor_volume:.2f} mm³")
