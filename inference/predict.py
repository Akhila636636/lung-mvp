import SimpleITK as sitk
import numpy as np
import torch
from pathlib import Path

from model.model import load_trained_model

# Load trained model once
model = load_trained_model()


def predict(_):

    # --- locate scan reliably ---
    base_dir = Path(__file__).resolve().parents[1]
    scan_path = base_dir / "test_scan" / "scan.mhd"

    print("Loading from:", scan_path)

    # --- load CT volume ---
    image = sitk.ReadImage(str(scan_path))
    volume = sitk.GetArrayFromImage(image)

    print("Volume shape:", volume.shape)
    print("Raw min/max:", volume.min(), volume.max())

    # --- HU normalization ---
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400.0

    print("Normalized min/max:", volume.min(), volume.max())

    # --- extract center 64³ patch ---
    z, y, x = volume.shape

    patch = volume[
        z//2 - 32 : z//2 + 32,
        y//2 - 32 : y//2 + 32,
        x//2 - 32 : x//2 + 32
    ]

    patch = torch.tensor(patch).float().unsqueeze(0).unsqueeze(0)

    # --- inference ---
    with torch.no_grad():
        prob = float(model(patch).item())

    # --- risk interpretation ---
    if prob < 0.3:
        risk = "Low risk"
    elif prob < 0.7:
        risk = "Medium risk"
    else:
        risk = "High risk"

    return prob, risk, volume
