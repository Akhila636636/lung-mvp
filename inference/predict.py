import SimpleITK as sitk
import numpy as np
import torch
from pathlib import Path
from scipy import ndimage

from model.model import load_trained_model

# load model once
model = load_trained_model()


def detect_nodules(volume):
    """
    Simple heuristic nodule detection.
    Returns a binary mask of suspicious regions.
    """

    # threshold dense regions
    mask = volume > 0.6

    # connected components cleanup
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(num + 1))

    clean_mask = np.zeros_like(mask)

    for i in range(1, num + 1):
        if sizes[i] > 100:  # minimum blob size
            clean_mask[labeled == i] = 1

    return clean_mask


def predict(_):

    # --- resolve scan path FIRST ---
    base_dir = Path(__file__).resolve().parents[1]
    scan_path = base_dir / "test_scan" / "scan.mhd"

    print("Loading scan from:", scan_path)
    print("Exists:", scan_path.exists())

    if not scan_path.exists():
        raise FileNotFoundError(f"Scan not found: {scan_path}")

    # --- load CT volume ---
    image = sitk.ReadImage(str(scan_path))
    volume = sitk.GetArrayFromImage(image)

    print("Volume shape:", volume.shape)

    # --- normalize HU ---
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400.0

    # --- extract center patch for classification ---
    z, y, x = volume.shape

    patch = volume[
        z // 2 - 32: z // 2 + 32,
        y // 2 - 32: y // 2 + 32,
        x // 2 - 32: x // 2 + 32
    ]

    patch = torch.tensor(patch).float().unsqueeze(0).unsqueeze(0)

    # --- inference ---
    with torch.no_grad():
        prob = float(model(patch).item())

    if prob < 0.3:
        risk = "Low risk"
    elif prob < 0.7:
        risk = "Medium risk"
    else:
        risk = "High risk"

    # --- detect nodules ---
    nodule_mask = detect_nodules(volume)

    return prob, risk, volume, nodule_mask
