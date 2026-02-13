"""
preprocessing.py — LUNA16 CT‑scan preprocessing for lung‑nodule detection MVP.

Pipeline:
  1. Unzip subset archives → raw .mhd/.raw pairs
  2. Load annotations.csv
  3. For each scan that has annotated nodules:
       a. Load the .mhd volume with SimpleITK
       b. Convert world coords → voxel coords
       c. Extract 64×64×64 patches centred on each nodule (positive)
       d. Extract one random negative patch per positive
  4. Clip HU to [‑1000, 400], normalise to [0, 1]
  5. Save as processed_data.npz  (X: (N,1,64,64,64), y: (N,))

Usage:
    python preprocessing/preprocessing.py
"""

import os
import sys
import glob
import time
import random

import numpy as np
import pandas as pd
import SimpleITK as sitk

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Where extracted .mhd/.raw files live (already unzipped)
DATA_DIR      = os.path.join(SCRIPT_DIR, "subset0 (1)")
# Annotations CSV path
ANNOTATIONS   = os.path.join(SCRIPT_DIR, "annotations.csv")
# Output file
OUTPUT_FILE   = os.path.join(SCRIPT_DIR, "processed_data.npz")

PATCH_SIZE    = 64          # px per axis
HU_MIN        = -1000
HU_MAX        = 400
MAX_SAMPLES   = 200         # prototype cap – first 200 positive samples


# ──────────────────────────────────────────────
# 1. Unzip helpers
# ──────────────────────────────────────────────
def unzip_subsets(zip_dir: str, dest_dir: str) -> None:
    """Extract every subset*.zip found in *zip_dir* into *dest_dir*."""
    os.makedirs(dest_dir, exist_ok=True)

    zips = sorted(glob.glob(os.path.join(zip_dir, "subset*.zip")))
    if not zips:
        print(f"[WARN] No subset*.zip found in {zip_dir}")
        return

    for zp in zips:
        basename = os.path.basename(zp)
        print(f"[UNZIP] Extracting {basename} → {dest_dir} ...")
        t0 = time.time()
        with zipfile.ZipFile(zp, "r") as z:
            z.extractall(dest_dir)
        elapsed = time.time() - t0
        print(f"[UNZIP] {basename} done in {elapsed:.1f}s")


# ──────────────────────────────────────────────
# 2. Load annotations
# ──────────────────────────────────────────────
def load_annotations(csv_path: str) -> pd.DataFrame:
    """Read LUNA16 annotations.csv and return a DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"[DATA]  Loaded {len(df)} annotations from {os.path.basename(csv_path)}")
    print(f"[DATA]  Unique series: {df['seriesuid'].nunique()}")
    return df


# ──────────────────────────────────────────────
# 3. Load a CT volume
# ──────────────────────────────────────────────
def load_ct_scan(mhd_path: str):
    """
    Load a .mhd file with SimpleITK.

    Returns
    -------
    image_array : np.ndarray   (Z, Y, X)  — raw HU values
    origin      : np.ndarray   (3,)
    spacing     : np.ndarray   (3,)
    """
    itk_img = sitk.ReadImage(mhd_path)
    image_array = sitk.GetArrayFromImage(itk_img)       # shape: (Z, Y, X)
    origin      = np.array(itk_img.GetOrigin())          # (X, Y, Z)
    spacing     = np.array(itk_img.GetSpacing())         # (X, Y, Z)
    return image_array, origin, spacing


# ──────────────────────────────────────────────
# 4. World → voxel coordinate conversion
# ──────────────────────────────────────────────
def world_to_voxel(world_coord: np.ndarray,
                   origin: np.ndarray,
                   spacing: np.ndarray) -> np.ndarray:
    """
    Convert physical‑world coordinate to voxel index.

    Parameters
    ----------
    world_coord : (3,)  — (X, Y, Z) in mm
    origin      : (3,)  — image origin (X, Y, Z)
    spacing     : (3,)  — voxel spacing (X, Y, Z)

    Returns
    -------
    voxel : (3,)  — integer voxel indices (Z, Y, X) matching numpy axis order
    """
    stretched = np.abs(world_coord - origin) / spacing   # (X, Y, Z)
    voxel = np.round(stretched).astype(int)
    # Flip to numpy axis order: (Z, Y, X)
    voxel = voxel[::-1]
    return voxel


# ──────────────────────────────────────────────
# 5. Patch extraction
# ──────────────────────────────────────────────
def extract_patch(volume: np.ndarray,
                  centre_zyx: np.ndarray,
                  size: int = PATCH_SIZE) -> np.ndarray | None:
    """
    Extract a cubic patch of shape (size, size, size) centred on *centre_zyx*.

    Handles boundary clipping: if the patch would exceed the volume, it is
    zero‑padded so the output always has the requested shape.

    Returns None only if the centre is completely outside the volume.
    """
    half = size // 2
    z, y, x = centre_zyx

    # Absolute start / end in volume space
    z0, z1 = z - half, z + half
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    # Check if centre is wildly outside the volume
    dz, dy, dx = volume.shape
    if z < -half or z >= dz + half or \
       y < -half or y >= dy + half or \
       x < -half or x >= dx + half:
        return None

    # Clamp to valid volume range
    vz0, vz1 = max(z0, 0), min(z1, dz)
    vy0, vy1 = max(y0, 0), min(y1, dy)
    vx0, vx1 = max(x0, 0), min(x1, dx)

    patch = np.zeros((size, size, size), dtype=np.float32)

    # Offsets inside the patch tensor
    pz0 = vz0 - z0
    py0 = vy0 - y0
    px0 = vx0 - x0
    pz1 = pz0 + (vz1 - vz0)
    py1 = py0 + (vy1 - vy0)
    px1 = px0 + (vx1 - vx0)

    patch[pz0:pz1, py0:py1, px0:px1] = volume[vz0:vz1, vy0:vy1, vx0:vx1]
    return patch


def random_negative_patch(volume: np.ndarray,
                          positive_centres: list[np.ndarray],
                          size: int = PATCH_SIZE,
                          min_distance: int = 80,
                          max_tries: int = 50) -> np.ndarray | None:
    """
    Sample a random patch that is at least *min_distance* voxels away
    from every positive centre.  Returns None after *max_tries* failures.
    """
    dz, dy, dx = volume.shape
    half = size // 2
    for _ in range(max_tries):
        cz = random.randint(half, max(half, dz - half - 1))
        cy = random.randint(half, max(half, dy - half - 1))
        cx = random.randint(half, max(half, dx - half - 1))
        candidate = np.array([cz, cy, cx])
        if all(np.linalg.norm(candidate - pc) > min_distance
               for pc in positive_centres):
            return extract_patch(volume, candidate, size)
    return None


# ──────────────────────────────────────────────
# 6. HU clipping & normalisation
# ──────────────────────────────────────────────
def normalise_hu(patch: np.ndarray) -> np.ndarray:
    """Clip HU to [HU_MIN, HU_MAX] and scale to [0, 1]."""
    patch = np.clip(patch, HU_MIN, HU_MAX).astype(np.float32)
    patch = (patch - HU_MIN) / (HU_MAX - HU_MIN)
    return patch


# ──────────────────────────────────────────────
# 7. Main pipeline
# ──────────────────────────────────────────────
def run_pipeline() -> None:
    t_start = time.time()
    print("=" * 60)
    print(" LUNA16 Preprocessing Pipeline — Prototype")
    print("=" * 60)

    # ── Step 1: Check data directory ─────────
    print(f"\n[STEP 1/5] Checking data directory: {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        sys.exit(1)
    print(f"[OK]    Data directory exists.")

    # ── Step 2: Load annotations ─────────────
    print("\n[STEP 2/5] Loading annotations ...")
    annotations = load_annotations(ANNOTATIONS)

    # ── Step 3: Discover available .mhd files ─
    print("\n[STEP 3/5] Discovering .mhd files ...")
    mhd_files = glob.glob(os.path.join(DATA_DIR, "**", "*.mhd"), recursive=True)
    uid_to_path: dict[str, str] = {}
    for f in mhd_files:
        uid = os.path.splitext(os.path.basename(f))[0]
        uid_to_path[uid] = f
    print(f"[DATA]  Found {len(uid_to_path)} .mhd files on disk")

    # Match annotations to available files
    available_uids = set(uid_to_path.keys())
    matched = annotations[annotations["seriesuid"].isin(available_uids)]
    print(f"[DATA]  Annotations matched to available scans: {len(matched)}")

    if matched.empty:
        print("[ERROR] No annotation matches any .mhd file. "
              "Check that the zip files extracted correctly.")
        sys.exit(1)

    # ── Step 4: Extract patches ──────────────
    print(f"\n[STEP 4/5] Extracting patches (limit: {MAX_SAMPLES} positive) ...")
    X_patches: list[np.ndarray] = []
    y_labels:  list[int]        = []
    positives_extracted = 0

    grouped = matched.groupby("seriesuid")
    total_series = len(grouped)

    for idx, (uid, group) in enumerate(grouped, start=1):
        if positives_extracted >= MAX_SAMPLES:
            print(f"[INFO]  Reached {MAX_SAMPLES} positive samples — stopping early.")
            break

        mhd_path = uid_to_path[uid]
        print(f"\n[SCAN {idx}/{total_series}] Loading {uid[:40]}...")

        try:
            volume, origin, spacing = load_ct_scan(mhd_path)
        except Exception as exc:
            print(f"  [WARN] Failed to load: {exc}")
            continue

        print(f"  Volume shape: {volume.shape}  |  "
              f"spacing: {np.round(spacing, 3)}  |  "
              f"nodules in scan: {len(group)}")

        # Collect positive centres for negative sampling
        pos_centres: list[np.ndarray] = []

        for _, row in group.iterrows():
            if positives_extracted >= MAX_SAMPLES:
                break

            world = np.array([row["coordX"], row["coordY"], row["coordZ"]])
            voxel = world_to_voxel(world, origin, spacing)

            patch = extract_patch(volume, voxel, PATCH_SIZE)
            if patch is None:
                print(f"  [SKIP] Nodule at voxel {voxel} is out of bounds")
                continue

            patch = normalise_hu(patch)
            X_patches.append(patch)
            y_labels.append(1)
            pos_centres.append(voxel)
            positives_extracted += 1

            print(f"  [+] Positive #{positives_extracted}  "
                  f"world=({row['coordX']:.1f}, {row['coordY']:.1f}, "
                  f"{row['coordZ']:.1f})  voxel={voxel}  "
                  f"diameter={row['diameter_mm']:.2f}mm")

        # Negative patches — one per positive in this scan
        neg_count = 0
        for _ in pos_centres:
            neg = random_negative_patch(volume, pos_centres, PATCH_SIZE)
            if neg is not None:
                neg = normalise_hu(neg)
                X_patches.append(neg)
                y_labels.append(0)
                neg_count += 1
        if neg_count:
            print(f"  [−] Added {neg_count} negative patches")

    print(f"\n[PROGRESS] Total patches: {len(X_patches)}  "
          f"(positives={sum(y_labels)}, negatives={len(y_labels)-sum(y_labels)})")

    # ── Step 5: Build arrays & save ──────────
    print(f"\n[STEP 5/5] Building arrays and saving to {os.path.basename(OUTPUT_FILE)} ...")

    X = np.stack(X_patches, axis=0)          # (N, 64, 64, 64)
    X = X[:, np.newaxis, :, :, :]            # (N, 1, 64, 64, 64)
    y = np.array(y_labels, dtype=np.int64)   # (N,)

    print(f"  X shape : {X.shape}   dtype: {X.dtype}")
    print(f"  y shape : {y.shape}   dtype: {y.dtype}")
    print(f"  X range : [{X.min():.4f}, {X.max():.4f}]")

    np.savez_compressed(OUTPUT_FILE, X=X, y=y)
    file_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  Saved  → {OUTPUT_FILE}  ({file_mb:.1f} MB)")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f" Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
