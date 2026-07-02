# prepare_dataset.py
"""
Auto-create train/val splits for a soil-type dataset.

How it works:
- Looks for an existing top-level folder that contains subfolders of class names
  (each subfolder contains images).
- If 'train' already exists, it will skip (assumes data already prepared).
- Otherwise it will copy images (not move) into 'train/<class>/' and 'val/<class>/'.

Usage:
  python prepare_dataset.py
If it can't find your dataset automatically, set RAW_DIR variable below.
"""

import os
import random
import shutil
from glob import glob

# ---------- USER SETTINGS ----------
RAW_DIR = 'soil_images' # set to path if auto-detection fails, e.g., 'dataset' or 'soil_images'
TRAIN_DIR = 'train'
VAL_DIR   = 'val'
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
TRAIN_RATIO = 0.8  # fraction for training
# -----------------------------------

def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTS)

def find_candidate_raw_dir():
    # look for likely dirs in current folder
    skip = {'venv', '__pycache__', '.git', 'backup_old_codes'}
    for name in os.listdir('.'):
        if not os.path.isdir(name) or name in skip:
            continue
        # check if this directory contains subdirectories that have image files
        subdirs = [d for d in os.listdir(name) if os.path.isdir(os.path.join(name, d))]
        if not subdirs:
            continue
        for sd in subdirs:
            files = os.listdir(os.path.join(name, sd))
            if any(is_image_file(f) for f in files):
                return name
    return None

def prepare(raw_dir):
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if not classes:
        raise SystemExit(f"No class subfolders found in {raw_dir}. Each soil-type should be a subfolder.")
    print(f"Detected class folders: {classes}")

    # create train/val folders
    for c in classes:
        os.makedirs(os.path.join(TRAIN_DIR, c), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, c), exist_ok=True)

    total_copied = 0
    for c in classes:
        class_src = os.path.join(raw_dir, c)
        imgs = [f for f in os.listdir(class_src) if is_image_file(f)]
        random.shuffle(imgs)
        n_train = int(len(imgs) * TRAIN_RATIO)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]
        print(f"Class '{c}': {len(imgs)} images -> {len(train_imgs)} train, {len(val_imgs)} val")

        for f in train_imgs:
            src = os.path.join(class_src, f)
            dst = os.path.join(TRAIN_DIR, c, f)
            shutil.copy2(src, dst)
            total_copied += 1
        for f in val_imgs:
            src = os.path.join(class_src, f)
            dst = os.path.join(VAL_DIR, c, f)
            shutil.copy2(src, dst)
            total_copied += 1

    print(f"\nDone. Copied {total_copied} images into '{TRAIN_DIR}/' and '{VAL_DIR}/'.")

if __name__ == "__main__":
    if os.path.isdir(TRAIN_DIR) and os.listdir(TRAIN_DIR):
        print(f"'{TRAIN_DIR}' already exists and is not empty. Skipping dataset creation.")
    else:
        raw = RAW_DIR or find_candidate_raw_dir()
        if raw is None:
            print("Could not auto-detect your dataset folder.")
            print("Please set RAW_DIR variable at the top of prepare_dataset.py to the folder that contains soil-type subfolders.")
            print("Example structure expected: dataset_root/alluvial/*.jpg , dataset_root/black/*.jpg , ...")
        else:
            print(f"Using raw dataset folder: {raw}")
            prepare(raw)
