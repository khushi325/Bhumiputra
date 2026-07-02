# show_misclassified.py
import os
from PIL import Image
import matplotlib.pyplot as plt
import json

MIS_DIR = "misclassified"    # Folder created during evaluation
CLASS_MAP_FILE = "class_indices.json"
N = 20   # Number of misclassified examples to display at once

# Load class index mapping
with open(CLASS_MAP_FILE, 'r') as f:
    class_map = json.load(f)
idx_to_label = {v: k for k, v in class_map.items()}

files = sorted([f for f in os.listdir(MIS_DIR) if os.path.isfile(os.path.join(MIS_DIR,f))])
files = files[:N]

if not files:
    print("⚠️ No misclassified images found in", MIS_DIR)
    exit()

cols = 5
rows = (len(files)+cols-1)//cols
plt.figure(figsize=(4*cols, 3*rows))

for i, fname in enumerate(files):
    path = os.path.join(MIS_DIR, fname)

    # Parse filename to extract true/predicted labels
    parts = fname.split("__")[0]  # e.g. "true2_pred0"
    true_idx = int(parts.split("_")[0].replace("true",""))
    pred_idx = int(parts.split("_")[1].replace("pred",""))

    true_label = idx_to_label.get(true_idx, str(true_idx))
    pred_label = idx_to_label.get(pred_idx, str(pred_idx))

    img = Image.open(path).convert("RGB")

    plt.subplot(rows, cols, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"T: {true_label}\nP: {pred_label}")

plt.tight_layout()
plt.show()
