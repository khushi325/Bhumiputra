import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import shutil

# Paths
MIS_DIR = "misclassified"
DATA_DIR = "data/soil_images/Train"   # Change if needed
CLASS_MAP_FILE = "class_indices.json"

# Load class map
with open(CLASS_MAP_FILE, "r") as f:
    class_map = json.load(f)
idx_to_label = {v: k for k, v in class_map.items()}
label_to_idx = {k: v for v, k in idx_to_label.items()}

# List of misclassified files
files = sorted([f for f in os.listdir(MIS_DIR) if os.path.isfile(os.path.join(MIS_DIR, f))])

print("Found", len(files), "misclassified images.")

for fname in files:
    path = os.path.join(MIS_DIR, fname)

    # Show image
    img = Image.open(path).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.title(fname)
    plt.show(block=False)

    # Ask user to pick correct class
    print("\nChoose the correct class for this image:")
    for i, label in idx_to_label.items():
        print(f"[{i}] {label}")
    choice = input("Enter number (or press Enter to skip): ")

    plt.close()

    if choice.strip() == "":
        print("⏭️ Skipped\n")
        continue

    try:
        choice = int(choice)
        correct_class = idx_to_label[choice]

        # Move file to correct folder
        dest_folder = os.path.join(DATA_DIR, correct_class)
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, fname)
        shutil.move(path, dest_path)

        print(f"✅ Moved {fname} → {dest_folder}\n")

    except Exception as e:
        print("⚠️ Invalid choice:", e, "\n")
