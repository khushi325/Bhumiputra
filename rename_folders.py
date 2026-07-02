import os

# Path to your dataset
BASE_DIR = "soil_images"

# Folders to process
SUBFOLDERS = ["Train", "val"]

for subfolder in SUBFOLDERS:
    folder_path = os.path.join(BASE_DIR, subfolder)

    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping: {folder_path} (not found)")
        continue

    # List all class subfolders
    for class_name in os.listdir(folder_path):
        old_path = os.path.join(folder_path, class_name)

        if os.path.isdir(old_path):
            # Make lowercase + replace spaces with underscores
            new_name = class_name.lower().replace(" ", "_")
            new_path = os.path.join(folder_path, new_name)

            # Rename
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {old_path} → {new_path}")

# Finally, rename "Train" to "train"
old_train = os.path.join(BASE_DIR, "Train")
new_train = os.path.join(BASE_DIR, "train")
if os.path.exists(old_train):
    os.rename(old_train, new_train)
    print(f"✅ Renamed: {old_train} → {new_train}")

print("\n🎉 All folder names cleaned up!")
