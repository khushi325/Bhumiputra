import os
import shutil

# Paths to your train and val folders
train_dir = 'train'
val_dir = 'val'

classes = ['healthy', 'unhealthy']  # class names

def organize(folder):
    for c in classes:
        class_folder = os.path.join(folder, c)
        os.makedirs(class_folder, exist_ok=True)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            # Check file name to decide class
            if 'healthy' in file.lower():
                shutil.move(file_path, os.path.join(folder, 'healthy', file))
            elif 'unhealthy' in file.lower():
                shutil.move(file_path, os.path.join(folder, 'unhealthy', file))
            else:
                print(f"⚠️ Skipped (no class match): {file}")

# Organize train and val folders
organize(train_dir)
organize(val_dir)

print("✅ Images organized into class subfolders!")
print("Folder structure should now be like:")
print("train/healthy, train/unhealthy")
print("val/healthy, val/unhealthy")
