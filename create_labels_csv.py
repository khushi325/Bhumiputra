import os
import csv

# ---------------- PARAMETERS ----------------
train_dir = 'train'  # folder with training images
val_dir = 'val'      # folder with validation images
# --------------------------------------------

def create_csv(folder, csv_name):
    data = []
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            file_lower = file.lower()
            if 'healthy' in file_lower:
                label = 'healthy'
            elif 'unhealthy' in file_lower:
                label = 'unhealthy'
            else:
                print(f"⚠️ Skipped (no class match): {file}")
                continue
            data.append([file, label])
    
    # Write to CSV
    with open(csv_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(data)
    print(f"✅ CSV file created: {csv_name} ({len(data)} entries)")

# Create CSV for train and val
create_csv(train_dir, 'train_labels.csv')
create_csv(val_dir, 'val_labels.csv')
