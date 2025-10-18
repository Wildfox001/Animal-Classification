import os
import shutil
import random

# Path to the folder containing the original cattle/buffalo images
dataset_dir = "dataset"  # <-- adjust only if your folders are somewhere else

# Classes (detect automatically)
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
print(f"Detected classes: {classes}")

# Base directory for train/val/test
base_dir = "dataset"

# Create train/val/test folders
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Split and copy images
for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    images = os.listdir(cls_path)
    images = [img for img in images if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(base_dir, "train", cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(base_dir, "val", cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(base_dir, "test", cls, img))

print("Dataset split completed!")
