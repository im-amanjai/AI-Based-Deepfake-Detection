import os
import shutil
import random

RAW_DIR = "raw_data"
BASE_DIR = "dataset"

SPLITS = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15
}

CLASSES = ["real", "fake"]

for cls in CLASSES:
    files = os.listdir(os.path.join(RAW_DIR, cls))
    random.shuffle(files)

    total = len(files)
    train_end = max(1, int(0.7 * total))
    val_end = max(train_end + 1, int(0.85 * total))


    split_files = {
        "train": files[:train_end],
        "validation": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_files_list in split_files.items():
        split_path = os.path.join(BASE_DIR, split, cls)
        os.makedirs(split_path, exist_ok=True)

        for file in split_files_list:
            src = os.path.join(RAW_DIR, cls, file)
            dst = os.path.join(split_path, file)
            shutil.copy(src, dst)

print("Dataset split completed successfully.")
