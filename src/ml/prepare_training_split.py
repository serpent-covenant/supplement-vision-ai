#Build a computer-vision (CV) dataset split from the normalized products CSV.
#    - Reads products with a mapped label and a downloaded local image path
#    - Drops underrepresented classes (too few images)
#    - Caps overrepresented classes (MAX_PER_CLASS) to balance the dataset
#    - Creates a folder structure: OUT_DIR/{train,val,test}/{label}/
#    - Copies images into the split folders

import csv, os, pathlib, random, shutil

# -------------------- CONFIG --------------------
PRODUCTS = "data/processed/products_merged.csv" # CSV with 'image_url' and 'category_mapped'
IMG_ROOT = "data/images"                        # Root directory where images were saved
OUT_DIR = "data/processed/cv_split"             # Output root for split folders
SPLIT = (0.8, 0.1, 0.1)   # Proportions for train/val/test
MIN_PER_CLASS = 10        # Minimum images per class to keep that class
MAX_PER_CLASS = 150       # Cap per class to reduce class imbalance

def main():
# Read CSV -> bucket images per label -> filter/trim classes -> create folders -> copy files.
    random.seed(42)  # Reproducible sampling/shuffling
    by_label = {}

    # 1) Read rows from CSV and collect image paths grouped by label
    with open(PRODUCTS, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            label = row.get("category_mapped")
            img = row.get("image_url") or ""
            if not label or not img: # Skip records without a label or without an image path
                continue
            if not img.startswith("data/images/") or not os.path.exists(img):
                continue  # skip images that weren't downloaded or keep only images that were actually downloaded (local paths under IMG_ROOT)
            by_label.setdefault(label, []).append(img)

    # 2) Drop classes with too few images
    by_label = {k: v for k, v in by_label.items() if len(v) >= MIN_PER_CLASS}

    if not by_label:
        raise RuntimeError("No valid classes with enough images.")

    # 3) Apply MAX_PER_CLASS cap to reduce class imbalance, basically balancing 
    for k in by_label:
        imgs = by_label[k]
        if len(imgs) > MAX_PER_CLASS:
            by_label[k] = random.sample(imgs, MAX_PER_CLASS)

    # 4) Create OUT_DIR/{train,val,test}/{label} folders
    for split in ["train", "val", "test"]:
        for label in by_label.keys():
            pathlib.Path(f"{OUT_DIR}/{split}/{label}").mkdir(parents=True, exist_ok=True)
    # 5) Shuffle per-class images, compute counts, then copy into split folders
    total = 0
    for label, imgs in by_label.items():
        random.shuffle(imgs)
        n = len(imgs)
        # Compute split counts (integer floor); leftover goes to 'test'
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        chunks = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:],
        }
        # Copy files (use copy2 to preserve timestamps/metadata)
        for split, paths in chunks.items():
            for src in paths:
                dst = f"{OUT_DIR}/{split}/{label}/{os.path.basename(src)}"
                if not os.path.exists(dst):  # Do not overwrite if the file already exists
                    shutil.copy2(src, dst)
                total += 1

    print(f"Prepared CV split in {OUT_DIR} (files: {total}, classes: {len(by_label)})")

if __name__ == "__main__":
    main()
