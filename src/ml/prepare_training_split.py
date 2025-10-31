#Build a computer-vision (CV) dataset split from the normalized products CSV.
#    - Reads products with a mapped label and a downloaded local image path
#    - Drops underrepresented classes (too few images)
#    - Caps overrepresented classes (MAX_PER_CLASS) to balance the dataset
#    - Creates a folder structure: OUT_DIR/{train,val,test}/{label}/
#    - Copies images into the split folders

import csv, os, pathlib, random, shutil
from collections import Counter

# -------------------- CONFIG --------------------
PRODUCTS = "data/processed/products_merged.csv" # CSV with 'image_url' and 'category_mapped'
IMG_ROOT = "data/images"                        # Root directory where images were saved
OUT_DIR = "data/processed/cv_split"             # Output root for split folders
SPLIT = (0.8, 0.1, 0.1)   # Proportions for train/val/test
MIN_PER_CLASS = 30        # Minimum images per class to keep that class
MAX_PER_CLASS = 500       # Cap per class to reduce class imbalance

def main():
# Read CSV -> bucket images per label -> filter/trim classes -> create folders -> copy files.
    random.seed(42)  # Reproducible sampling/shuffling
    by_label = {}

    # NEW: Statistics
    total_rows = 0
    skipped_no_label = 0
    skipped_no_image = 0
    skipped_image_not_local = 0
    skipped_image_not_exists = 0

    # 1) Reading data from CSV
    print("📂 Reading CSV...")
    with open(PRODUCTS, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1
            label = row.get("category_mapped")
            img = row.get("image_url") or ""
            
            # Checks
            if not label:
                skipped_no_label += 1
                continue
            
            if not img:
                skipped_no_image += 1
                continue
            
            # IMPROVED: checking that it is a local path
            if not img.startswith("data/images/"):
                skipped_image_not_local += 1
                continue
            
            if not os.path.exists(img):
                skipped_image_not_exists += 1
                continue
            
            by_label.setdefault(label, []).append(img)
    
    print(f"Total rows: {total_rows}")
    print(f"Skipped (no label): {skipped_no_label}")
    print(f"Skipped (no image URL): {skipped_no_image}")
    print(f"Skipped (image not local): {skipped_image_not_local}")
    print(f"Skipped (image not exists): {skipped_image_not_exists}")

    # 2) Drop classes with not many images
    print(f"\n🔍 Filtering classes (MIN_PER_CLASS={MIN_PER_CLASS})...")
    before_filter = len(by_label)
    by_label = {k: v for k, v in by_label.items() if len(v) >= MIN_PER_CLASS}
    after_filter = len(by_label)
    
    if not by_label:
        raise RuntimeError("No valid classes with enough images!")
    
    print(f"Classes before filter: {before_filter}")
    print(f"Classes after filter: {after_filter}")
    print(f"Removed: {before_filter - after_filter}")

    # 3) Apply MAX_PER_CLASS cap to reduce class imbalance, basically balancing 
    print(f"\n⚖️  Balancing classes (MAX_PER_CLASS={MAX_PER_CLASS})...")
    balanced_stats = {}
    for k in by_label:
        imgs = by_label[k]
        original_count = len(imgs)
        if len(imgs) > MAX_PER_CLASS:
            by_label[k] = random.sample(imgs, MAX_PER_CLASS)
        balanced_stats[k] = (original_count, len(by_label[k]))

    # 4) Create OUT_DIR/{train,val,test}/{label} folders
    print(f"\n📁 Creating directory structure...")
    for split in ["train", "val", "test"]:
        for label in by_label.keys():
            pathlib.Path(f"{OUT_DIR}/{split}/{label}").mkdir(parents=True, exist_ok=True)

    # 5) NEW Shuffle per-class images, compute counts, then copy into split folders
    print(f"\n🔀 Splitting into train/val/test...")
    total_copied = 0
    split_stats = {s: Counter() for s in ["train", "val", "test"]}
    
    for label, imgs in by_label.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        
        chunks = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:],
        }
        
        for split, paths in chunks.items():
            for src in paths:
                dst = f"{OUT_DIR}/{split}/{label}/{os.path.basename(src)}"
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                total_copied += 1
                split_stats[split][label] += 1

    # NEW: detailed statistics
    print("\n" + "="*70)
    print("📊 TRAINING SPLIT SUMMARY")
    print("="*70)
    print(f"Total images copied: {total_copied}")
    print(f"Total classes: {len(by_label)}")
    print(f"\n📦 Class distribution:")
    print(f"{'Class':<20} {'Original':<10} {'Balanced':<10} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-"*70)
    
    for label in sorted(by_label.keys()):
        orig, balanced = balanced_stats[label]
        train_c = split_stats["train"][label]
        val_c = split_stats["val"][label]
        test_c = split_stats["test"][label]
        print(f"{label:<20} {orig:<10} {balanced:<10} {train_c:<8} {val_c:<8} {test_c:<8}")
    
    print("-"*70)
    total_train = sum(split_stats["train"].values())
    total_val = sum(split_stats["val"].values())
    total_test = sum(split_stats["test"].values())
    
    print(f"{'TOTAL':<20} {'':<10} {'':<10} {total_train:<8} {total_val:<8} {total_test:<8}")
    print(f"{'Percentage':<20} {'':<10} {'':<10} {total_train/total_copied*100:.1f}%   {total_val/total_copied*100:.1f}%   {total_test/total_copied*100:.1f}%")
    print("="*70)
    print(f"\n✅ Prepared CV split in {OUT_DIR}")

if __name__ == "__main__":
    main()
