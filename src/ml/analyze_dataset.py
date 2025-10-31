"""
Скрипт для аналізу датасету перед тренуванням.
Показує розподіл класів, якість даних, можливі проблеми.

Usage: python src/ml/analyze_dataset.py
"""

import os
import csv
from collections import Counter
from pathlib import Path

PRODUCTS_CSV = "data/processed/products_merged.csv"
IMAGES_DIR = "data/images"
CV_SPLIT_DIR = "data/processed/cv_split"

def analyze_csv():
    """Аналізує CSV з продуктами"""
    print("="*70)
    print("📊 DATASET ANALYSIS")
    print("="*70)
    
    if not os.path.exists(PRODUCTS_CSV):
        print(f"❌ CSV not found: {PRODUCTS_CSV}")
        print("   Run: python src/ingestion/openfoodfacts_ingest.py")
        return
    
    total = 0
    with_images = 0
    local_images = 0
    mapped = 0
    categories = Counter()
    sources = Counter()
    brands = Counter()
    
    with open(PRODUCTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            
            img_url = row.get("image_url", "")
            if img_url:
                with_images += 1
                if img_url.startswith("data/images/"):
                    local_images += 1
            
            cat = row.get("category_mapped")
            if cat:
                mapped += 1
                categories[cat] += 1
            
            sources[row.get("source", "unknown")] += 1
            
            brand = row.get("brand")
            if brand:
                brands[brand] += 1
    
    print(f"\n📦 Products:")
    print(f"  Total: {total}")
    print(f"  With images: {with_images} ({with_images/total*100:.1f}%)")
    print(f"  Local images: {local_images} ({local_images/total*100:.1f}%)")
    print(f"  Mapped categories: {mapped} ({mapped/total*100:.1f}%)")
    
    print(f"\n🏷️  Categories ({len(categories)}):")
    for cat, count in categories.most_common():
        bar_len = int(count / max(categories.values()) * 40)
        bar = "█" * bar_len
        print(f"  {cat:<20} {count:>5} {bar}")
    
    print(f"\n📍 Data sources:")
    for src, count in sources.most_common():
        print(f"  {src}: {count}")
    
    print(f"\n🏢 Top brands:")
    for brand, count in brands.most_common(10):
        print(f"  {brand:<30} {count:>4}")
    
    # Check image files
    if os.path.exists(IMAGES_DIR):
        img_files = len([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\n🖼️  Images on disk: {img_files}")
        
        if img_files < local_images:
            print(f"  ⚠️  Warning: CSV reports {local_images} local images, but only {img_files} files found")

def analyze_cv_split():
    """Аналізує CV split (якщо є)"""
    if not os.path.exists(CV_SPLIT_DIR):
        print(f"\n⚠️  CV split not found: {CV_SPLIT_DIR}")
        print("   Run: python src/ml/prepare_training_split.py")
        return
    
    print("\n" + "="*70)
    print("🔀 TRAIN/VAL/TEST SPLIT")
    print("="*70)
    
    splits = {}
    for split in ["train", "val", "test"]:
        split_dir = Path(CV_SPLIT_DIR) / split
        if not split_dir.exists():
            continue
        
        classes = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                classes[class_dir.name] = count
        
        splits[split] = classes
    
    # Print table
    all_classes = sorted(set(k for s in splits.values() for k in s.keys()))
    
    print(f"\n{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-"*70)
    
    for cls in all_classes:
        train = splits.get("train", {}).get(cls, 0)
        val = splits.get("val", {}).get(cls, 0)
        test = splits.get("test", {}).get(cls, 0)
        total = train + val + test
        print(f"{cls:<20} {train:>8} {val:>8} {test:>8} {total:>8}")
    
    print("-"*70)
    total_train = sum(splits.get("train", {}).values())
    total_val = sum(splits.get("val", {}).values())
    total_test = sum(splits.get("test", {}).values())
    grand_total = total_train + total_val + total_test
    
    print(f"{'TOTAL':<20} {total_train:>8} {total_val:>8} {total_test:>8} {grand_total:>8}")
    print(f"{'Percentage':<20} {total_train/grand_total*100:>7.1f}% {total_val/grand_total*100:>7.1f}% {total_test/grand_total*100:>7.1f}%")
    
    # Warnings
    print("\n💡 Recommendations:")
    
    if grand_total < 1000:
        print("  ⚠️  Small dataset (<1000 images) - high risk of overfitting")
        print("     → Use aggressive data augmentation")
        print("     → Consider getting more data")
    
    min_per_class = min(sum(splits[s].get(c, 0) for s in splits) for c in all_classes)
    if min_per_class < 30:
        print(f"  ⚠️  Some classes have <30 images - may not train well")
        print("     → Consider removing small classes or collecting more data")
    
    imbalance = max(all_classes, key=lambda c: sum(splits[s].get(c, 0) for s in splits))
    imbalance_ratio = sum(splits[s].get(imbalance, 0) for s in splits) / min_per_class
    if imbalance_ratio > 3:
        print(f"  ⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f}x)")
        print("     → Most data in: {imbalance}")
        print("     → Consider class weights in loss function")

def main():
    analyze_csv()
    analyze_cv_split()
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()