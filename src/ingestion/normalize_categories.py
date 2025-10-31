# This script reads a raw products CSV (downloaded from OpenFoodFacts)
# loads label_mapping.json, and maps messy category/ingredient/product names
# to standardized labels (e.g., “whey protein” - “protein”).

import csv, json, re, os
from collections import Counter

# -------------------- PATHS --------------------

IN_CSV = "data/processed/openfoodfacts_products.csv"     # Input CSV produced by openfoodfacts_ingest.py
OUT_CSV = "data/processed/products_merged.csv"           # Output CSV file with mapped labels
UNMAPPED_CSV = "data/processed/unmapped_products.csv"    # NEW: for analysis
MAP_FILE = "src/ml/label_mapping.json"                   # JSON file with keyword - label mapping

# -------------------- CATEGORY MAPPER --------------------

def map_category_priority(row: dict, mapping: dict) -> tuple[str, str]:
#    Try to match a product’s text fields against known mapping keys.
#       Priority order: category - ingredients - product_name.
#
#    Parameters
#    ----------
#    row : dict
#        One product row (fields: category, ingredients, product_name)
#    mapping : dict
#        Dictionary where keys are keywords and values are standardized labels
#
#    Returns
#    -------
#    str | None
#        The matched label or None if nothing matched.    
#   Loop through each relevant field in priority order.

#   IMPROVED: Priority search Searches for a category in priority order: category → ingredients → product_name. Returns (mapped_label, source_field)
    # Priority 1: category (most reliable)
    category_text = row.get("category", "")
    if category_text:
        for keyword, label in mapping.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", category_text.lower()):
                return (label, "category")
    
    # Priority 2: ingredients
    ingredients_text = row.get("ingredients", "")
    if ingredients_text:
        for keyword, label in mapping.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", ingredients_text.lower()):
                return (label, "ingredients")
    
    # Priority 3: product_name (least reliable, but better than nothing)
    product_name = row.get("product_name", "")
    if product_name:
        for keyword, label in mapping.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", product_name.lower()):
                return (label, "product_name")
    
    return (None, None)

# -------------------- MAIN PIPELINE --------------------

def main():

#        Read the input CSV, apply map_category() to every product, and write
#        the new file with 'category_mapped' field added.

 # Ensure the source CSV exists
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"No input: {IN_CSV}")
    # Load the mapping dictionary (keyword - label)
    with open(MAP_FILE, encoding="utf-8") as f:
        mapping = json.load(f)

    out_rows = []
    unmapped_rows = []

    # NEW: Statistics
    total_count = 0
    mapped_count = 0
    source_stats = Counter()  # counter: from which field the category was taken
    category_stats = Counter()  # counter: how many of each category
    # Read the input CSV row by row
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total_count += 1
            # If there is already category_mapped (from a previous run) - skip
            if row.get("category_mapped"):
                mapped_label = row["category_mapped"]
                source = "previous_run"
            else:
                mapped_label, source = map_category_priority(row, mapping)
                row["category_mapped"] = mapped_label
            
            if mapped_label:
                mapped_count += 1
                category_stats[mapped_label] += 1
                source_stats[source] += 1
            else:
                # NEW: keep unmapped for analysis
                unmapped_rows.append({
                    "source_id": row.get("source_id"),
                    "product_name": row.get("product_name"),
                    "category": row.get("category"),
                    "ingredients": row.get("ingredients", "")[:100],  # перші 100 символів
                })
            out_rows.append(row)

    # Validate that we actually got rows to write
    if not out_rows:
        raise RuntimeError("No rows to write. Did ingestion produce data?")
    # Write normalized CSV to disk
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    # NEW: keep unmapped for analysis
    if unmapped_rows:
        with open(UNMAPPED_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=unmapped_rows[0].keys())
            w.writeheader()
            w.writerows(unmapped_rows)
    # NEW: detailed statistics
    print("\n" + "="*60)
    print("📊 NORMALIZATION SUMMARY")
    print("="*60)
    print(f"Total products: {total_count}")
    print(f"Mapped: {mapped_count} ({mapped_count/total_count*100:.1f}%)")
    print(f"Unmapped: {len(unmapped_rows)} ({len(unmapped_rows)/total_count*100:.1f}%)")
    
    print(f"\n📍 Mapping sources:")
    for source, count in source_stats.most_common():
        print(f"  {source}: {count} ({count/mapped_count*100:.1f}%)")
    
    print(f"\n🏷️  Category distribution:")
    for category, count in category_stats.most_common():
        print(f"  {category}: {count}")
    
    print("="*60)
    print(f"\n✅ Saved:")
    print(f"  - Mapped products → {OUT_CSV}")
    if unmapped_rows:
        print(f"  - Unmapped products → {UNMAPPED_CSV} (review these!)")
    print()

# -------------------- ENTRYPOINT --------------------
if __name__ == "__main__":
    main()
