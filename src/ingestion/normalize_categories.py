# This script reads a raw products CSV (downloaded from OpenFoodFacts)
# loads label_mapping.json, and maps messy category/ingredient/product names
# to standardized labels (e.g., “whey protein” - “protein”).

import csv, json, re, os

# -------------------- PATHS --------------------

IN_CSV = "data/processed/openfoodfacts_products.csv" # Input CSV produced by openfoodfacts_ingest.py
OUT_CSV = "data/processed/products_merged.csv"       # Output CSV file with mapped labels
MAP_FILE = "src/ml/label_mapping.json"               # JSON file with keyword - label mapping

# -------------------- CATEGORY MAPPER --------------------

def map_category(row, mapping):
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
    for field in ["category", "ingredients", "product_name"]:
        text = row.get(field, "")
        if text:
            for k, v in mapping.items():
                #Check if keyword appears as a whole word (case-insensitive)
                if re.search(r"\b" + re.escape(k) + r"\b", text.lower()):
                    return v
    return None

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

    # Read the input CSV row by row
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # Combine several fields into one text blob for broader matching
            src_text = " | ".join(filter(None, [
                row.get("category"), row.get("ingredients"), row.get("product_name")
            ]))
            #     If 'category_mapped' is already set, keep it
            #     otherwise try to map based on combined text
            row["category_mapped"] = row.get("category_mapped") or map_category(row, mapping)
            out_rows.append(row)

# Validate that we actually got rows to write
    if not out_rows:
        raise RuntimeError("No rows to write. Did ingestion produce data?")
# Write normalized CSV to disk
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    print(f"Normalized → {OUT_CSV} (rows: {len(out_rows)})")

# -------------------- ENTRYPOINT --------------------
if __name__ == "__main__":
    main()
