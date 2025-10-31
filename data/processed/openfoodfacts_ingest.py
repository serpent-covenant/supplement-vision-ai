import os, csv, json, time, hashlib, pathlib, sys, mimetypes
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------
# Paths and constants
# -------------------------
OUT_CSV = "data/processed/openfoodfacts_products.csv"
IMAGES_DIR = "data/images"
pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
pathlib.Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"

CATEGORIES = [
    "en:sports-nutrition",
    "en:creatine",
    "en:proteins",
    "en:whey-proteins",  
    "en:protein-powders",  
    "en:vitamins-and-minerals",
    "en:amino-acids",
    "en:pre-workout",
    "en:omega-3",
    "en:collagen",
    "en:bcaa",  
    "en:protein-bars",  
]

# Limits for a safe run; feel free to tweak.
MAX_PAGES_PER_CAT = 60          # more data for ML -> 60
PAGE_SIZE = 100
APPEND_FLUSH_EVERY = 50
REQUEST_TIMEOUT = (10, 60)       # Connect_timeout, read_timeout
MAX_IMAGE_WORKERS = 5          # was 10 → now 5 (less workload, more success rate)

HEADERS = {
    "User-Agent": "SupplementVisionAI/0.1 (+contact@example.com)"
}

# -------------------------
# Resilient HTTP session
# -------------------------

def make_session() -> requests.Session:
    """
    Build a session with retries and backoff for robustness.
    """
    retry = Retry(
        total=8,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    s = requests.Session()
    s.headers.update(HEADERS)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

session = make_session()

# -------------------------
# Helpers
# -------------------------
def _stable_image_name(url: str, content_type: Optional[str]) -> str:
    """
    Make a stable image filename from URL hash; infer extension from content-type or URL.
    """
    # Try from content-type first
    ext = None
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
    
    if not ext:
        # Strip query/fragment from URL and use its suffix if present
        u = urlparse(url)
        clean = urlunparse((u.scheme, u.netloc, u.path, "", "", ""))
        _, url_ext = os.path.splitext(clean)
        if url_ext and len(url_ext) <= 5:
            ext = url_ext
    
    if not ext:
        ext = ".jpg"
    
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ext

# -------------------------
# Fetch / Save / Normalize
# -------------------------
def fetch_page(category: str, page: int = 1, page_size: int = PAGE_SIZE) -> Dict[str, Any]:
    """
    Fetch one paginated slice of products for a given OFF category.
    Uses a shared `session` with retries/backoff.

    Parameters
    ----------
    category : str
        OFF taxonomy slug (e.g., "en:sports-nutrition").
    page : int
        1-based page index to request from the API.
    page_size : int
        Number of results per page.

    Returns
    -------
    dict
        Parsed JSON with keys like 'count', 'page', 'products'.
    """
    params = {
        "action": "process",
        "json": 1,
        "tagtype_0": "categories",
        "tag_contains_0": "contains",
        "tag_0": category,
        "page": page,
        "page_size": page_size,
        "fields": "code,brands,product_name,ingredients_text,image_url,categories,countries,lang",
    }
    try:
        r = session.get(SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout as e:
        print(f"[timeout] fetch_page {category=} {page=} -> {e}")
        return {"products": []}
    except requests.exceptions.RequestException as e:
        print(f"[request error] fetch_page {category=} {page=} -> {e}")
        return {"products": []}


def save_image(url: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Download a product image (if URL present) and persist it under IMAGES_DIR.
    The filename is stable: md5(url) + extension (guessed from content-type/URL).

    Returns
    -------
    Optional[str]
        Local filesystem path to the saved image, or None if download failed/skipped.
        NEW_UPDATE - Returns (original_url, local_path) or (url, None) if error
    """
    if not url:
        return (None, None)
    try:
        # First check if the file already exists
        fname = _stable_image_name(url, None)
        dest = os.path.join(IMAGES_DIR, fname)

        if os.path.exists(dest): # already exists, we are not loading again
            return (url, dest)
        # Downloading 
        resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        if resp.status_code != 200 or not resp.content:
            return (url, None)
        # Check if this is really an image
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return (url, None)
        
        # Save
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return (url, dest)
    
    except requests.exceptions.Timeout:
        return (url, None)
    except requests.exceptions.RequestException:
        return (url, None)
    except Exception:
        return (url, None)

# NEW: Parallel image loading
def save_images_batch(urls: List[str]) -> Dict[str, Optional[str]]:
    """
    Loads many images in parallel.
    Returns dict: {original_url: local_path or None}
    """
    result = {}
    with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
        future_to_url = {executor.submit(save_image, url): url for url in urls if url}
        
        for future in as_completed(future_to_url):
            original_url, local_path = future.result()
            if original_url:
                result[original_url] = local_path
    
    return result

def normalize_record(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a raw OFF product dict into our flat schema for CSV storage.
    Only essential fields are kept; heavy nested data stays in 'raw_json'.
    """
    brand = (p.get("brands") or "").split(",")[0].strip() if p.get("brands") else None
    return {
        "source": "openfoodfacts",                  # Provenance 
        "source_id": p.get("code"),                 # OFF code 
        "brand": brand,
        "product_name": p.get("product_name"),
        "category": p.get("categories"),            # Raw categories from OpenFoodFacts
        "category_mapped": None,                    # filled later by normalization step
        "ingredients": p.get("ingredients_text"),
        "form": None,                               # future: powder/capsule/liquid
        "image_url": p.get("image_url"),            # may be replaced with local path after save_image()
        "upc_ean": p.get("code"),
        "country": p.get("countries"),
        "lang": p.get("lang"),
        "raw_json": json.dumps(p, ensure_ascii=False),
    }

# CSV column order — ensures consistent header order in all saved files.
FIELDNAMES = [
    "source","source_id","brand","product_name","category","category_mapped",
    "ingredients","form","image_url","upc_ean","country","lang","raw_json"
]

def write_rows_append(rows: List[Dict[str, Any]]):
    """
    Incremental CSV writer:
        - If OUT_CSV doesn’t exist → create a new file and write the header.
        - Otherwise → append new rows without overwriting existing data.
    """
    file_exists = os.path.exists(OUT_CSV)
    mode = "a" if file_exists else "w"
    with open(OUT_CSV, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

# NEW STATS
class Stats:
    def __init__(self):
        self.total_products = 0
        self.images_downloaded = 0
        self.images_failed = 0
        self.images_skipped = 0
        self.by_category = {}
    
    def print_summary(self):
        print("\n" + "="*50)
        print("📊 INGESTION SUMMARY")
        print("="*50)
        print(f"Total products: {self.total_products}")
        print(f"Images downloaded: {self.images_downloaded}")
        print(f"Images failed: {self.images_failed}")
        print(f"Success rate: {self.images_downloaded/(self.images_downloaded+self.images_failed)*100:.1f}%")
        print("\nBy category:")
        for cat, count in sorted(self.by_category.items()):
            print(f"  {cat}: {count}")
        print("="*50)

def main():
    """
    Main ingestion loop:
        - Iterate categories/pages, normalize records, download images (non-blocking),
          and append to CSV in small batches.
    """
    buffer: List[Dict[str, Any]] = []
    stats = Stats()
    
    try:
        for cat_idx, cat in enumerate(CATEGORIES, 1):
            print(f"\n[{cat_idx}/{len(CATEGORIES)}] Processing category: {cat}")
            cat_count = 0
            
            for page in range(1, MAX_PAGES_PER_CAT + 1):
                print(f"  Page {page}/{MAX_PAGES_PER_CAT}...", end=" ")
                data = fetch_page(cat, page=page, page_size=PAGE_SIZE)
                products = data.get("products", [])
                
                if not products:
                    print("empty, stopping category")
                    break
                
                print(f"{len(products)} products")
                
                # Normalize records
                records = [normalize_record(p) for p in products]
                
                # Batch download images
                image_urls = [r["image_url"] for r in records if r.get("image_url")]
                url_to_path = save_images_batch(image_urls)
                
                # Update records with local paths
                for rec in records:
                    original_url = rec["image_url"]
                    if original_url and original_url in url_to_path:
                        local_path = url_to_path[original_url]
                        if local_path:
                            rec["image_url"] = local_path
                            stats.images_downloaded += 1
                        else:
                            stats.images_failed += 1
                    
                    buffer.append(rec)
                    stats.total_products += 1
                    cat_count += 1
                
                # Flush buffer
                if len(buffer) >= APPEND_FLUSH_EVERY:
                    write_rows_append(buffer)
                    buffer.clear()
                
                time.sleep(1.0)  # polite delay
            
            stats.by_category[cat] = cat_count
        
        # Final flush
        if buffer:
            write_rows_append(buffer)
            buffer.clear()
        
        stats.print_summary()
        print(f"\n✅ Saved to {OUT_CSV}")
    
    except KeyboardInterrupt:
        if buffer:
            write_rows_append(buffer)
            print(f"\n⚠️  Interrupted. Partial save done.")
        stats.print_summary()
        sys.exit(1)

if __name__ == "__main__":
    main()
    