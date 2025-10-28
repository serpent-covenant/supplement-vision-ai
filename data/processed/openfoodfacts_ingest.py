import os, csv, json, time, hashlib, pathlib, sys, mimetypes
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urlunparse
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
    "en:vitamins-and-minerals",
    "en:amino-acids",
    "en:pre-workout",
    "en:omega-3",
    "en:collagen",
]

# Limits for a safe run; feel free to tweak.
MAX_PAGES_PER_CAT = 10
PAGE_SIZE = 80                 # a bit lower than 100 to reduce timeouts
APPEND_FLUSH_EVERY = 50
REQUEST_TIMEOUT = (5, 30)      # connect_timeout, read_timeout

HEADERS = {
    "User-Agent": "SupplementVisionAI/0.1 (+contact@example.com)"
}

# -------------------------
# Resilient HTTP session
# -------------------------
def make_session() -> requests.Session:
    """
    EN: Build a session with retries and backoff for robustness.
    UKR: Створює сесію з ретраями та бекофом для надійності.
    """
    retry = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=0.6,               # 0.6, 1.2, 2.4, ...
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
    EN: Make a stable image filename from URL hash; infer extension from content-type or URL.
    UKR: Формує стабільну назву файлу з хеша URL; визначає розширення з content-type або URL.
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
        print(f"[timeout] fetch_page {category=} {page=} -> {e}; continuing with empty page")
        return {"products": []}
    except requests.exceptions.RequestException as e:
        print(f"[request error] fetch_page {category=} {page=} -> {e}")
        return {"products": []}

def save_image(url: Optional[str]) -> Optional[str]:
    """
    Download a product image (if URL present) and persist it under IMAGES_DIR.
    The filename is stable: md5(url) + extension (guessed from content-type/URL).

    Returns
    -------
    Optional[str]
        Local filesystem path to the saved image, or None if download failed/skipped.
    """
    if not url:
        return None
    try:
        # Head or get to infer content-type (use GET directly here for simplicity)
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200 or not resp.content:
            return None

        fname = _stable_image_name(url, resp.headers.get("Content-Type"))
        dest = os.path.join(IMAGES_DIR, fname)

        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(resp.content)
        return dest
    except requests.exceptions.Timeout as e:
        print(f"[img timeout] {url} -> {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[img request error] {url} -> {e}")
        return None
    except Exception as e:
        print(f"[img error] {url} -> {e}")
        return None

def normalize_record(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a raw OFF product dict into our flat schema for CSV storage.
    Only essential fields are kept; heavy nested data stays in 'raw_json'.
    """
    brand = (p.get("brands") or "").split(",")[0].strip() if p.get("brands") else None
    return {
        "source": "openfoodfacts",                 # Provenance 
        "source_id": p.get("code"),                # OFF code   
        "brand": brand,
        "product_name": p.get("product_name"),
        "category": p.get("categories"),           # Raw categories from OpenFoodFacts
        "category_mapped": None,                   # filled later by normalization step
        "ingredients": p.get("ingredients_text"),
        "form": None,                              # future: powder/capsule/liquid
        "image_url": p.get("image_url"),           # may be replaced with local path after save_image()
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

def main():
    """
    Main ingestion loop:
        - Iterate categories/pages, normalize records, download images (non-blocking),
          and append to CSV in small batches.
    """
    buffer: List[Dict[str, Any]] = []
    total = 0
    try:
        for cat in CATEGORIES:
            for page in range(1, MAX_PAGES_PER_CAT + 1):
                data = fetch_page(cat, page=page, page_size=PAGE_SIZE)
                products = data.get("products", [])
                if not products:
                    break

                for p in products:
                    rec = normalize_record(p)
                    local_img = save_image(rec["image_url"]) if rec.get("image_url") else None
                    if local_img:
                        rec["image_url"] = local_img
                    buffer.append(rec)
                    total += 1

                    if len(buffer) >= APPEND_FLUSH_EVERY:
                        write_rows_append(buffer)
                        buffer.clear()

                # polite delay to avoid rate limits
                time.sleep(0.5)

        if buffer:
            write_rows_append(buffer)
            buffer.clear()

        print(f"Saved {total} records -> {OUT_CSV}")

    except KeyboardInterrupt:
        if buffer:
            write_rows_append(buffer)
            print(f"\nInterrupted. Partial save done. Current total written -> {OUT_CSV}")
        sys.exit(1)

if __name__ == "__main__":
    main()
