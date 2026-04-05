"""
Wajah AI — Multi-source intelligent scraper.
Sources: Pexels API (primary), Bing (fallback)
Filters: front-view, symmetry, single face, resolution, AI detection, gender, hash
Age estimation: OpenCV DNN → auto-route to 25-39 / 40-65
"""

import os
import re
import argparse
import hashlib
from datetime import datetime

import cv2
import numpy as np
import requests
import pandas as pd
import imagehash

from PIL import Image
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw_images")
REGISTRY_PATH = os.path.join(BASE, "registry.xlsx")
HASH_FILE = os.path.join(BASE, "data", "hash_registry.txt")
AGE_PROTO = os.path.join(BASE, "data", "models", "age_deploy.prototxt")
AGE_MODEL = os.path.join(BASE, "data", "models", "age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE, "data", "models", "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(BASE, "data", "models", "gender_net.caffemodel")
GENDER_LIST = ["pria", "wanita"]  # index 0=Male, index 1=Female
COLUMNS = [
    "no",              # nomor urut otomatis
    "filename",        # nama file saat download (img_0001.jpg)
    "url",             # URL sumber gambar (anti-duplikasi, TIDAK PERNAH dihapus)
    "source",          # sumber: pexels / bing
    "query",           # keyword pencarian yang dipakai
    "gender",          # pria / wanita
    "age_group",       # 25-39 / 40-65
    "status",          # raw → approved / reject (diupdate oleh batch_manager)
    "batch",           # batch_1, batch_2, dst (diupdate oleh batch_manager)
    "batch_filename",  # nama file di batch (pria_25-39_hidung_001.jpg)
    "created_at",      # waktu download
]

TIMEOUT = 8
MAX_RETRY = 2
MIN_WIDTH = 256
MIN_HEIGHT = 256
HASH_DISTANCE_THRESHOLD = 4
MIN_FACE_WIDTH_RATIO = 0.18  # wajah minimal 18% lebar gambar
MAX_SYMMETRY = 0.18          # front view check (toleran wajah sedikit miring)
MIN_TURN_RATIO = 0.50        # tidak menoleh terlalu jauh

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

BAD_URL_KEYWORDS = [
    "logo", "icon", "cartoon", "anime", "illustration", "vector",
    "avatar", "drawing", "clipart", "emoji", "sticker", "sketch",
    "3d-render", "cgi", "pixar",
]


# Pexels queries (from Thoriq)
PEXELS_QUERIES_WANITA = [
    "before botox face natural woman close up",
    "unretouched face close up woman portrait",
    "raw portrait photography natural lighting woman face",
    "caucasian woman face close up no makeup natural skin",
    "woman face front view plain background natural",
    "woman headshot no makeup natural skin close up",
    "before dermal filler face woman natural",
    "woman portrait straight face natural light",
    "female face closeup studio portrait",
    "mature woman face no filter portrait",
    "asian woman face natural portrait close up",
    "woman passport photo style face",
    "woman bare face skincare before after",
]

PEXELS_QUERIES_PRIA = [
    "before botox face natural man close up",
    "unretouched face close up man portrait",
    "raw portrait photography natural lighting man face",
    "caucasian man face close up natural headshot",
    "man face front view plain background natural",
    "man headshot natural close up portrait plain background",
    "before dermal filler face man natural",
    "man portrait straight face natural light",
    "male face closeup studio portrait",
    "mature man face no filter portrait",
    "asian man face natural portrait close up",
    "man passport photo style face",
    "man bare face grooming before after",
]

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        df = pd.read_excel(REGISTRY_PATH)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=COLUMNS)


def save_registry(df):
    df.to_excel(REGISTRY_PATH, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# HASH REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def load_hashes():
    hashes = []
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        hashes.append(imagehash.hex_to_hash(line))
                    except Exception:
                        pass
    return hashes


def save_hash(h):
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    with open(HASH_FILE, "a") as f:
        f.write(str(h) + "\n")


def is_duplicate_hash(img_path, existing_hashes):
    try:
        img = Image.open(img_path)
        h = imagehash.phash(img)
        for eh in existing_hashes:
            if abs(h - eh) < HASH_DISTANCE_THRESHOLD:
                return True, h
        return False, h
    except Exception:
        return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SOURCE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_pexels(query, api_key, max_results=80):
    """Search Pexels API — high quality, no watermark."""
    urls = []
    pexels_ids = set()
    for page in range(1, 8):  # 7 halaman × 40 = max 280 URL
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": api_key},
                params={"query": query, "per_page": 40, "page": page,
                         "orientation": "portrait", "size": "medium"},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            photos = resp.json().get("photos", [])
            if not photos:
                break
            for p in photos:
                if p["id"] not in pexels_ids:
                    pexels_ids.add(p["id"])
                    urls.append(p["src"]["large"])
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_unsplash(query, api_key, max_results=80):
    """Search Unsplash API — high quality, no watermark."""
    urls = []
    seen_ids = set()
    for page in range(1, 6):  # 5 halaman × 30 = max 150 URL
        try:
            resp = requests.get(
                "https://api.unsplash.com/search/photos",
                headers={"Authorization": f"Client-ID {api_key}"},
                params={"query": query, "per_page": 30, "page": page,
                         "orientation": "portrait"},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for photo in results:
                pid = photo["id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    # Ambil ukuran regular (1080px width) — cukup untuk face crop
                    url = photo.get("urls", {}).get("regular", "")
                    if url:
                        urls.append(url)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_pixabay(query, api_key, max_results=80):
    """Search Pixabay API — good quality, no watermark."""
    urls = []
    seen_ids = set()
    for page in range(1, 6):  # 5 halaman × 40 = max 200 URL
        try:
            resp = requests.get(
                "https://pixabay.com/api/",
                params={"key": api_key, "q": query, "per_page": 40, "page": page,
                         "image_type": "photo", "orientation": "vertical",
                         "category": "people", "safesearch": "true"},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            hits = resp.json().get("hits", [])
            if not hits:
                break
            for h in hits:
                pid = h["id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    url = h.get("largeImageURL", "")
                    if url:
                        urls.append(url)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_openverse(query, max_results=80):
    """Search Openverse API — CC-licensed images from Flickr, Wikimedia, etc."""
    urls = []
    seen_ids = set()
    for page in range(1, 5):  # 4 halaman × 20 = max 80 URL
        try:
            resp = requests.get(
                "https://api.openverse.org/v1/images/",
                params={"q": query, "page_size": 20, "page": page},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for img in results:
                pid = img.get("id", "")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    url = img.get("url", "")
                    if url:
                        urls.append(url)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_stocksnap(query, max_results=80):
    """Search StockSnap.io — hidden JSON API, CC0 licensed."""
    urls = []
    seen_ids = set()
    q = query.replace(" ", "+")
    for page in range(1, 5):
        try:
            resp = requests.get(
                f"https://stocksnap.io/api/search-photos/{q}/relevance/desc/{page}",
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for item in results:
                pid = item.get("img_id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    urls.append(f"https://cdn.stocksnap.io/img-thumbs/960w/{pid}.jpg")
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_burst(query, max_results=60):
    """Search Burst by Shopify — free stock photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            resp = requests.get(
                "https://www.shopify.com/stock-photos/photos/search",
                params={"q": query, "page": page},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            found = re.findall(r'(https://burst\.shopifycdn\.com/photos/[^"?]+)', resp.text)
            if not found:
                break
            for u in found:
                full = u + "?width=1850&format=pjpg"
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_nappy(query, max_results=60):
    """Search Nappy.co — diverse face photos, CC0 licensed."""
    urls = []
    seen = set()
    q = query.replace(" ", "-")
    try:
        resp = requests.get(
            f"https://nappy.co/s/{q}",
            headers=HEADERS, timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            found = re.findall(r'(https://images\.nappy\.co/photo/[^"?]+)', resp.text)
            for u in found:
                full = u + "?width=2048"
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
    except Exception:
        pass
    return urls[:max_results]


def scrape_negativespace(query, max_results=60):
    """Search Negative Space — free high-res photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            resp = requests.get(
                f"https://negativespace.co/page/{page}/",
                params={"s": query},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            found = re.findall(
                r'(https://negativespace\.co/wp-content/uploads/\d+/\d+/[^"]+\.jpg)',
                resp.text
            )
            if not found:
                break
            for u in found:
                # Hapus suffix dimensi untuk full-res
                full = re.sub(r'-\d+x\d+\.jpg$', '.jpg', u)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_isorepublic(query, max_results=60):
    """Search ISO Republic — free high-res photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            resp = requests.get(
                f"https://isorepublic.com/page/{page}/",
                params={"s": query},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            found = re.findall(
                r'(https://isorepublic\.com/wp-content/uploads/\d+/\d+/[^"]+\.jpg)',
                resp.text
            )
            if not found:
                break
            for u in found:
                full = re.sub(r'-\d+x\d+\.jpg$', '.jpg', u)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_wallhaven(query, max_results=80):
    """Search Wallhaven API — high-res, no key needed, people category."""
    urls = []
    seen_ids = set()
    for page in range(1, 5):
        try:
            resp = requests.get(
                "https://wallhaven.cc/api/v1/search",
                params={"q": query, "categories": "010", "purity": "100",
                         "sorting": "relevance", "page": page},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            data = resp.json().get("data", [])
            if not data:
                break
            for item in data:
                pid = item.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    url = item.get("path", "")
                    if url:
                        urls.append(url)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_wikimedia(query, max_results=60):
    """Search Wikimedia Commons — public domain, huge volume."""
    urls = []
    seen = set()
    for offset in range(0, max_results, 20):  # Kurangi iterations
        try:
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={"action": "query", "list": "search",
                         "srsearch": query + " photograph",
                         "srnamespace": "6", "format": "json",
                         "srlimit": 30, "sroffset": offset},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            results = resp.json().get("query", {}).get("search", [])
            if not results:
                break
            # Batch get image URLs
            titles = "|".join(r["title"] for r in results)
            info_resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={"action": "query", "titles": titles,
                         "prop": "imageinfo", "iiprop": "url|mime",
                         "format": "json"},
                timeout=TIMEOUT,
            )
            if info_resp.status_code != 200:
                break
            pages = info_resp.json().get("query", {}).get("pages", {})
            for pid, page in pages.items():
                info = page.get("imageinfo", [{}])[0]
                mime = info.get("mime", "")
                url = info.get("url", "")
                if url and "image" in mime and url not in seen:
                    seen.add(url)
                    urls.append(url)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_barnimages(query, max_results=60):
    """Search Barnimages — free high-res stock photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            resp = requests.get(
                f"https://barnimages.com/page/{page}/",
                params={"s": query},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            found = re.findall(
                r'(https://barnimages\.com/wp-content/uploads/\d+/\d+/[^"\s]+\.jpg)',
                resp.text
            )
            if not found:
                break
            for u in found:
                full = re.sub(r'-\d+x\d+\.jpg$', '.jpg', u)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_startupstock(query, max_results=60):
    """Search StartupStockPhotos — free business/people photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            resp = requests.get(
                f"https://startupstockphotos.com/page/{page}/",
                params={"s": query},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            found = re.findall(
                r'(https://startupstockphotos\.com/wp-content/uploads/\d+/\d+/[^"\s]+\.jpg)',
                resp.text
            )
            if not found:
                break
            for u in found:
                full = re.sub(r'-\d+x\d+\.jpg$', '.jpg', u)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_picjumbo(query, max_results=60):
    """Search Picjumbo — free stock photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            resp = requests.get(
                f"https://picjumbo.com/page/{page}/",
                params={"s": query},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            found = re.findall(
                r'(https://i0\.wp\.com/picjumbo\.com/wp-content/uploads/[^"\s\'> ]+\.(?:jpg|jpeg|png))',
                resp.text
            )
            if not found:
                break
            for u in found:
                # Remove i0.wp.com proxy prefix
                direct = u.replace("https://i0.wp.com/", "https://")
                direct = re.sub(r'\?.*$', '', direct)  # remove query params
                if direct not in seen:
                    seen.add(direct)
                    urls.append(direct)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_pixnio(query, max_results=60):
    """Search Pixnio — CC0 public domain photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            params = {"s": query}
            if page > 1:
                url = f"https://pixnio.com/page/{page}/"
            else:
                url = "https://pixnio.com/"
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200:
                break
            found = re.findall(
                r'(https://pixnio\.com/free-images/[^"\'\s]+\.(?:jpg|jpeg|png))',
                resp.text
            )
            if not found:
                break
            for u in found:
                full = re.sub(r'-\d+x\d+\.', '.', u)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_shotstash(query, max_results=60):
    """Search ShotStash — free stock photos."""
    urls = []
    seen = set()
    for page in range(1, 4):
        try:
            resp = requests.get(
                f"https://shotstash.com/page/{page}/",
                params={"s": query},
                headers=HEADERS, timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                break
            found = re.findall(
                r'(https://shotstash\.com/wp-content/uploads/[^"\'\s]+\.(?:jpg|jpeg|png|webp))',
                resp.text
            )
            if not found:
                break
            for u in found:
                full = re.sub(r'-\d+x\d+\.', '.', u)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception:
            break
        if len(urls) >= max_results:
            break
    return urls[:max_results]


def scrape_iwaria(max_results=60):
    """Scrape Iwaria — African face photos, JSON endpoint."""
    urls = []
    try:
        resp = requests.get(
            "https://iwaria.com/photos",
            headers=HEADERS, timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            try:
                items = resp.json()
                for item in items:
                    url = item.get("url", "")
                    if url:
                        if not url.startswith("http"):
                            url = "https:" + url
                        urls.append(url)
            except Exception:
                # Fallback: regex dari HTML
                found = re.findall(r'(https?://[^"\s]+iwaria[^"\s]+\.(?:jpg|jpeg|png))', resp.text)
                urls = list(set(found))
    except Exception:
        pass
    return urls[:max_results]


def scrape_randomuser(gender, max_results=50):
    """RandomUser.me — direct portrait URLs, guaranteed faces."""
    urls = []
    # Direct URL pattern: men/0-99, women/0-99
    folder = "women" if gender == "wanita" else "men"
    import random
    indices = list(range(100))
    random.shuffle(indices)
    for i in indices[:max_results]:
        urls.append(f"https://randomuser.me/api/portraits/{folder}/{i}.jpg")
    return urls


def scrape_thisperson(max_results=30):
    """ThisPersonDoesNotExist — AI-generated faces, unlimited unique."""
    urls = []
    # Setiap request URL unik dengan random seed
    import random
    for _ in range(max_results):
        seed = random.randint(1, 999999)
        urls.append(f"https://thispersondoesnotexist.com/?seed={seed}")
    return urls


def scrape_bing(query, max_results=80):
    urls = []
    seen = set()
    page = 1
    while len(urls) < max_results:
        try:
            resp = requests.get(
                "https://www.bing.com/images/search",
                params={"q": query, "form": "HDRSC2", "first": page},
                headers=HEADERS, timeout=TIMEOUT,
            )
            resp.raise_for_status()
            found = re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', resp.text)
            if not found:
                break
            for u in found:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
            page += len(found)
            if len(found) < 5:
                break
        except Exception:
            break
    return urls[:max_results]


UNSPLASH_KEY = "rujPfKrr4OF5cPo8ra05WAhK4OG62j5WR8x9qDhUP0k"
PIXABAY_KEY = "55289989-28be865a55a6c1b347cb55321"


def search_all(query, gender, api_key=None, max_results=50):
    """Sequential search — coba sumber satu per satu.
    Minimal 4 sumber dicoba untuk variasi, berhenti setelah cukup.
    Returns list of (url, source) tuples."""
    all_urls = []
    seen = set()
    MIN_SOURCES = 4  # Minimal coba 4 sumber untuk variasi URL

    def _add(urls, source_name):
        added = 0
        for u in urls:
            if u not in seen:
                seen.add(u)
                all_urls.append((u, source_name))
                added += 1
        return added

    # Daftar semua sumber, urut prioritas
    sources = [
        ("Pexels",       lambda q: scrape_pexels(q, api_key, 80) if api_key else []),
        ("Unsplash",     lambda q: scrape_unsplash(q, UNSPLASH_KEY, 60)),
        ("Pixabay",      lambda q: scrape_pixabay(q, PIXABAY_KEY, 60)),
        ("Openverse",    lambda q: scrape_openverse(q, 60)),
        ("StockSnap",    lambda q: scrape_stocksnap(q, 30)),
        ("Wallhaven",    lambda q: scrape_wallhaven(q, 30)),
        ("Burst",        lambda q: scrape_burst(q, 25)),
        ("Nappy",        lambda q: scrape_nappy(q, 25)),
        ("Pixnio",       lambda q: scrape_pixnio(q, 20)),
        ("NegSpace",     lambda q: scrape_negativespace(q, 20)),
        ("ISORepub",     lambda q: scrape_isorepublic(q, 20)),
        ("Wikimedia",    lambda q: scrape_wikimedia(q, 25)),
        ("Barnimages",   lambda q: scrape_barnimages(q, 20)),
        ("StartupStock", lambda q: scrape_startupstock(q, 20)),
        ("Picjumbo",     lambda q: scrape_picjumbo(q, 20)),
        ("ShotStash",    lambda q: scrape_shotstash(q, 20)),
        ("Iwaria",       lambda q: scrape_iwaria(20)),
        ("RandomUser",   lambda q: scrape_randomuser(gender, 20)),
        ("TPDNE",        lambda q: scrape_thisperson(15)),
        ("Bing",         lambda q: scrape_bing(q, 30)),
    ]

    for i, (name, fetch_fn) in enumerate(sources):
        # Berhenti jika sudah cukup DAN sudah lewat minimum sumber
        if len(all_urls) >= max_results and i >= MIN_SOURCES:
            break

        print(f"  [{name}] searching...", end=" ")
        try:
            urls = fetch_fn(query)
            added = _add(urls, name.lower())
            print(f"→ {added} new (total: {len(all_urls)})")
        except Exception as e:
            print(f"→ skip ({str(e)[:30]})")

    print(f"  ✓ Got {len(all_urls)} candidates from {min(i+1, len(sources))} sources")
    return all_urls[:max_results * 2]


# ═══════════════════════════════════════════════════════════════════════════════
# URL PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def is_bad_url(url):
    url_lower = url.lower()
    return any(kw in url_lower for kw in BAD_URL_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD WITH RETRY
# ═══════════════════════════════════════════════════════════════════════════════

def download_image(url, save_path):
    for _ in range(MAX_RETRY):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            if resp.status_code != 200:
                continue
            ct = resp.headers.get("Content-Type", "")
            if "image" not in ct:
                return False
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            if os.path.getsize(save_path) > 5000:
                return True
        except Exception:
            pass
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# FACE VALIDATION (Thoriq's method — strict front-view + close-up)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_face(img_path, mesh):
    """
    Thoriq-style validation:
    1. Single face
    2. Close-up (face >= 25% image width)
    3. Front view (symmetry < 0.08)
    4. Not turning (eye ratio > 0.7)
    5. Face centered
    Returns (ok, reason, landmarks_or_None)
    """
    img = cv2.imread(img_path)
    if img is None:
        return False, "cannot read", None

    h, w = img.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, f"too small ({w}x{h})", None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return False, "no face", None

    if len(result.multi_face_landmarks) > 1:
        return False, "multiple faces", None

    lm = result.multi_face_landmarks[0].landmark

    # Face size check
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    face_width = max(xs) - min(xs)

    if face_width < MIN_FACE_WIDTH_RATIO:
        return False, f"face too small ({face_width:.0%})", None

    # Front view symmetry (Thoriq)
    nose_x = lm[1].x
    left_eye_x = lm[33].x
    right_eye_x = lm[263].x
    face_center_x = (left_eye_x + right_eye_x) / 2
    symmetry = abs(nose_x - face_center_x) / face_width

    if symmetry > MAX_SYMMETRY:
        return False, f"not front view (sym={symmetry:.3f})", None

    # Turn check (Thoriq)
    dist_left = abs(nose_x - left_eye_x)
    dist_right = abs(right_eye_x - nose_x)
    if dist_left > 0 and dist_right > 0:
        ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        if ratio < MIN_TURN_RATIO:
            return False, f"face turning (ratio={ratio:.2f})", None

    # Face centered (toleransi lebih lebar)
    face_cx = (min(xs) + max(xs)) / 2
    if abs(face_cx - 0.5) > 0.30:
        return False, "face not centered", None

    landmarks = np.array([(int(p.x * w), int(p.y * h)) for p in lm])
    return True, f"ok (w={face_width:.0%} sym={symmetry:.3f})", landmarks


# ═══════════════════════════════════════════════════════════════════════════════
# AGE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

_age_net = None

def get_age_net():
    global _age_net
    if _age_net is None:
        _age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    return _age_net


def estimate_age(img_path):
    """
    Estimate age group using OpenCV DNN.
    Returns: "25-39" or "40-65"
    """
    img = cv2.imread(img_path)
    if img is None:
        return "25-39"

    net = get_age_net()
    blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227),
                                  (78.4263377603, 87.7689143744, 114.895847746),
                                  swapRB=False)
    net.setInput(blob)
    preds = net.forward()
    age_idx = preds[0].argmax()
    age_bucket = AGE_BUCKETS[age_idx]

    # Map to our groups: 25-39 or 40-65
    # (0-2), (4-6), (8-12), (15-20) → skip (too young)
    # (25-32) → 25-39
    # (38-43) → 25-39 or 40-65 (borderline, use confidence)
    # (48-53), (60-100) → 40-65

    if age_idx <= 2:  # 0-12 → terlalu muda, tolak
        return "young"
    elif age_idx == 3:  # 15-20 → borderline, cek confidence
        # Jika probability (15-20) + (25-32) tinggi → terima sebagai 25-39
        prob_adult = sum(preds[0][3:])  # 15-20 ke atas
        prob_child = sum(preds[0][:3])  # 0-12
        if prob_adult > 0.6:
            return "25-39"
        return "young"
    elif age_idx == 4:  # 25-32
        return "25-39"
    elif age_idx == 5:  # 38-43
        prob_young = sum(preds[0][:5])
        prob_old = sum(preds[0][5:])
        return "40-65" if prob_old > prob_young else "25-39"
    else:  # 48-53, 60-100
        return "40-65"


# ═══════════════════════════════════════════════════════════════════════════════
# GENDER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_gender_net = None


def get_gender_net():
    global _gender_net
    if _gender_net is None:
        _gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    return _gender_net


def detect_gender_visual(img_path):
    """
    Detect gender dari gambar wajah menggunakan OpenCV DNN.
    Returns: ("pria", confidence) atau ("wanita", confidence)
    """
    img = cv2.imread(img_path)
    if img is None:
        return "unknown", 0.0

    net = get_gender_net()
    blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227),
                                  (78.4263377603, 87.7689143744, 114.895847746),
                                  swapRB=False)
    net.setInput(blob)
    preds = net.forward()
    gender_idx = preds[0].argmax()
    confidence = float(preds[0][gender_idx])
    return GENDER_LIST[gender_idx], confidence


# ═══════════════════════════════════════════════════════════════════════════════
# AI-GENERATED DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_ai_generated(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, ""
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    ai_score = 0
    reasons = []
    if lap_var < 35:
        ai_score += 2
        reasons.append(f"smooth(lap={lap_var:.0f})")
    elif lap_var < 80:
        ai_score += 1
    if edge_density < 0.015:
        ai_score += 2
        reasons.append(f"low-edge({edge_density:.3f})")
    elif edge_density < 0.03:
        ai_score += 1
    return ai_score >= 3, "; ".join(reasons)


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT / WATERMARK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# FILENAME GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def next_filename(df):
    os.makedirs(RAW_DIR, exist_ok=True)
    folder_files = set(f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    reg_files = set(df["filename"].dropna().tolist()) if not df.empty else set()
    all_files = folder_files | reg_files
    num = len(all_files) + 1
    fname = f"img_{num:04d}.jpg"
    while fname in all_files:
        num += 1
        fname = f"img_{num:04d}.jpg"
    return fname


def detect_gender(keyword):
    q = keyword.lower()
    if any(w in q for w in ["wanita", "female", "woman", "women", "girl"]):
        return "wanita"
    if any(w in q for w in ["pria", "male", "man ", "men ", "boy"]):
        return "pria"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

def filter_url(url, save_path, fname, mesh, existing_urls, existing_hashes, stats, df, expected_gender=None):
    """Download and filter one URL. Returns (keep, age_group, phash) or (False, None, None)."""

    if is_bad_url(url):
        stats["url_skip"] += 1
        return False, None, None

    if url in existing_urls:
        stats["url_dup"] += 1
        return False, None, None

    if not download_image(url, save_path):
        stats["dl_fail"] += 1
        if os.path.exists(save_path):
            os.remove(save_path)
        return False, None, None

    # Face validation
    face_ok, face_reason, landmarks = validate_face(save_path, mesh)
    if not face_ok:
        os.remove(save_path)
        if "multiple" in face_reason:
            stats["multi_face"] += 1
        elif "small" in face_reason or "too small" in face_reason:
            stats["small"] += 1
        elif "front" in face_reason or "turn" in face_reason or "center" in face_reason:
            stats["not_front"] += 1
        else:
            stats["no_face"] += 1
        print(f"    [DELETE] {fname} — {face_reason}")
        return False, None, None

    # Gender verification — pastikan gender visual sesuai request
    if expected_gender and expected_gender in ("pria", "wanita"):
        detected_gender, gender_conf = detect_gender_visual(save_path)
        if detected_gender != expected_gender and gender_conf > 0.65:
            os.remove(save_path)
            stats["wrong_gender"] += 1
            print(f"    [DELETE] {fname} — wrong gender (detected={detected_gender} conf={gender_conf:.0%}, expected={expected_gender})")
            return False, None, None

    # AI detection
    ai_flag, ai_reason = is_ai_generated(save_path)
    if ai_flag:
        os.remove(save_path)
        stats["ai"] += 1
        print(f"    [DELETE] {fname} — AI ({ai_reason})")
        return False, None, None

    # Hash duplicate
    is_dup, phash = is_duplicate_hash(save_path, existing_hashes)
    if is_dup:
        os.remove(save_path)
        stats["hash_dup"] += 1
        print(f"    [DELETE] {fname} — hash duplicate")
        return False, None, None

    # Age estimation
    age_group = estimate_age(save_path)
    if age_group == "young":
        os.remove(save_path)
        stats["young"] += 1
        print(f"    [DELETE] {fname} — too young")
        return False, None, None

    return True, age_group, phash


def scrape(keyword, limit=10, api_key=None):
    import random

    gender = detect_gender(keyword)
    print(f"\n[SCRAPE START] query='{keyword}' limit={limit} gender={gender}")
    print("=" * 60)

    # Load registries
    df = load_registry()
    existing_urls = set(df["url"].dropna().tolist())
    existing_hashes = load_hashes()
    print(f"  Registry: {len(existing_urls)} URLs, {len(existing_hashes)} hashes\n")

    new_rows = []
    downloaded = 0
    stats = {"url_skip": 0, "url_dup": 0, "dl_fail": 0, "no_face": 0,
             "multi_face": 0, "small": 0, "not_front": 0, "wrong_gender": 0,
             "ai": 0, "hash_dup": 0, "young": 0, "kept": 0}

    # Init FaceMesh once
    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=2,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    # Build query rotation list
    base_queries = PEXELS_QUERIES_WANITA if gender == "wanita" else PEXELS_QUERIES_PRIA
    all_queries = [keyword] + list(base_queries)
    random.shuffle(all_queries[1:])  # shuffle alternatives, keep user query first

    MAX_ROUNDS = 40
    STALE_LIMIT = 3   # berhenti jika N round berturut tanpa fresh URL
    round_num = 0
    stale_count = 0
    tried_urls = set()

    while downloaded < limit and round_num < MAX_ROUNDS:
        round_num += 1
        remaining = limit - downloaded

        # Pick query for this round
        query = all_queries[(round_num - 1) % len(all_queries)]

        print(f"\n  ── Round {round_num} (need {remaining} more) ──")
        print(f"  Query: '{query[:60]}'")

        # Search — returns list of (url, source) tuples
        # Multiplier x6 untuk kompensasi filter ketat
        url_pairs = search_all(query, gender, api_key=api_key, max_results=remaining * 6)

        # Filter out already tried URLs
        fresh_pairs = [(u, s) for u, s in url_pairs if u not in tried_urls]
        tried_urls.update(u for u, _ in url_pairs)

        if not fresh_pairs:
            stale_count += 1
            print(f"  No fresh URLs found ({stale_count}/{STALE_LIMIT} stale rounds)...")
            if stale_count >= STALE_LIMIT:
                print(f"\n  [STOP] {STALE_LIMIT} rounds berturut tanpa URL baru — semua sumber habis.")
                break
            continue

        # Reset stale counter karena dapat fresh URLs
        stale_count = 0

        print(f"  Fresh candidates: {len(fresh_pairs)}\n")

        for url, source in fresh_pairs:
            if downloaded >= limit:
                break

            fname = next_filename(df)
            save_path = os.path.join(RAW_DIR, fname)

            keep, age_group, phash = filter_url(
                url, save_path, fname, mesh,
                existing_urls, existing_hashes, stats, df,
                expected_gender=gender
            )

            # Simpan hash juga dari gambar yang ditolak — mencegah re-download
            if not keep and phash:
                save_hash(phash)
                existing_hashes.append(phash)

            if keep:
                if phash:
                    save_hash(phash)
                    existing_hashes.append(phash)

                next_no = len(df) + 1
                new_rows.append({
                    "no": next_no,
                    "filename": fname,
                    "url": url,
                    "source": source,
                    "query": query[:80],
                    "gender": gender,
                    "age_group": age_group,
                    "status": "raw",
                    "batch": "",
                    "batch_filename": "",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                existing_urls.add(url)
                df = pd.concat([df, pd.DataFrame([new_rows[-1]])], ignore_index=True)
                downloaded += 1
                stats["kept"] += 1
                print(f"    [KEEP] #{next_no} {fname} → {source}/{gender}/{age_group} ({downloaded}/{limit})")
            else:
                existing_urls.add(url)

        # Save after each round (in case of interruption)
        if new_rows:
            save_registry(df)

    mesh.close()

    if new_rows:
        save_registry(df)

    # Summary
    print(f"\n{'='*60}")
    print(f"[SCRAPE DONE] in {round_num} rounds")
    print(f"  Kept:          {stats['kept']}/{limit}")
    print(f"  URL filtered:  {stats['url_skip']}")
    print(f"  URL duplicate: {stats['url_dup']}")
    print(f"  Download fail: {stats['dl_fail']}")
    print(f"  No face:       {stats['no_face']}")
    print(f"  Not front:     {stats['not_front']}")
    print(f"  Wrong gender:  {stats['wrong_gender']}")
    print(f"  Multi face:    {stats['multi_face']}")
    print(f"  Too small:     {stats['small']}")
    print(f"  Too young:     {stats['young']}")
    print(f"  AI detected:   {stats['ai']}")
    print(f"  Hash dup:      {stats['hash_dup']}")
    print(f"  Registry:      {len(df)} total")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wajah AI Intelligent Scraper")
    parser.add_argument("--query", type=str,
                        default="natural human face close up unretouched",
                        help="Search keyword (include pria/wanita)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of quality images to keep")
    parser.add_argument("--pexels-key", type=str, default=None,
                        help="Pexels API key (free at pexels.com/api)")
    args = parser.parse_args()

    scrape(args.query, args.limit, api_key=args.pexels_key)
