"""
Script pencarian gambar wajah usia 40-65 tahun dari Pexels API.
Kata kunci: before botox, unretouched face, before dermal filler, raw portrait.

Penggunaan:
    python pexels_search_40_65.py YOUR_API_KEY
"""

import requests
import os
import sys
import time
import random
import hashlib
import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
BASE_DIR = SCRIPT_DIR

# Query khusus usia 40-65 tahun
QUERIES = [
    "before botox face natural mature woman close up 40s",
    "before botox face natural mature man close up 40s",
    "unretouched face close up mature woman portrait wrinkles",
    "unretouched face close up mature man portrait wrinkles",
    "before dermal filler face mature woman natural 50s",
    "before dermal filler face mature man natural 50s",
    "raw portrait photography natural lighting mature woman face",
    "raw portrait photography natural lighting mature man face",
    "middle aged woman face close up no makeup natural skin",
    "middle aged man face close up natural headshot",
    "older woman face portrait close up natural no filter 40",
    "older man face portrait close up natural no filter 40",
    "mature caucasian woman face headshot natural aging",
    "mature caucasian man face headshot natural aging",
    "woman 40s 50s face close up portrait natural no makeup",
    "man 40s 50s face close up portrait natural headshot",
    "aging face woman natural skin close up portrait",
    "aging face man natural skin close up portrait",
    "middle age woman portrait front view close up",
    "middle age man portrait front view close up",
]


def get_all_used_ids_and_hashes():
    """Kumpulkan semua Pexels ID dan MD5 hash yang sudah digunakan."""
    used_ids = {
        3861592, 3373716, 3431739, 9718932, 4079215, 1222271,
        30450838, 30496628, 614810, 1043474, 2379005, 1681010,
        91227, 775358, 846741, 834863, 3785079, 1516680,
        3760856, 3763188, 3785424, 1181686, 3769021,
        2182970, 2589653, 3211476, 1212984, 2380794, 3785077,
        1300402, 2379004, 1933873, 2955376, 2340978, 1121796,
        1722198, 697509, 220453, 3760583,
        3760514, 3760529, 3764119, 1239291, 774909, 3760607,
        3762804, 3760573, 3760263, 1587009, 3760811, 3760809,
        2379003, 3777943, 3778603, 3778876, 3778680, 3771089,
        3778212, 3778966, 3770317, 3778014, 3776932, 3771836,
        3768911,
        # Batch dari pexels_search.py run sebelumnya
        10194766, 7298876, 15603012, 5438476, 10521294, 17388022,
        8727383, 8727558, 8875609, 18160382, 5945335, 13357913,
        24913614, 16106146, 7597466, 5863100, 31630384, 12446392,
        32588096, 11125350, 4816518, 14138750, 24430436,
        24913619, 19138993, 17492272, 13346836, 6337527, 32224555,
        7719521, 31428193,
        8244560, 10648948, 7298625, 1552543, 9421385, 13771127,
        6659419, 7994381, 20068197, 18191488,
        15716139, 15098950, 32651782, 9256869, 3762776, 16041479,
        6659420, 32651790, 15486158, 8675181, 4939925, 11526914,
        10985506, 19799245, 10040264, 7222370, 18525514, 8783504,
        34775253, 8917969, 31870729, 8727474, 8964675, 8964676,
        36675191, 10194769, 33249721, 6335086, 9775435, 8964679,
        8875610, 10918892, 34775442, 4787918, 26125922, 31954387,
        7298926, 8727377, 36613594, 8964040, 7302958, 6625959,
        8964677, 10520767,
    }

    # Scan semua MD5 hash dari gambar yang sudah ada
    existing_hashes = set()
    for folder in ["contoh_wanita/25-39", "contoh_wanita/40-65",
                    "contoh_pria/25-39", "contoh_pria/40-65"]:
        d = os.path.join(BASE_DIR, folder)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.jpg'):
                    filepath = os.path.join(d, f)
                    with open(filepath, "rb") as fh:
                        existing_hashes.add(hashlib.md5(fh.read()).hexdigest())

    return used_ids, existing_hashes


def is_front_view_close_up(img_path, landmarker):
    """Cek apakah gambar front view dan close-up."""
    img = cv2.imread(img_path)
    if img is None:
        return False, "Tidak bisa dibaca"

    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return False, "Wajah tidak terdeteksi"

    landmarks = result.face_landmarks[0]
    face_xs = [lm.x for lm in landmarks]
    face_ys = [lm.y for lm in landmarks]
    face_width = max(face_xs) - min(face_xs)

    if face_width < 0.25:
        return False, f"Wajah terlalu kecil ({face_width:.1%})"

    nose_x = landmarks[1].x
    left_eye_x = landmarks[33].x
    right_eye_x = landmarks[263].x
    face_center_x = (left_eye_x + right_eye_x) / 2
    symmetry = abs(nose_x - face_center_x) / face_width

    if symmetry > 0.08:
        return False, f"Bukan front view (asimetri: {symmetry:.3f})"

    dist_left = abs(nose_x - left_eye_x)
    dist_right = abs(right_eye_x - nose_x)
    if dist_left > 0 and dist_right > 0:
        ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        if ratio < 0.7:
            return False, f"Wajah menoleh (rasio: {ratio:.2f})"

    if len(result.face_landmarks) > 1:
        return False, "Lebih dari 1 wajah"

    face_center_x_abs = (min(face_xs) + max(face_xs)) / 2
    if abs(face_center_x_abs - 0.5) > 0.25:
        return False, "Wajah tidak di tengah"

    return True, f"OK (lebar={face_width:.1%}, simetri={symmetry:.3f})"


def download_and_filter(api_key, target_women=10, target_men=10):
    """Download dan filter gambar usia 40-65 dari Pexels."""

    headers = {"Authorization": api_key}
    women_dir = os.path.join(BASE_DIR, "contoh_wanita", "40-65")
    men_dir = os.path.join(BASE_DIR, "contoh_pria", "40-65")

    os.makedirs(women_dir, exist_ok=True)
    os.makedirs(men_dir, exist_ok=True)

    existing_women = len([f for f in os.listdir(women_dir) if f.endswith('.jpg')])
    existing_men = len([f for f in os.listdir(men_dir) if f.endswith('.jpg')])

    women_count = 0
    men_count = 0
    women_idx = existing_women + 1
    men_idx = existing_men + 1

    # Tentukan nama file berikutnya (cari nomor tertinggi)
    for f in os.listdir(women_dir):
        if f.endswith('.jpg'):
            try:
                num = int(f.replace('wanita_', '').replace('.jpg', ''))
                women_idx = max(women_idx, num + 1)
            except ValueError:
                pass
    for f in os.listdir(men_dir):
        if f.endswith('.jpg'):
            try:
                num = int(f.replace('pria_', '').replace('.jpg', ''))
                men_idx = max(men_idx, num + 1)
            except ValueError:
                pass

    print(f"=== Pencarian Wajah Usia 40-65 Tahun ===")
    print(f"Target: {target_women} wanita, {target_men} pria")
    print(f"Sudah ada: {existing_women} wanita, {existing_men} pria")
    print(f"Akan mulai dari: wanita_{women_idx}, pria_{men_idx}")
    print()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=2,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    used_ids, existing_hashes = get_all_used_ids_and_hashes()
    total_checked = 0
    total_downloaded = 0

    queries = list(QUERIES)
    random.shuffle(queries)

    for query in queries:
        if women_count >= target_women and men_count >= target_men:
            break

        print(f"\n{'='*60}")
        print(f"Query: \"{query}\"")
        print(f"{'='*60}")

        q_lower = query.lower()
        is_woman_query = "woman" in q_lower or "wanita" in q_lower
        is_man_query = (" man " in f" {q_lower} " or q_lower.endswith(" man") or "pria" in q_lower) and not is_woman_query

        if is_woman_query and women_count >= target_women:
            print("  Skip - sudah cukup wanita")
            continue
        if is_man_query and men_count >= target_men:
            print("  Skip - sudah cukup pria")
            continue

        for page in range(1, 5):  # Max 4 pages per query
            if women_count >= target_women and men_count >= target_men:
                break

            url = "https://api.pexels.com/v1/search"
            params = {
                "query": query,
                "per_page": 15,
                "page": page,
                "orientation": "portrait",
                "size": "medium",
            }

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                if resp.status_code != 200:
                    print(f"  API error: {resp.status_code}")
                    break

                data = resp.json()
                photos = data.get("photos", [])

                if not photos:
                    print(f"  Page {page}: tidak ada hasil")
                    break

                print(f"  Page {page}: {len(photos)} foto ditemukan")

                for photo in photos:
                    photo_id = photo["id"]

                    if photo_id in used_ids:
                        continue

                    used_ids.add(photo_id)
                    img_url = photo["src"]["large"]
                    tmp_path = os.path.join(BASE_DIR, f"_tmp_{photo_id}.jpg")

                    try:
                        img_resp = requests.get(img_url, timeout=15)
                        if img_resp.status_code != 200:
                            continue

                        with open(tmp_path, "wb") as f:
                            f.write(img_resp.content)

                        # Cek duplikat via MD5
                        with open(tmp_path, "rb") as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        if file_hash in existing_hashes:
                            os.remove(tmp_path)
                            print(f"    [{photo_id}] SKIP - Duplikat (MD5)")
                            continue

                        total_checked += 1
                        is_valid, reason = is_front_view_close_up(tmp_path, landmarker)

                        if not is_valid:
                            os.remove(tmp_path)
                            print(f"    [{photo_id}] SKIP - {reason}")
                            continue

                        existing_hashes.add(file_hash)

                        if is_woman_query and women_count < target_women:
                            dest = os.path.join(women_dir, f"wanita_{women_idx}.jpg")
                            os.rename(tmp_path, dest)
                            women_count += 1
                            print(f"    [{photo_id}] SAVED wanita_{women_idx} - {reason}")
                            women_idx += 1
                            total_downloaded += 1
                        elif is_man_query and men_count < target_men:
                            dest = os.path.join(men_dir, f"pria_{men_idx}.jpg")
                            os.rename(tmp_path, dest)
                            men_count += 1
                            print(f"    [{photo_id}] SAVED pria_{men_idx} - {reason}")
                            men_idx += 1
                            total_downloaded += 1
                        else:
                            os.remove(tmp_path)

                    except Exception as e:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        continue

                time.sleep(0.5)

            except Exception as e:
                print(f"  Request error: {e}")
                time.sleep(1)

    landmarker.close()

    print(f"\n{'='*60}")
    print(f"SELESAI!")
    print(f"  Total dicek: {total_checked}")
    print(f"  Total disimpan: {total_downloaded}")
    print(f"  Wanita 40-65 baru: {women_count} (total: {existing_women + women_count})")
    print(f"  Pria 40-65 baru: {men_count} (total: {existing_men + men_count})")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python pexels_search_40_65.py YOUR_PEXELS_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    download_and_filter(api_key, target_women=10, target_men=10)
