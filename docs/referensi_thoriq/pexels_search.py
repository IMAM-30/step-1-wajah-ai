"""
Script otomatis pencarian & download gambar wajah dari Pexels API.
Mencari wajah natural, front view, close-up, Caucasian/Slavic, usia 25-40.

Penggunaan:
    python pexels_search.py YOUR_API_KEY
"""

import requests
import os
import sys
import time
import random
import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
BASE_DIR = SCRIPT_DIR

QUERIES = [
    "before botox face natural woman close up",
    "before botox face natural man close up",
    "unretouched face close up woman portrait",
    "unretouched face close up man portrait",
    "before dermal filler face woman natural",
    "before dermal filler face man natural",
    "raw portrait photography natural lighting woman face",
    "raw portrait photography natural lighting man face",
    "caucasian woman face close up no makeup natural skin",
    "caucasian man face close up natural headshot",
    "slavic woman face portrait close up natural",
    "slavic man face portrait close up natural",
    "woman face front view plain background natural",
    "man face front view plain background natural",
    "woman headshot no makeup natural skin close up",
    "man headshot natural close up portrait plain background",
]

# Pexels IDs yang SUDAH digunakan sebelumnya (anti duplikat)
USED_PEXELS_IDS = {
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
    3768911, 3760583,
}


def is_front_view_close_up(img_path, landmarker):
    """
    Cek apakah gambar menampilkan wajah front view dan close-up.
    Returns: (is_valid, reason)
    """
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

    # 1. Cek apakah wajah cukup besar (close-up) — minimal 25% lebar gambar
    face_xs = [lm.x for lm in landmarks]
    face_ys = [lm.y for lm in landmarks]
    face_width = max(face_xs) - min(face_xs)
    face_height = max(face_ys) - min(face_ys)

    if face_width < 0.25:
        return False, f"Wajah terlalu kecil ({face_width:.1%} lebar)"

    # 2. Cek front view — bandingkan simetri kiri-kanan
    # Landmark 33 = mata kiri luar, 263 = mata kanan luar, 1 = ujung hidung
    nose_x = landmarks[1].x
    left_eye_x = landmarks[33].x
    right_eye_x = landmarks[263].x

    face_center_x = (left_eye_x + right_eye_x) / 2
    symmetry = abs(nose_x - face_center_x) / face_width

    if symmetry > 0.08:
        return False, f"Bukan front view (asimetri: {symmetry:.3f})"

    # 3. Cek apakah wajah menghadap kamera (bukan menoleh)
    # Jarak mata kiri ke hidung vs mata kanan ke hidung harus mirip
    dist_left = abs(nose_x - left_eye_x)
    dist_right = abs(right_eye_x - nose_x)
    if dist_left > 0 and dist_right > 0:
        ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        if ratio < 0.7:
            return False, f"Wajah menoleh (rasio: {ratio:.2f})"

    # 4. Cek apakah ada lebih dari 1 wajah
    if len(result.face_landmarks) > 1:
        return False, "Lebih dari 1 wajah"

    # 5. Cek wajah di tengah gambar
    face_center_x_abs = (min(face_xs) + max(face_xs)) / 2
    if abs(face_center_x_abs - 0.5) > 0.25:
        return False, "Wajah tidak di tengah"

    return True, f"OK (lebar={face_width:.1%}, simetri={symmetry:.3f})"


def download_and_filter(api_key, target_women=10, target_men=10):
    """Download dan filter gambar dari Pexels."""

    headers = {"Authorization": api_key}
    women_dir = os.path.join(BASE_DIR, "contoh_wanita")
    men_dir = os.path.join(BASE_DIR, "contoh_pria")

    # Hitung file yang sudah ada
    existing_women = len([f for f in os.listdir(women_dir) if f.endswith('.jpg')])
    existing_men = len([f for f in os.listdir(men_dir) if f.endswith('.jpg')])

    women_count = 0
    men_count = 0
    women_idx = existing_women + 1
    men_idx = existing_men + 1

    print(f"Target: {target_women} wanita, {target_men} pria")
    print(f"Sudah ada: {existing_women} wanita, {existing_men} pria")
    print(f"Akan mulai dari: wanita_{women_idx}, pria_{men_idx}")
    print()

    # Init face landmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=2,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    downloaded_ids = set(USED_PEXELS_IDS)
    total_checked = 0
    total_downloaded = 0

    # Shuffle queries
    queries = list(QUERIES)
    random.shuffle(queries)

    for query in queries:
        if women_count >= target_women and men_count >= target_men:
            break

        print(f"\n{'='*60}")
        print(f"Query: \"{query}\"")
        print(f"{'='*60}")

        # Determine gender from query
        q_lower = query.lower()
        is_woman_query = "woman" in q_lower or "wanita" in q_lower
        is_man_query = (" man " in f" {q_lower} " or "man " in q_lower.split("wo")[-1] or "pria" in q_lower) and not is_woman_query

        if is_woman_query and women_count >= target_women:
            print("  Skip - sudah cukup wanita")
            continue
        if is_man_query and men_count >= target_men:
            print("  Skip - sudah cukup pria")
            continue

        for page in range(1, 4):  # Max 3 pages per query
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

                    if photo_id in downloaded_ids:
                        continue

                    downloaded_ids.add(photo_id)

                    # Download medium size
                    img_url = photo["src"]["large"]
                    tmp_path = os.path.join(BASE_DIR, f"_tmp_{photo_id}.jpg")

                    try:
                        img_resp = requests.get(img_url, timeout=15)
                        if img_resp.status_code != 200:
                            continue

                        with open(tmp_path, "wb") as f:
                            f.write(img_resp.content)

                        total_checked += 1

                        # Validasi dengan face detection
                        is_valid, reason = is_front_view_close_up(tmp_path, landmarker)

                        if not is_valid:
                            os.remove(tmp_path)
                            print(f"    [{photo_id}] SKIP - {reason}")
                            continue

                        # Tentukan gender berdasarkan query
                        if is_woman_query and women_count < target_women:
                            dest = os.path.join(women_dir, f"wanita_{women_idx}.jpg")
                            os.rename(tmp_path, dest)
                            women_count += 1
                            women_idx += 1
                            total_downloaded += 1
                            print(f"    [{photo_id}] SAVED wanita_{women_idx-1} - {reason}")
                        elif is_man_query and men_count < target_men:
                            dest = os.path.join(men_dir, f"pria_{men_idx}.jpg")
                            os.rename(tmp_path, dest)
                            men_count += 1
                            men_idx += 1
                            total_downloaded += 1
                            print(f"    [{photo_id}] SAVED pria_{men_idx-1} - {reason}")
                        else:
                            os.remove(tmp_path)

                    except Exception as e:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        continue

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"  Request error: {e}")
                time.sleep(1)

    landmarker.close()

    print(f"\n{'='*60}")
    print(f"SELESAI!")
    print(f"  Total dicek: {total_checked}")
    print(f"  Total disimpan: {total_downloaded}")
    print(f"  Wanita baru: {women_count} (total: {existing_women + women_count})")
    print(f"  Pria baru: {men_count} (total: {existing_men + men_count})")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python pexels_search.py YOUR_PEXELS_API_KEY")
        print()
        print("Dapatkan API key gratis di: https://www.pexels.com/api/")
        sys.exit(1)

    api_key = sys.argv[1]
    download_and_filter(api_key, target_women=10, target_men=10)
