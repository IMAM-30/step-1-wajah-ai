"""
Wajah AI — First Time Setup
Jalankan file ini sekali saja sebelum menggunakan dashboard.

Usage:
    python setup.py
"""

import subprocess
import sys
import os

# Naik satu level dari docs/ ke root project
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, desc):
    print(f"\n  [{desc}]")
    result = subprocess.run(cmd, shell=True, cwd=BASE)
    if result.returncode != 0:
        print(f"  GAGAL: {desc}")
        sys.exit(1)
    print(f"  OK")


def main():
    print("=" * 50)
    print("  Wajah AI — First Time Setup")
    print("=" * 50)

    # 1. Buat virtual environment
    venv_path = os.path.join(BASE, "venv")
    if not os.path.exists(venv_path):
        run(f"{sys.executable} -m venv venv", "Membuat virtual environment")
    else:
        print("\n  [Virtual environment sudah ada, skip]")

    # 2. Tentukan pip path
    if sys.platform == "win32":
        pip = os.path.join(venv_path, "Scripts", "pip")
        python = os.path.join(venv_path, "Scripts", "python")
    else:
        pip = os.path.join(venv_path, "bin", "pip")
        python = os.path.join(venv_path, "bin", "python")

    # 3. Install dependencies
    run(f"{pip} install --upgrade pip", "Upgrade pip")
    run(f"{pip} install -r requirements.txt", "Install dependencies")

    # 4. Buat folder yang diperlukan
    folders = [
        "data/raw_images",
        "data/raw_approved/pria",
        "data/raw_approved/wanita",
        "data/dataset",
    ]
    for folder in folders:
        path = os.path.join(BASE, folder)
        os.makedirs(path, exist_ok=True)
    print("\n  [Membuat folder struktur data]")
    print("  OK")

    # 5. Verifikasi model files
    print("\n  [Memeriksa model files]")
    models = [
        "data/models/age_deploy.prototxt",
        "data/models/age_net.caffemodel",
        "data/models/gender_deploy.prototxt",
        "data/models/gender_net.caffemodel",
        "data/models/smart_filter/model_general.pkl",
    ]
    all_ok = True
    for m in models:
        path = os.path.join(BASE, m)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    {m} ({size // 1024}KB)")
        else:
            print(f"    MISSING: {m}")
            all_ok = False

    if not all_ok:
        print("\n  WARNING: Beberapa model file tidak ditemukan!")
        print("  Program tetap bisa jalan tapi fitur tertentu mungkin tidak aktif.")

    # 6. Test import
    print("\n  [Test import modules]")
    try:
        result = subprocess.run(
            [python, "-c", "import flask, cv2, numpy, pandas, sklearn, mediapipe, imagehash; print('Semua library OK')"],
            capture_output=True, text=True, cwd=BASE
        )
        if result.returncode == 0:
            print(f"    {result.stdout.strip()}")
        else:
            print(f"    ERROR: {result.stderr.strip()[:200]}")
            print("    Coba install ulang: pip install -r requirements.txt")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Done
    print("\n" + "=" * 50)
    print("  Setup selesai!")
    print()
    print("  Jalankan dashboard:")
    if sys.platform == "win32":
        print("    venv\\Scripts\\python dashboard.py")
    else:
        print("    source venv/bin/activate")
        print("    python dashboard.py")
    print()
    print("  Buka browser: http://127.0.0.1:8000")
    print("=" * 50)


if __name__ == "__main__":
    main()
