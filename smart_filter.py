"""
Smart Filter — ML-based quality filter for Wajah AI.

Two levels:
  Level 1: Score foto FULL sebelum crop (apakah layak di-crop?)
           → Belajar dari: foto yang mayoritas potongannya di-approve vs reject
  Level 2: Score tiap POTONGAN setelah crop (apakah kualitas crop bagus?)
           → 7 model terpisah (hidung, mata, bibir, dagu, rambut, telinga, baju)

Tiga tier output:
  confidence >= 0.75  → AUTO-APPROVE (langsung masuk batch)
  confidence <= 0.25  → AUTO-REJECT  (langsung dibuang)
  0.25 < conf < 0.75  → REVIEW       (masuk staging, review manual)

Auto re-train setiap ada data baru dari review manual.
"""

import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "data", "models", "smart_filter")
DATASET_DIR = os.path.join(BASE, "data", "dataset")
EXTENSIONS = (".jpg", ".jpeg", ".png")
PARTS = ["hidung", "mata", "bibir", "dagu", "rambut", "telinga", "baju"]

# Confidence thresholds
THRESH_APPROVE = 0.75
THRESH_REJECT = 0.25


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — 12 fitur per gambar
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(img_path):
    """
    Extract 12 visual features dari satu gambar.
    Returns numpy array of features, or None if failed.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # 1. Brightness (mean luminance)
    brightness = np.mean(gray)

    # 2. Contrast (std luminance)
    contrast = np.std(gray)

    # 3. Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 4. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 5-6. Color saturation & value
    sat_mean = np.mean(hsv[:, :, 1])
    val_mean = np.mean(hsv[:, :, 2])

    # 7. Aspect ratio
    aspect = w / max(h, 1)

    # 8. Image area (normalized)
    area = (w * h) / 1e6  # in megapixels

    # 9. Dark pixel ratio (< 30)
    dark_ratio = np.sum(gray < 30) / gray.size

    # 10. Bright pixel ratio (> 230)
    bright_ratio = np.sum(gray > 230) / gray.size

    # 11. Color diversity (std of hue)
    hue_std = np.std(hsv[:, :, 0])

    # 12. Blur score (inverse of sharpness, normalized)
    blur = 1.0 / (1.0 + sharpness / 500.0)

    return np.array([
        brightness, contrast, sharpness, edge_density,
        sat_mean, val_mean, aspect, area,
        dark_ratio, bright_ratio, hue_std, blur
    ])


FEATURE_NAMES = [
    "brightness", "contrast", "sharpness", "edge_density",
    "saturation", "value", "aspect_ratio", "area_mpx",
    "dark_ratio", "bright_ratio", "hue_std", "blur_score"
]


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECT TRAINING DATA — dari folder dataset yang sudah di-approve/reject
# ═══════════════════════════════════════════════════════════════════════════════

def collect_training_data(part=None):
    """
    Kumpulkan training data dari dataset.
    Jika part=None → Level 1 (semua gambar campur)
    Jika part='hidung' → Level 2 khusus hidung

    Returns: (X, y) — features array dan labels (1=approved, 0=reject)
    """
    X, y = [], []

    for root, dirs, files in os.walk(DATASET_DIR):
        imgs = [f for f in files if f.lower().endswith(EXTENSIONS)]
        if not imgs:
            continue

        parts = root.replace(DATASET_DIR + "/", "").split("/")
        # batch_N/gender/age/decision/part_name
        if len(parts) < 5:
            continue

        decision = parts[3]  # approved or reject
        part_name = parts[4]

        # Filter by part jika Level 2
        if part is not None and part_name != part:
            continue

        label = 1 if decision == "approved" else 0

        for fname in imgs:
            fpath = os.path.join(root, fname)
            features = extract_features(fpath)
            if features is not None:
                X.append(features)
                y.append(label)

    if X:
        return np.array(X), np.array(y)
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN — buat/update model
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(part=None):
    """
    Train satu model.
    part=None → Level 1 (general quality)
    part='hidung' → Level 2 khusus hidung

    Returns: (accuracy, model_path) atau (None, None) jika data kurang
    """
    label = part if part else "general"
    print(f"\n  Training model: {label}")

    X, y = collect_training_data(part)
    if X is None or len(X) < 20:
        print(f"    Skip — data kurang ({0 if X is None else len(X)} samples, min 20)")
        return None, None

    n_approve = np.sum(y == 1)
    n_reject = np.sum(y == 0)
    print(f"    Data: {len(X)} samples (approved={n_approve}, reject={n_reject})")

    if n_reject < 5:
        print(f"    Skip — reject terlalu sedikit ({n_reject}, min 5)")
        return None, None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train GradientBoosting — lebih akurat dari RandomForest untuk data kecil
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=3,
        random_state=42,
    )

    # Cross-validation score
    if len(X) >= 30 and n_reject >= 10:
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
        accuracy = scores.mean()
        print(f"    CV Accuracy: {accuracy:.1%} (±{scores.std():.1%})")
    else:
        accuracy = 0.0
        print(f"    Data terlalu kecil untuk CV, training langsung")

    # Train on full data
    model.fit(X_scaled, y)

    # Save model + scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model_{label}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{label}.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"    Saved: {model_path}")

    # Feature importance
    importances = model.feature_importances_
    top3 = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:3]
    print(f"    Top features: {', '.join(f'{n}({v:.2f})' for n, v in top3)}")

    return accuracy, model_path


def train_all():
    """Train semua model: 1 general + 7 per-part."""
    print("=" * 60)
    print("  SMART FILTER — TRAINING")
    print("=" * 60)

    results = {}

    # Level 1: General model (semua potongan campur)
    acc, path = train_model(part=None)
    results["general"] = {"accuracy": acc, "path": path}

    # Level 2: Per-part models
    for part in PARTS:
        acc, path = train_model(part=part)
        results[part] = {"accuracy": acc, "path": path}

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  {'Model':<12} {'Accuracy':<12} {'Status'}")
    print(f"{'─' * 60}")
    for name, r in results.items():
        if r["accuracy"] is not None:
            print(f"  {name:<12} {r['accuracy']:>8.1%}     TRAINED")
        else:
            print(f"  {name:<12} {'—':>8}     SKIPPED")

    trained = sum(1 for r in results.values() if r["path"])
    print(f"\n  {trained}/8 models trained")
    print("=" * 60)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT — score gambar baru
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(label):
    """Load model + scaler. Returns (model, scaler) or (None, None)."""
    model_path = os.path.join(MODEL_DIR, f"model_{label}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{label}.pkl")
    if not os.path.exists(model_path):
        return None, None
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        return None, None


def predict(img_path, part=None):
    """
    Score satu gambar.
    Returns: (confidence, tier)
      confidence: 0.0 - 1.0 (probabilitas approved)
      tier: 'auto_approve', 'review', 'auto_reject'

    Jika model belum ada → return (0.5, 'review') — semua masuk review
    """
    label = part if part else "general"
    model, scaler = _load_model(label)

    if model is None:
        return 0.5, "review"

    features = extract_features(img_path)
    if features is None:
        return 0.0, "auto_reject"

    X = scaler.transform(features.reshape(1, -1))
    proba = model.predict_proba(X)[0]

    # Index 1 = probability of "approved"
    confidence = proba[1] if len(proba) > 1 else proba[0]

    if confidence >= THRESH_APPROVE:
        return confidence, "auto_approve"
    elif confidence <= THRESH_REJECT:
        return confidence, "auto_reject"
    else:
        return confidence, "review"


def predict_batch(image_paths, part=None):
    """
    Score banyak gambar sekaligus (lebih cepat dari predict satu-satu).
    Returns: list of (path, confidence, tier)
    """
    label = part if part else "general"
    model, scaler = _load_model(label)

    results = []
    if model is None:
        for p in image_paths:
            results.append((p, 0.5, "review"))
        return results

    for p in image_paths:
        features = extract_features(p)
        if features is None:
            results.append((p, 0.0, "auto_reject"))
            continue

        X = scaler.transform(features.reshape(1, -1))
        proba = model.predict_proba(X)[0]
        confidence = proba[1] if len(proba) > 1 else proba[0]

        if confidence >= THRESH_APPROVE:
            tier = "auto_approve"
        elif confidence <= THRESH_REJECT:
            tier = "auto_reject"
        else:
            tier = "review"

        results.append((p, confidence, tier))

    return results


def filter_staging(gender, part):
    """
    Filter semua gambar di staging folder untuk satu part.
    Auto-approve, auto-reject, atau keep di staging (review).

    Returns: dict {auto_approve: N, auto_reject: N, review: N}
    """
    from batch_manager import batch_move

    staging_dir = os.path.join(BASE, "data", "pipelines", gender, f".staging_{part}")
    if not os.path.exists(staging_dir):
        return {"auto_approve": 0, "auto_reject": 0, "review": 0}

    images = [f for f in os.listdir(staging_dir) if f.lower().endswith(EXTENSIONS)]
    if not images:
        return {"auto_approve": 0, "auto_reject": 0, "review": 0}

    counts = {"auto_approve": 0, "auto_reject": 0, "review": 0}

    for fname in images:
        fpath = os.path.join(staging_dir, fname)
        confidence, tier = predict(fpath, part=part)

        if tier == "auto_approve":
            # Extract age group dari filename
            age_group = "25-39"
            for age in ["25-39", "40-65"]:
                if f"_{age}" in fname:
                    age_group = age
                    break

            batch_move(fpath, gender, age_group, part, "approved",
                      original_filename=fname)
            counts["auto_approve"] += 1
            print(f"    [AUTO-APPROVE] {fname} (conf={confidence:.2f})")

        elif tier == "auto_reject":
            age_group = "25-39"
            for age in ["25-39", "40-65"]:
                if f"_{age}" in fname:
                    age_group = age
                    break

            batch_move(fpath, gender, age_group, part, "reject",
                      original_filename=fname)
            counts["auto_reject"] += 1
            print(f"    [AUTO-REJECT] {fname} (conf={confidence:.2f})")

        else:
            # Keep in staging for manual review
            counts["review"] += 1

    return counts


def filter_all_staging(gender):
    """
    Filter semua staging parts untuk satu gender.
    Returns: total counts dict
    """
    total = {"auto_approve": 0, "auto_reject": 0, "review": 0}

    print(f"\n  [SMART FILTER] Processing {gender}...")
    for part in PARTS:
        counts = filter_staging(gender, part)
        for k in total:
            total[k] += counts[k]

    a, r, rv = total["auto_approve"], total["auto_reject"], total["review"]
    print(f"\n  [SMART FILTER] Done: {a} auto-approved, {r} auto-rejected, {rv} need review")
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO RE-TRAIN — dipanggil setelah ada data baru dari review manual
# ═══════════════════════════════════════════════════════════════════════════════

COUNTER_FILE = os.path.join(MODEL_DIR, ".last_data_count")


def _load_counter():
    """Load last known data count dari file (persistent across restart)."""
    try:
        if os.path.exists(COUNTER_FILE):
            with open(COUNTER_FILE) as f:
                return int(f.read().strip())
    except Exception:
        pass
    return 0


def _save_counter(count):
    """Simpan data count ke file."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))


def maybe_retrain():
    """
    Cek apakah perlu re-train (ada data baru sejak training terakhir).
    Auto re-train jika data bertambah >= 50 dari terakhir kali.
    Counter disimpan ke file agar persistent walau dashboard restart.
    """
    last_count = _load_counter()

    # Hitung total gambar di dataset
    count = 0
    for root, dirs, files in os.walk(DATASET_DIR):
        count += len([f for f in files if f.lower().endswith(EXTENSIONS)])

    diff = count - last_count
    if diff >= 50:
        print(f"\n[SMART FILTER] Data bertambah {diff} (total {count}) → re-training...")
        train_all()
        _save_counter(count)
        return True
    else:
        print(f"[SMART FILTER] Data baru: {diff} (butuh 50 untuk re-train, total {count})")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_all()
    elif len(sys.argv) > 1 and sys.argv[1] == "--stats":
        if not os.path.exists(MODEL_DIR):
            print("No models trained yet. Run: python smart_filter.py --train")
        else:
            models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and f.startswith('model_')]
            print(f"Trained models: {len(models)}")
            for m in sorted(models):
                print(f"  {m}")
    else:
        print("Usage:")
        print("  python smart_filter.py --train   Train all models")
        print("  python smart_filter.py --stats   Show model status")
