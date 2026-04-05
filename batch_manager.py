"""
Batch Manager — Auto batch system for Wajah AI dataset.

Rules:
- Each batch subfolder holds max 100 images.
- When a subfolder reaches 100, overflow to next batch (fill remaining first).
- Batch numbering: sequential cycling 1→2→...→10→1→2→...
  Selalu maju: setelah batch_6 = batch_7, setelah batch_10 = batch_1.
- Max 5 active batches at any time.
  → Saat batch ke-6 dibuat, batch TERTUA dihapus otomatis.
  → Selalu hanya 5 batch folder yang tampil.
- Registry (Excel) TIDAK PERNAH dihapus — tetap utuh sebagai anti-duplikasi.
  → Kolom 'status' dan 'batch' di-update saat approve/reject.
- File naming: {gender}_{age}_{part}_{NNN}.jpg
  e.g. pria_25-39_hidung_001.jpg (counter 001-100, reset per batch).

Usage:
    from batch_manager import batch_move
    batch_move(src_path, gender, age_group, part, decision)
"""

import os
import re
import shutil
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "data", "dataset")
REGISTRY_PATH = os.path.join(BASE, "registry.xlsx")
EXTENSIONS = (".jpg", ".jpeg", ".png")
BATCH_LIMIT = 100   # max gambar per subfolder per batch
MAX_BATCHES = 5      # max batch aktif yang tampil
MAX_BATCH_NUM = 10   # nomor batch cycling 1-10
TRAINING_MODE = False # False = auto-cycling aktif (batch tertua dihapus otomatis)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _count_images(folder):
    """Hitung jumlah file gambar di folder."""
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])


def _get_existing_batches():
    """
    Dapatkan list batch yang ada, sorted by nomor.
    Returns: [(batch_num, batch_name, batch_path), ...]
    """
    batches = []
    if not os.path.exists(DATASET_DIR):
        return batches
    for name in os.listdir(DATASET_DIR):
        m = re.match(r"^batch_(\d+)$", name)
        if m:
            num = int(m.group(1))
            batches.append((num, name, os.path.join(DATASET_DIR, name)))
    return sorted(batches, key=lambda x: x[0])


def _get_next_batch_num(existing_batches):
    """
    Nomor batch berikutnya.
    - Normal mode: sequential cycling 1→2→...→10→1 (max 10)
    - Training mode: terus naik tanpa batas (batch_11, batch_12, dst)
    """
    if not existing_batches:
        return 1
    highest = max(b[0] for b in existing_batches)
    if TRAINING_MODE:
        return highest + 1  # terus naik, tanpa batas
    # Normal: wrap 10→1
    return (highest % MAX_BATCH_NUM) + 1


def _auto_cleanup(need_room=1):
    """
    Jaga agar batch aktif + batch baru tidak lebih dari MAX_BATCHES.
    Hapus batch TERTUA (nomor urut paling kecil yang ada) untuk memberi ruang.
    HANYA hapus folder batch — registry Excel TIDAK disentuh.

    Jika TRAINING_MODE = True, cleanup dinonaktifkan agar data ML terkumpul.
    """
    if TRAINING_MODE:
        print(f"[BATCH] Training mode ON — auto-cleanup dinonaktifkan, semua batch disimpan")
        return

    batches = _get_existing_batches()
    while len(batches) + need_room > MAX_BATCHES:
        oldest_num, oldest_name, oldest_path = batches[0]
        print(f"[BATCH CLEANUP] Menghapus folder {oldest_name} (menjaga max {MAX_BATCHES} batch aktif)")
        print(f"  → Data di registry.xlsx TETAP tersimpan (anti-duplikasi)")
        shutil.rmtree(oldest_path)
        batches = batches[1:]


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY UPDATE — update Excel saat approve/reject
# ═══════════════════════════════════════════════════════════════════════════════

def _update_registry(original_filename, status, batch_name, new_filename):
    """
    Update registry.xlsx: set status, batch, dan nama file baru.
    Cari berdasarkan nama file original (img_XXXX.jpg).
    Registry TIDAK PERNAH dihapus — hanya di-update.
    """
    if not os.path.exists(REGISTRY_PATH):
        return

    try:
        df = pd.read_excel(REGISTRY_PATH)
    except Exception:
        return

    if df.empty or "filename" not in df.columns:
        return

    # Cari baris berdasarkan original filename (img_0041.jpg)
    # Staging filename format: img_0041_hidung_25-39.jpg → extract img_0041
    base_name = original_filename
    # Coba extract base: ambil bagian img_XXXX
    m = re.match(r"(img_\d+)", original_filename)
    if m:
        base_name = m.group(1)

    mask = df["filename"].astype(str).str.contains(base_name, na=False)
    if mask.any():
        # Pastikan kolom ada dan bertipe string (bukan float64)
        for col in ["status", "batch", "batch_filename"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        df.loc[mask, "status"] = status
        df.loc[mask, "batch"] = batch_name
        df.loc[mask, "batch_filename"] = new_filename
        df.to_excel(REGISTRY_PATH, index=False)
        print(f"[REGISTRY] {base_name} → status={status}, batch={batch_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIND BATCH — cari batch yang tepat untuk file baru
# ═══════════════════════════════════════════════════════════════════════════════

def _find_batch(gender, age_group, part, decision):
    """
    Cari batch yang masih ada slot untuk subfolder ini.
    Logika: cek batch yang sudah ada → isi yang belum penuh dulu → baru buat baru.

    Returns: (folder_path, next_number, batch_num)
    """
    batches = _get_existing_batches()

    # Cek batch yang sudah ada — isi yang belum penuh dulu
    for batch_num, batch_name, batch_path in batches:
        folder = os.path.join(batch_path, gender, age_group, decision, part)
        count = _count_images(folder)
        if count < BATCH_LIMIT:
            return folder, count + 1, batch_num

    # Semua batch penuh (atau belum ada) — buat batch baru
    # Auto cleanup dulu — hapus folder tertua jika perlu
    _auto_cleanup(need_room=1)

    # Nomor batch baru = sequential maju dari tertinggi
    # Re-read batches setelah cleanup
    batches = _get_existing_batches()
    new_num = _get_next_batch_num(batches)
    new_name = f"batch_{new_num}"
    folder = os.path.join(DATASET_DIR, new_name, gender, age_group, decision, part)

    print(f"[BATCH NEW] Membuat {new_name}")
    return folder, 1, new_num


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FILENAME
# ═══════════════════════════════════════════════════════════════════════════════

def _build_filename(gender, age_group, part, number, ext=".jpg"):
    """
    Format nama file standar.
    Contoh: pria_25-39_hidung_001.jpg
    """
    return f"{gender}_{age_group}_{part}_{number:03d}{ext}"


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH MOVE — fungsi utama yang dipanggil dari dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def batch_move(src_path, gender, age_group, part, decision, original_filename=None):
    """
    Pindahkan file ke batch yang tepat dengan penamaan otomatis.

    Alur:
    1. Cari batch yang masih ada slot (< 100) untuk subfolder ini
    2. Jika semua penuh → hapus batch tertua → buat batch baru (sequential)
    3. Rename file sesuai format standar
    4. Pindahkan ke folder tujuan
    5. Update registry.xlsx (status + batch)

    Args:
        src_path:          Path lengkap file sumber
        gender:            "pria" atau "wanita"
        age_group:         "25-39" atau "40-65"
        part:              "hidung", "mata", "bibir", "dagu", "rambut", "telinga", "baju"
        decision:          "approved" atau "reject"
        original_filename: Nama file asli dari staging (untuk update registry)

    Returns:
        (dst_path, batch_num) atau (None, None) jika file tidak ada
    """
    if not os.path.exists(src_path):
        return None, None

    # Ekstensi file
    _, ext = os.path.splitext(src_path)
    if not ext:
        ext = ".jpg"

    # Cari batch yang tepat
    dst_dir, number, batch_num = _find_batch(gender, age_group, part, decision)
    os.makedirs(dst_dir, exist_ok=True)

    # Bangun nama file baru
    new_name = _build_filename(gender, age_group, part, number, ext.lower())
    dst_path = os.path.join(dst_dir, new_name)

    # Safety: jika file sudah ada, cari nomor berikutnya
    while os.path.exists(dst_path):
        number += 1
        if number > BATCH_LIMIT:
            # Subfolder ini penuh, cari/buat batch berikutnya
            dst_dir, number, batch_num = _find_batch(gender, age_group, part, decision)
            os.makedirs(dst_dir, exist_ok=True)
        new_name = _build_filename(gender, age_group, part, number, ext.lower())
        dst_path = os.path.join(dst_dir, new_name)

    # Pindahkan file
    shutil.move(src_path, dst_path)
    batch_name = f"batch_{batch_num}"
    print(f"[BATCH] {new_name} → {batch_name}/{gender}/{age_group}/{decision}/{part}/")

    # Update registry Excel
    if original_filename:
        _update_registry(original_filename, decision, batch_name, new_name)

    return dst_path, batch_num


# ═══════════════════════════════════════════════════════════════════════════════
# STATS — lihat ringkasan semua batch
# ═══════════════════════════════════════════════════════════════════════════════

def get_batch_stats():
    """
    Ringkasan semua batch aktif.
    Returns list of dicts: [{batch, gender, age, decision, part, count}, ...]
    """
    stats = []
    for batch_num, batch_name, batch_path in _get_existing_batches():
        for gender in ["pria", "wanita"]:
            for age in ["25-39", "40-65"]:
                for decision in ["approved", "reject"]:
                    for part in ["hidung", "mata", "bibir", "dagu", "rambut", "telinga", "baju"]:
                        folder = os.path.join(batch_path, gender, age, decision, part)
                        count = _count_images(folder)
                        if count > 0:
                            stats.append({
                                "batch": batch_name,
                                "batch_num": batch_num,
                                "gender": gender,
                                "age": age,
                                "decision": decision,
                                "part": part,
                                "count": count,
                                "full": count >= BATCH_LIMIT,
                            })
    return stats


def get_batch_summary():
    """Ringkasan singkat per batch — total file dan status."""
    batches = _get_existing_batches()
    summary = []
    for batch_num, batch_name, batch_path in batches:
        total = 0
        full_parts = 0
        total_parts = 0
        for root, dirs, files in os.walk(batch_path):
            imgs = [f for f in files if f.lower().endswith(EXTENSIONS)]
            if imgs:
                total += len(imgs)
                total_parts += 1
                if len(imgs) >= BATCH_LIMIT:
                    full_parts += 1
        summary.append({
            "batch": batch_name,
            "batch_num": batch_num,
            "total_images": total,
            "full_parts": full_parts,
            "total_parts": total_parts,
        })
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — jalankan langsung untuk lihat statistik
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    batches = _get_existing_batches()
    if not batches:
        print("Dataset kosong. Belum ada batch.")
        print(f"\nKonfigurasi:")
        print(f"  Max per subfolder : {BATCH_LIMIT} gambar")
        print(f"  Max batch aktif   : {MAX_BATCHES}")
        print(f"  Nomor batch       : 1 - {MAX_BATCH_NUM} (cycling sequential)")
        raise SystemExit(0)

    # Ringkasan per batch
    print("=" * 60)
    print("  BATCH MANAGER — Dataset Summary")
    print("=" * 60)
    summary = get_batch_summary()
    for s in summary:
        status = "ACTIVE" if s["total_images"] > 0 else "EMPTY"
        print(f"\n  {s['batch']} — {s['total_images']} gambar ({s['full_parts']}/{s['total_parts']} subfolder penuh) [{status}]")

    # Detail per subfolder
    stats = get_batch_stats()
    if stats:
        print(f"\n{'─' * 60}")
        print(f"  {'Batch':<10} {'Gender':<8} {'Age':<6} {'Status':<10} {'Part':<10} {'Count':<7} {'Full'}")
        print(f"{'─' * 60}")
        for s in stats:
            full_mark = "PENUH" if s["full"] else ""
            print(f"  {s['batch']:<10} {s['gender']:<8} {s['age']:<6} {s['decision']:<10} {s['part']:<10} {s['count']:>3}/100 {full_mark}")

    # Registry info
    if os.path.exists(REGISTRY_PATH):
        try:
            reg = pd.read_excel(REGISTRY_PATH)
            print(f"\n  Registry: {len(reg)} total entries (PERMANENT — tidak pernah dihapus)")
        except Exception:
            pass

    print(f"\n  Batch aktif: {len(batches)}/{MAX_BATCHES}")
    print(f"  Slot tersedia: {MAX_BATCHES - len(batches)}")
    print("=" * 60)
