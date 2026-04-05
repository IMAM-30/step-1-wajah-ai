# Wajah AI — Face Dataset Pipeline

## Quick Start

```bash
# 1. Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan dashboard
python dashboard.py
```

Buka browser: http://127.0.0.1:8000

## Cara Pakai

1. Masukkan keyword pencarian (contoh: `pria face natural headshot close up`)
   - Harus ada kata **pria** atau **wanita** di keyword
2. Masukkan jumlah gambar yang diminta
3. Klik **Start** — sistem otomatis:
   - Scrape gambar dari 20 sumber
   - Filter: wajah frontal, gender, usia, AI detection, duplikat
   - Crop 7 bagian wajah (hidung, mata, bibir, dagu, rambut, telinga, baju)
   - Smart Filter ML auto-approve/reject
4. Review hasil di 3 tab: **Approved / Rejected / Review**
   - Koreksi jika ada yang salah (approve/reject per gambar)
5. Klik **Finish** — data tersimpan ke batch, ML auto re-train jika ada 50+ data baru

## Struktur Folder

```
dashboard.py          — Web UI (Flask)
scraper.py            — Multi-source scraper + filter pipeline
smart_filter.py       — ML auto-approve/reject (8 model)
batch_manager.py      — Auto-batch + naming + cycling
data/
  models/             — Age/Gender DNN + Smart Filter ML models
  pipelines/          — 14 crop pipelines (7 pria + 7 wanita)
  dataset/            — Output: batch_N/{gender}/{age}/approved|reject/{part}/
  raw_images/         — Temporary download folder
registry.xlsx         — Anti-duplikasi URL tracking
```

## ML Models (Sudah Trained)

8 model Smart Filter sudah trained (82-88% akurasi) dari 2231 data. Siap pakai langsung, akan makin pintar seiring penggunaan.
