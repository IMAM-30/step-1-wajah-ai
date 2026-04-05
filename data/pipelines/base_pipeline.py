"""
Base pipeline template for facial part extraction.

Child pipelines only need to define:
  GENDER, PART, LANDMARKS, PAD, and optionally a custom crop_part() function.

Usage in child pipeline:
  from base_pipeline import BasePipeline

  pipeline = BasePipeline(
      gender="wanita",
      part="hidung",
      landmarks=[1, 2, 3, ...],
      pad=0.25,
      port=6001,
  )
  pipeline.run()
"""

import os
import sys
import shutil
import cv2
import numpy as np
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory
from mediapipe.python.solutions import face_mesh as mp_face_mesh

EXTENSIONS = (".jpg", ".jpeg", ".png")
MIN_CROP = 30
PER_PAGE = 10

# ─── HTML Template ────────────────────────────────────────────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline: {{ gender }} - {{ part }}</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { margin-bottom: 5px; }
        .info { color: #666; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }
        .card { background: white; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .card img { width: 100%; height: 180px; object-fit: cover; border-radius: 4px; cursor: pointer; }
        .card .fname { font-size: 13px; font-weight: bold; margin: 6px 0 8px; }
        .actions { display: flex; gap: 6px; }
        .actions form { flex: 1; }
        .btn { width: 100%; padding: 6px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
        .btn-approve { background: #22c55e; color: white; }
        .btn-reject { background: #ef4444; color: white; }
        .btn-approve:hover { background: #16a34a; }
        .btn-reject:hover { background: #dc2626; }
        .bulk-bar { margin-bottom: 16px; display: flex; gap: 8px; align-items: center; }
        .bulk-bar button { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
        .pagination { margin-top: 20px; display: flex; gap: 8px; align-items: center; }
        .pagination a { padding: 6px 12px; background: white; border-radius: 4px; text-decoration: none; color: #333; }
        .pagination .current { padding: 6px 12px; background: #333; color: white; border-radius: 4px; }
        .empty { text-align: center; padding: 60px; color: #999; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                 background: rgba(0,0,0,0.85); z-index: 999; justify-content: center; align-items: center; }
        .modal.active { display: flex; }
        .modal img { max-width: 90%; max-height: 90%; object-fit: contain; }
    </style>
</head>
<body>
    <h1>Pipeline: {{ gender }} / {{ part }}</h1>
    <p class="info">{{ total }} crops pending &middot; Page {{ page }}/{{ total_pages }}</p>

    {% if images %}
    <form method="POST" action="/bulk" id="bulkForm">
        <input type="hidden" name="page" value="{{ page }}">
        <div class="bulk-bar">
            <label><input type="checkbox" id="selectAll"> Select all</label>
            <button type="submit" name="decision" value="approve" class="btn btn-approve">Approve selected</button>
            <button type="submit" name="decision" value="reject" class="btn btn-reject">Reject selected</button>
        </div>
        <div class="grid">
            {% for fname in images %}
            <div class="card">
                <img src="/staging/{{ fname }}" alt="{{ fname }}" onclick="openModal(this.src)">
                <div class="fname">{{ fname }}</div>
                <input type="checkbox" name="selected" value="{{ fname }}">
                <div class="actions">
                    <button type="button" class="btn btn-approve" onclick="doAction('{{ fname }}','approve')">Approve</button>
                    <button type="button" class="btn btn-reject" onclick="doAction('{{ fname }}','reject')">Reject</button>
                </div>
            </div>
            {% endfor %}
        </div>
    </form>
    <form method="POST" action="/action" id="actionForm" style="display:none">
        <input type="hidden" name="filename" id="af_filename">
        <input type="hidden" name="decision" id="af_decision">
        <input type="hidden" name="page" value="{{ page }}">
    </form>
    <div class="pagination">
        {% if page > 1 %}<a href="?page={{ page - 1 }}">&larr; Prev</a>{% endif %}
        {% for p in range(1, total_pages + 1) %}
            {% if p == page %}<span class="current">{{ p }}</span>
            {% else %}<a href="?page={{ p }}">{{ p }}</a>{% endif %}
        {% endfor %}
        {% if page < total_pages %}<a href="?page={{ page + 1 }}">Next &rarr;</a>{% endif %}
    </div>
    {% else %}
    <div class="empty"><p>No crops to review. Run with --process first.</p></div>
    {% endif %}

    <div class="modal" id="modal" onclick="this.classList.remove('active')">
        <img id="modalImg" src="">
    </div>
    <script>
        document.getElementById('selectAll')?.addEventListener('change', function() {
            document.querySelectorAll('input[name="selected"]').forEach(cb => cb.checked = this.checked);
        });
        function openModal(src) {
            document.getElementById('modalImg').src = src;
            document.getElementById('modal').classList.add('active');
        }
        function doAction(fname, decision) {
            document.getElementById('af_filename').value = fname;
            document.getElementById('af_decision').value = decision;
            document.getElementById('actionForm').submit();
        }
        document.getElementById('bulkForm')?.addEventListener('submit', function(e) {
            var clicked = e.submitter;
            if (clicked && clicked.name) {
                var h = document.createElement('input');
                h.type = 'hidden'; h.name = clicked.name; h.value = clicked.value;
                this.appendChild(h);
            }
            this.querySelectorAll('button').forEach(btn => {
                btn.disabled = true;
                btn.style.opacity = '0.5';
            });
        });
    </script>
</body>
</html>
"""


class BasePipeline:
    def __init__(self, gender, part, landmarks, pad=0.25, port=6001):
        self.gender = gender
        self.part = part
        self.landmarks = landmarks
        self.pad = pad
        self.port = port

        # resolve project root (3 levels up from data/pipelines/{gender}/)
        self.base = os.path.abspath(os.path.join(
            os.path.dirname(sys.argv[0]), "..", "..", ".."
        ))

        self.input_dir = os.path.join(self.base, "data", "raw_approved", gender)
        self.staging_dir = os.path.join(self.base, "data", "pipelines", gender, f".staging_{part}")
        self.skipped_dir = os.path.join(self.base, "data", "pipelines", gender, f".skipped_{part}")
        self.approved_dir = os.path.join(self.base, "data", "dataset", gender, "approved", part)
        self.rejected_dir = os.path.join(self.base, "data", "dataset", gender, "reject", part)

    # ─── Face detection ───────────────────────────────────────────────────
    def get_landmarks(self, image_rgb, mesh):
        result = mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = image_rgb.shape[:2]
        lm = result.multi_face_landmarks[0].landmark
        return np.array([(int(p.x * w), int(p.y * h)) for p in lm])

    # ─── Crop (override in child for custom logic) ────────────────────────
    def crop_part(self, image, landmarks):
        h, w = image.shape[:2]
        pts = landmarks[self.landmarks]
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        px, py = int(bw * self.pad), int(bh * self.pad)

        x1 = max(0, x_min - px)
        y1 = max(0, y_min - py)
        x2 = min(w, x_max + px)
        y2 = min(h, y_max + py)

        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
            return None
        return crop

    # ─── Output filename: originalname_part.jpg ───────────────────────────
    def make_output_name(self, original_filename):
        name = os.path.splitext(original_filename)[0]
        return f"{name}_{self.part}.jpg"

    # ─── Collect images from input (supports age subfolders) ────────────
    def _collect_images(self):
        """Scan input_dir for images. Supports flat and age-group subfolders."""
        images = []
        if not os.path.exists(self.input_dir):
            return images

        # Direct images in input_dir
        for f in sorted(os.listdir(self.input_dir)):
            fp = os.path.join(self.input_dir, f)
            if os.path.isfile(fp) and f.lower().endswith(EXTENSIONS):
                images.append({"path": fp, "fname": f, "age_group": ""})

        # Age-group subfolders (25-39, 40-65)
        for age in ["25-39", "40-65"]:
            age_dir = os.path.join(self.input_dir, age)
            if os.path.isdir(age_dir):
                for f in sorted(os.listdir(age_dir)):
                    fp = os.path.join(age_dir, f)
                    if os.path.isfile(fp) and f.lower().endswith(EXTENSIONS):
                        images.append({"path": fp, "fname": f, "age_group": age})

        return images

    # ─── Process all images ───────────────────────────────────────────────
    def process(self):
        for d in [self.staging_dir, self.skipped_dir]:
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        images = self._collect_images()
        if not images:
            print(f"No images in {self.input_dir}")
            return

        total = len(images)
        success = 0
        skipped = 0

        print(f"Pipeline: {self.gender}-{self.part}")
        print(f"Processing {total} images from {self.input_dir}")
        print("-" * 50)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            for item in images:
                img_path = item["path"]
                fname = item["fname"]
                age_group = item["age_group"]

                img = cv2.imread(img_path)
                if img is None:
                    print(f"  [SKIP] Cannot read: {fname}")
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                landmarks = self.get_landmarks(rgb, mesh)

                if landmarks is None:
                    print(f"  [SKIP] No face: {fname}")
                    shutil.copy2(img_path, os.path.join(self.skipped_dir, fname))
                    skipped += 1
                    continue

                crop = self.crop_part(img, landmarks)
                if crop is None:
                    print(f"  [SKIP] Crop too small: {fname}")
                    shutil.copy2(img_path, os.path.join(self.skipped_dir, fname))
                    skipped += 1
                    continue

                out_name = self.make_output_name(fname)
                # Tag with age group in filename: name_part_25-39.jpg
                if age_group:
                    base, ext = os.path.splitext(out_name)
                    out_name = f"{base}_{age_group}{ext}"

                cv2.imwrite(
                    os.path.join(self.staging_dir, out_name),
                    crop, [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                success += 1
                age_tag = f" [{age_group}]" if age_group else ""
                print(f"  [OK] {fname} → {out_name}{age_tag}")

        print("-" * 50)
        print(f"SUMMARY: total={total}  success={success}  skipped={skipped}")
        print(f"  Staging: {self.staging_dir}")
        print(f"  Skipped: {self.skipped_dir}")

    # ─── Flask UI ─────────────────────────────────────────────────────────
    def _get_staged(self):
        if not os.path.exists(self.staging_dir):
            return []
        return sorted(f for f in os.listdir(self.staging_dir) if f.lower().endswith(EXTENSIONS))

    def _safe_dest(self, dst_dir, filename):
        path = os.path.join(dst_dir, filename)
        if not os.path.exists(path):
            return path
        name, ext = os.path.splitext(filename)
        n = 1
        while os.path.exists(os.path.join(dst_dir, f"{name}_{n}{ext}")):
            n += 1
        return os.path.join(dst_dir, f"{name}_{n}{ext}")

    def _extract_age_group(self, filename):
        """Extract age group from filename tag: name_part_25-39.jpg → '25-39'"""
        for age in ["25-39", "40-65"]:
            if f"_{age}" in filename:
                return age
        return ""

    def _move_staged(self, filename, decision):
        src = os.path.join(self.staging_dir, filename)
        if not os.path.exists(src):
            print(f"[SKIP] File already moved: {filename}")
            return

        # Route to age-aware folder if age tag present
        age_group = self._extract_age_group(filename)
        if decision == "approve":
            if age_group:
                dst_dir = os.path.join(self.base, "data", "dataset", self.gender, age_group, "approved", self.part)
            else:
                dst_dir = self.approved_dir
        else:
            if age_group:
                dst_dir = os.path.join(self.base, "data", "dataset", self.gender, age_group, "reject", self.part)
            else:
                dst_dir = self.rejected_dir

        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, filename)
        if os.path.exists(dst):
            print(f"[SKIP] Already exists in destination: {filename}")
            os.remove(src)
            return

        shutil.move(src, dst)
        print(f"[MOVE] {filename} → {decision}")

    def serve(self):
        pipeline = self
        app = Flask(__name__)

        @app.route("/")
        def index():
            page = request.args.get("page", 1, type=int)
            images = pipeline._get_staged()
            total = len(images)
            total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
            page = min(page, total_pages)
            start = (page - 1) * PER_PAGE
            return render_template_string(
                TEMPLATE,
                images=images[start : start + PER_PAGE],
                page=page, total_pages=total_pages, total=total,
                gender=pipeline.gender, part=pipeline.part,
            )

        @app.route("/staging/<filename>")
        def serve_staging(filename):
            return send_from_directory(pipeline.staging_dir, filename)

        @app.route("/action", methods=["POST"])
        def action():
            filename = request.form.get("filename")
            decision = request.form.get("decision")
            if filename and decision:
                pipeline._move_staged(filename, decision)
            return redirect(url_for("index", page=request.form.get("page", 1, type=int)))

        @app.route("/bulk", methods=["POST"])
        def bulk():
            decision = request.form.get("decision")
            if not decision:
                return redirect(url_for("index", page=request.form.get("page", 1, type=int)))
            for fname in request.form.getlist("selected"):
                pipeline._move_staged(fname, decision)
            return redirect(url_for("index", page=request.form.get("page", 1, type=int)))

        app.run(debug=True, port=self.port)

    # ─── CLI entry point ──────────────────────────────────────────────────
    def run(self):
        if "--process" in sys.argv:
            self.process()
        elif "--serve" in sys.argv:
            self.serve()
        else:
            print(f"Pipeline: {self.gender}-{self.part}")
            print(f"  Input:    {self.input_dir}")
            print(f"  Staging:  {self.staging_dir}")
            print(f"  Skipped:  {self.skipped_dir}")
            print(f"  Approved: {self.approved_dir}")
            print(f"  Rejected: {self.rejected_dir}")
            print()
            print("Usage:")
            print(f"  python {self.gender}-{self.part}.py --process")
            print(f"  python {self.gender}-{self.part}.py --serve")
