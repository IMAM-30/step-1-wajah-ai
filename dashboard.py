"""
Wajah AI Dashboard — Full auto flow system.
Flow: Scrape → Auto Process → Auto Serve → Review → Finish → Home
"""

import os
import json
import subprocess
import sys
import signal
import atexit
import time as _time
import shutil
from datetime import datetime
from flask import (Flask, render_template_string, request, redirect,
                   url_for, send_from_directory)
from batch_manager import batch_move
from smart_filter import filter_all_staging, maybe_retrain

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "data", ".state.json")
DATASET_DIR = os.path.join(BASE, "data", "dataset")
EXTENSIONS = (".jpg", ".jpeg", ".png")
PARTS = ["hidung", "mata", "bibir", "dagu", "rambut", "telinga", "baju"]


# ═══════════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════════

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {"pipelines": {}, "processing": False, "phase": "idle", "gender": None}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def is_port_in_use(port):
    try:
        r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=3)
        return bool(r.stdout.strip())
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_pipelines(gender=None):
    pipes = []
    genders = [gender] if gender else ["wanita", "pria"]
    for g in genders:
        folder = os.path.join(BASE, "data", "pipelines", g)
        if not os.path.exists(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".py") and fname not in ("__init__.py", "base_pipeline.py"):
                name = fname.replace(".py", "")
                port = 0
                try:
                    with open(os.path.join(folder, fname)) as f:
                        for line in f:
                            if "port=" in line:
                                port = int(line.split("port=")[1].split(",")[0].split(")")[0].strip())
                                break
                except Exception:
                    pass
                pipes.append({"name": name, "path": os.path.join(folder, fname),
                              "gender": g, "port": port,
                              "part": name.split("-")[-1] if "-" in name else name})
    return pipes


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])


def count_staging(gender):
    total = 0
    for part in PARTS:
        d = os.path.join(BASE, "data", "pipelines", gender, f".staging_{part}")
        if os.path.exists(d):
            total += len([f for f in os.listdir(d) if f.lower().endswith(EXTENSIONS)])
    return total


def detect_gender(query):
    q = query.lower()
    if any(w in q for w in ["wanita", "female", "woman", "women", "girl"]):
        return "wanita"
    if any(w in q for w in ["pria", "male", "man ", "men ", "boy"]):
        return "pria"
    return "unknown"


def kill_port(port):
    """Force-kill semua proses yang menempati port tertentu."""
    if not port or port == 0:
        return
    try:
        r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=3)
        for pid_str in r.stdout.strip().split("\n"):
            pid_str = pid_str.strip()
            if pid_str.isdigit():
                pid = int(pid_str)
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
    except Exception:
        pass


# Track semua child PIDs agar bisa cleanup saat exit
_child_pids = set()


def start_pipeline_serve(name):
    state = load_state()
    ps = state.get("pipelines", {}).get(name, {})
    if ps.get("running") and is_process_alive(ps.get("pid")):
        return
    for p in find_pipelines():
        if p["name"] == name:
            # Force-free port dulu — kill orphan process jika ada
            if p["port"] and is_port_in_use(p["port"]):
                kill_port(p["port"])
                _time.sleep(0.2)

            proc = subprocess.Popen(
                [sys.executable, p["path"], "--serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _child_pids.add(proc.pid)
            state.setdefault("pipelines", {})[name] = {"running": True, "pid": proc.pid, "port": p["port"]}
            save_state(state)
            print(f"[SERVE START] {name} pid={proc.pid} port={p['port']}")
            return


def stop_pipeline_serve(name):
    state = load_state()
    ps = state.get("pipelines", {}).get(name, {})
    pid = ps.get("pid")
    port = ps.get("port", 0)

    # Kill by PID
    if pid and is_process_alive(pid):
        try:
            os.kill(pid, 15)
            _time.sleep(0.1)
            if is_process_alive(pid):
                os.kill(pid, 9)
        except Exception:
            pass
        _child_pids.discard(pid)
        print(f"[SERVE STOP] {name} pid={pid}")

    # Kill by port — catch orphans yang PID-nya sudah berubah
    if port:
        kill_port(port)

    state.setdefault("pipelines", {})[name] = {"running": False, "pid": None, "port": port}
    save_state(state)


def stop_all_serves():
    """Stop semua pipeline serves — by PID + by port untuk catch orphans."""
    state = load_state()
    for name, ps in list(state.get("pipelines", {}).items()):
        pid = ps.get("pid")
        port = ps.get("port", 0)

        if pid and is_process_alive(pid):
            try:
                os.kill(pid, 15)
            except Exception:
                pass
            _child_pids.discard(pid)
            print(f"[SERVE STOP] {name}")

        # Juga kill by port untuk catch orphan processes
        if port:
            kill_port(port)

        state["pipelines"][name] = {"running": False, "pid": None, "port": port}
    save_state(state)


def _cleanup_on_exit():
    """Dipanggil saat dashboard mati — kill semua child processes."""
    print("\n[CLEANUP] Stopping all child processes...")
    for pid in list(_child_pids):
        try:
            os.kill(pid, 9)
        except Exception:
            pass
    # Kill by port dari state sebagai safety net
    try:
        state = load_state()
        for name, ps in state.get("pipelines", {}).items():
            port = ps.get("port", 0)
            if port:
                kill_port(port)
    except Exception:
        pass


atexit.register(_cleanup_on_exit)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Wajah AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: #0a0a0a; color: #e5e5e5;
               min-height: 100vh; display: flex; justify-content: center; align-items: center; }
        .bg { position: fixed; inset: 0; z-index: -1; overflow: hidden; }
        .bg .orb { position: absolute; border-radius: 50%; filter: blur(120px); opacity: 0.15; }
        .bg .orb-1 { width: 600px; height: 600px; background: #3b82f6; top: -200px; left: -100px; }
        .bg .orb-2 { width: 400px; height: 400px; background: #8b5cf6; bottom: -100px; right: -100px; }
        .container { text-align: center; width: 100%; max-width: 520px; padding: 40px 24px; }
        .logo { font-size: 42px; font-weight: 800; letter-spacing: -1px;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                margin-bottom: 6px; }
        .sub { color: #555; font-size: 13px; letter-spacing: 2px; text-transform: uppercase;
               margin-bottom: 48px; }
        form { display: flex; flex-direction: column; gap: 12px; }
        input[type="text"] { padding: 16px 20px; border-radius: 14px; border: 1px solid #222;
                             background: #141414; color: #fff; font-size: 15px; text-align: center;
                             font-family: 'Inter', sans-serif; transition: border-color 0.2s; }
        input[type="text"]:focus { border-color: #3b82f6; outline: none;
                                    box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
        input::placeholder { color: #444; }
        .row { display: flex; gap: 10px; }
        .row input { flex: 0 0 80px; text-align: center; }
        button { padding: 16px; border: none; border-radius: 14px; font-size: 15px;
                 cursor: pointer; font-weight: 700; flex: 1; font-family: 'Inter', sans-serif;
                 transition: all 0.2s; }
        .btn-go { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;
                  box-shadow: 0 4px 20px rgba(59,130,246,0.3); }
        .btn-go:hover { transform: translateY(-1px); box-shadow: 0 6px 25px rgba(59,130,246,0.4); }
        button:disabled { background: #222 !important; color: #444 !important; cursor: wait;
                          box-shadow: none !important; transform: none !important; }
        .error { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.2);
                 color: #f87171; border-radius: 12px; padding: 14px; margin-top: 20px; font-size: 13px; }
        .progress { margin-top: 24px; text-align: left; background: #141414; border: 1px solid #222;
                    border-radius: 14px; padding: 20px; font-size: 13px; line-height: 2.2; }
        .progress .ok { color: #22c55e; }
        .progress .ok::before { content: '\\2713 '; }
        .progress .run { color: #f59e0b; }
        .progress .run::before { content: '\\25CB '; }
        .progress .err { color: #ef4444; }
        .progress .err::before { content: '\\2717 '; }
        .meta { color: #333; font-size: 11px; margin-top: 28px; }
        #loading { display: none; margin-top: 28px; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #333;
                   border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.6s linear infinite;
                   vertical-align: middle; margin-right: 10px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .btn-review { display: block; margin-top: 24px; padding: 16px; text-decoration: none;
                      font-weight: 700; font-size: 15px; border-radius: 14px; color: #111;
                      background: linear-gradient(135deg, #22c55e, #16a34a);
                      box-shadow: 0 4px 20px rgba(34,197,94,0.3); transition: all 0.2s; }
        .btn-review:hover { transform: translateY(-1px); box-shadow: 0 6px 25px rgba(34,197,94,0.4); }
    </style>
</head>
<body>
    <div class="bg"><div class="orb orb-1"></div><div class="orb orb-2"></div></div>
    <div class="container">
        <div class="logo">Wajah AI</div>
        <p class="sub">Face Dataset Pipeline</p>

        <form method="POST" action="/go" id="goForm">
            <input type="text" name="query" placeholder="e.g. wanita face natural close up no makeup" required>
            <div class="row">
                <input type="text" name="limit" value="10" placeholder="n">
                <button type="submit" class="btn-go" id="goBtn">Start</button>
            </div>
        </form>

        <div id="loading">
            <span class="spinner"></span>
            <span style="color:#f59e0b;font-size:14px">Processing pipeline... this may take a moment</span>
        </div>

        {% if error %}<div class="error">{{ error }}</div>{% endif %}

        {% if results %}
        <div class="progress">
            {% for r in results %}
                <div class="{{ r.status }}">{{ r.msg }}</div>
            {% endfor %}
        </div>
        {% endif %}

        {% if show_review %}
        <a href="/review" class="btn-review">Review Images</a>
        {% endif %}

        {% if meta %}<div class="meta">{{ meta }}</div>{% endif %}
    </div>
    <script>
        document.getElementById('goForm').addEventListener('submit', function() {
            document.getElementById('goBtn').disabled = true;
            document.getElementById('goBtn').textContent = 'Working...';
            document.getElementById('loading').style.display = 'block';
        });
    </script>
</body>
</html>
"""


REVIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Review — {{ gender }} / {{ active_tab }} / {{ active_part }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: #0a0a0a; color: #e5e5e5; min-height: 100vh; }

        /* MAIN TABS — Approved / Rejected / Review */
        .main-tabs { display: flex; gap: 0; background: #0d0d0d; border-bottom: 1px solid #1a1a1a;
                     position: sticky; top: 0; z-index: 101; }
        .main-tabs a { flex: 1; text-align: center; padding: 14px 20px; color: #555;
                       text-decoration: none; font-size: 14px; font-weight: 700;
                       border-bottom: 3px solid transparent; transition: all 0.2s; }
        .main-tabs a:hover { color: #aaa; background: #111; }
        .main-tabs a.tab-approved { border-bottom-color: transparent; }
        .main-tabs a.tab-approved.active { color: #22c55e; border-bottom-color: #22c55e; background: #0a1a0f; }
        .main-tabs a.tab-rejected { border-bottom-color: transparent; }
        .main-tabs a.tab-rejected.active { color: #ef4444; border-bottom-color: #ef4444; background: #1a0a0a; }
        .main-tabs a.tab-review { border-bottom-color: transparent; }
        .main-tabs a.tab-review.active { color: #f59e0b; border-bottom-color: #f59e0b; background: #1a150a; }
        .main-tabs .tab-count { font-size: 11px; padding: 2px 8px; border-radius: 10px;
                                margin-left: 6px; font-weight: 600; }
        .tab-approved .tab-count { background: rgba(34,197,94,0.15); color: #22c55e; }
        .tab-rejected .tab-count { background: rgba(239,68,68,0.15); color: #ef4444; }
        .tab-review .tab-count { background: rgba(245,158,11,0.15); color: #f59e0b; }

        /* SUB-NAV — per body part */
        .subnav { display: flex; gap: 0; background: #111; border-bottom: 1px solid #1f1f1f;
                  overflow-x: auto; position: sticky; top: 49px; z-index: 100;
                  scrollbar-width: none; -ms-overflow-style: none; }
        .subnav::-webkit-scrollbar { display: none; }
        .subnav a { padding: 12px 18px; color: #555; text-decoration: none; font-size: 12px;
                    font-weight: 600; white-space: nowrap; border-bottom: 2px solid transparent;
                    transition: all 0.2s; text-transform: capitalize; }
        .subnav a:hover { color: #aaa; background: #141414; }
        .subnav a.active { color: #3b82f6; border-bottom-color: #3b82f6; background: #0d1321; }
        .subnav .count { font-size: 10px; background: #222; padding: 2px 6px; border-radius: 10px;
                         margin-left: 4px; color: #666; }
        .subnav a.active .count { background: rgba(59,130,246,0.2); color: #3b82f6; }

        /* TOP BAR */
        .top-bar { display: flex; justify-content: space-between; align-items: center;
                   padding: 12px 24px; background: #111; border-bottom: 1px solid #1f1f1f; }
        .top-bar .title { font-size: 13px; font-weight: 600; color: #888; }
        .top-bar .title strong { color: #fff; text-transform: capitalize; }
        .top-bar .actions { display: flex; gap: 8px; }
        .top-bar a, .top-bar button { padding: 8px 18px; border-radius: 8px; font-size: 12px;
                  font-weight: 600; cursor: pointer; text-decoration: none; border: none;
                  font-family: 'Inter', sans-serif; transition: all 0.15s; }
        .btn-finish { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;
                      box-shadow: 0 2px 10px rgba(59,130,246,0.25); }
        .btn-finish:hover { box-shadow: 0 4px 15px rgba(59,130,246,0.4); }
        .btn-home { background: #1a1a1a; color: #888; border: 1px solid #333; }
        .btn-home:hover { color: #fff; border-color: #555; }

        /* SMART FILTER SUMMARY */
        .sf-banner { display: flex; gap: 16px; padding: 10px 24px; background: #0d1117;
                     border-bottom: 1px solid #1f1f1f; font-size: 12px; font-weight: 600; }

        /* CONTENT */
        .content { padding: 24px; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
        .card { background: #141414; border-radius: 12px; padding: 10px; border: 1px solid #1f1f1f;
                transition: all 0.2s; }
        .card:hover { border-color: #333; transform: translateY(-2px);
                      box-shadow: 0 8px 25px rgba(0,0,0,0.4); }
        .card img { width: 100%; height: 170px; object-fit: cover; border-radius: 8px;
                    cursor: pointer; transition: opacity 0.2s; }
        .card img:hover { opacity: 0.85; }
        .card .fname { font-size: 9px; color: #444; margin: 6px 0 4px; word-break: break-all;
                       font-family: monospace; line-height: 1.3; }
        .card .actions { display: flex; gap: 6px; }
        .card .actions button { flex: 1; padding: 7px; border: none; border-radius: 8px;
                                cursor: pointer; font-size: 11px; font-weight: 600;
                                font-family: 'Inter', sans-serif; transition: all 0.15s; }
        .btn-approve { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3) !important; }
        .btn-approve:hover { background: #22c55e; color: #111; }
        .btn-reject { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2) !important; }
        .btn-reject:hover { background: #ef4444; color: white; }

        /* BULK BAR */
        .bulk-bar { display: flex; gap: 10px; align-items: center; margin-bottom: 20px;
                    background: #141414; padding: 12px 16px; border-radius: 12px; border: 1px solid #1f1f1f; }
        .bulk-bar label { font-size: 12px; color: #666; cursor: pointer; user-select: none; }
        .bulk-bar label input { margin-right: 6px; accent-color: #3b82f6; }
        .bulk-bar button { padding: 8px 18px; border: none; border-radius: 8px; font-size: 12px;
                           font-weight: 600; cursor: pointer; font-family: 'Inter', sans-serif; }

        /* EMPTY */
        .empty { text-align: center; padding: 80px 20px; }
        .empty-icon { font-size: 48px; margin-bottom: 16px; opacity: 0.3; }
        .empty-text { color: #444; font-size: 14px; }

        /* PAGINATION */
        .pagination { display: flex; gap: 6px; align-items: center; margin-top: 24px;
                      justify-content: center; flex-wrap: wrap; }
        .pagination a { padding: 8px 14px; background: #141414; border: 1px solid #222;
                        border-radius: 8px; text-decoration: none; color: #666; font-size: 12px;
                        font-weight: 600; transition: all 0.15s; }
        .pagination a:hover { color: #fff; border-color: #444; }
        .pagination .cur { background: #3b82f6; color: #fff; padding: 8px 14px;
                           border-radius: 8px; font-size: 12px; font-weight: 600; }

        /* MODAL */
        .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.95);
                 z-index: 999; justify-content: center; align-items: center; cursor: zoom-out; }
        .modal.active { display: flex; }
        .modal img { max-width: 90%; max-height: 90%; object-fit: contain; border-radius: 8px;
                     box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
    </style>
</head>
<body>
    <!-- MAIN TABS -->
    <div class="main-tabs">
        <a href="/review/approved/{{ active_part }}" class="tab-approved {{ 'active' if active_tab == 'approved' }}">
            Approved <span class="tab-count">{{ tab_totals.approved }}</span>
        </a>
        <a href="/review/rejected/{{ active_part }}" class="tab-rejected {{ 'active' if active_tab == 'rejected' }}">
            Rejected <span class="tab-count">{{ tab_totals.rejected }}</span>
        </a>
        <a href="/review/review/{{ active_part }}" class="tab-review {{ 'active' if active_tab == 'review' }}">
            Review <span class="tab-count">{{ tab_totals.review }}</span>
        </a>
    </div>

    <!-- SUB-NAV per body part -->
    <div class="subnav">
        {% for p in sub_parts %}
        <a href="/review/{{ active_tab }}/{{ p.name }}" class="{{ 'active' if p.name == active_part }}">
            {{ p.name }} <span class="count">{{ p.count }}</span>
        </a>
        {% endfor %}
    </div>

    <!-- TOP BAR -->
    <div class="top-bar">
        <div class="title">
            <strong>{{ gender }}</strong> / {{ active_tab }} / {{ active_part }} — {{ total }} images
        </div>
        <div class="actions">
            <a href="/" class="btn-home">Home</a>
            <form method="POST" action="/finish" style="display:inline">
                <button type="submit" class="btn-finish">Finish</button>
            </form>
        </div>
    </div>

    <!-- SMART FILTER SUMMARY -->
    {% if sf_summary %}
    <div class="sf-banner">
        <span style="color:#22c55e;">Auto-approved: {{ sf_summary.auto_approve }}</span>
        <span style="color:#ef4444;">Auto-rejected: {{ sf_summary.auto_reject }}</span>
        <span style="color:#f59e0b;">Need review: {{ sf_summary.review }}</span>
    </div>
    {% endif %}

    <!-- CONTENT -->
    <div class="content">
        {% if images %}
        <form method="POST" action="/review/{{ active_tab }}/{{ active_part }}/bulk" id="bulkForm">
            <input type="hidden" name="page" value="{{ page }}">
            <div class="bulk-bar">
                <label><input type="checkbox" id="selectAll"> Select all</label>
                {% if active_tab == 'review' %}
                <button type="submit" name="decision" value="approve" class="btn-approve">Approve selected</button>
                <button type="submit" name="decision" value="reject" class="btn-reject">Reject selected</button>
                {% elif active_tab == 'approved' %}
                <button type="submit" name="decision" value="reject" class="btn-reject">Reject selected</button>
                {% elif active_tab == 'rejected' %}
                <button type="submit" name="decision" value="approve" class="btn-approve">Approve selected</button>
                {% endif %}
            </div>

            <div class="grid">
                {% for img in images %}
                <div class="card">
                    <img src="/img/{{ active_tab }}/{{ active_part }}/{{ img }}" onclick="openModal(this.src)" loading="lazy">
                    <div class="fname">{{ img }}</div>
                    <input type="checkbox" name="selected" value="{{ img }}" style="accent-color:#3b82f6">
                    <div class="actions">
                        {% if active_tab == 'review' %}
                        <button type="button" class="btn-approve"
                            onclick="doAction('approve','{{ img }}')">Approve</button>
                        <button type="button" class="btn-reject"
                            onclick="doAction('reject','{{ img }}')">Reject</button>
                        {% elif active_tab == 'approved' %}
                        <button type="button" class="btn-reject"
                            onclick="doAction('reject','{{ img }}')">Reject</button>
                        {% elif active_tab == 'rejected' %}
                        <button type="button" class="btn-approve"
                            onclick="doAction('approve','{{ img }}')">Approve</button>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </form>

        <form method="POST" action="/review/{{ active_tab }}/{{ active_part }}/action" id="actionForm" style="display:none">
            <input type="hidden" name="filename" id="af_filename">
            <input type="hidden" name="decision" id="af_decision">
            <input type="hidden" name="page" value="{{ page }}">
        </form>

        <div class="pagination">
            {% if page > 1 %}
            <a href="/review/{{ active_tab }}/{{ active_part }}?page={{ page-1 }}">Prev</a>
            {% endif %}
            {% for p in range(1, total_pages+1) %}
                {% if p == page %}<span class="cur">{{ p }}</span>
                {% elif p <= 3 or p >= total_pages - 2 or (p >= page - 1 and p <= page + 1) %}
                <a href="/review/{{ active_tab }}/{{ active_part }}?page={{ p }}">{{ p }}</a>
                {% elif p == 4 or p == total_pages - 3 %}
                <span style="color:#444">...</span>
                {% endif %}
            {% endfor %}
            {% if page < total_pages %}
            <a href="/review/{{ active_tab }}/{{ active_part }}?page={{ page+1 }}">Next</a>
            {% endif %}
        </div>
        {% else %}
        <div class="empty">
            <div class="empty-icon">&#9744;</div>
            <div class="empty-text">No images in {{ active_tab }} / {{ active_part }}.</div>
        </div>
        {% endif %}
    </div>

    <div class="modal" id="modal" onclick="this.classList.remove('active')">
        <img id="modalImg">
    </div>

    <script>
        document.getElementById('selectAll')?.addEventListener('change', function() {
            document.querySelectorAll('input[name="selected"]').forEach(cb => cb.checked = this.checked);
        });
        function openModal(src) {
            document.getElementById('modalImg').src = src;
            document.getElementById('modal').classList.add('active');
        }
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') document.getElementById('modal').classList.remove('active');
        });
        function doAction(decision, fname) {
            document.getElementById('af_filename').value = fname;
            document.getElementById('af_decision').value = decision;
            document.getElementById('actionForm').submit();
        }
        document.getElementById('bulkForm')?.addEventListener('submit', function(e) {
            var clicked = e.submitter;
            if (clicked && clicked.name) {
                var h = document.createElement('input');
                h.type='hidden'; h.name=clicked.name; h.value=clicked.value;
                this.appendChild(h);
            }
            this.querySelectorAll('button').forEach(btn => { btn.disabled=true; btn.style.opacity='0.5'; });
        });
    </script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    state = load_state()
    meta = None
    if state.get("last_process"):
        meta = f"Last run: {state['last_process']}"
    return render_template_string(HOME_TEMPLATE, error=None, results=None,
                                  show_review=False, meta=meta)


@app.route("/go", methods=["POST"])
def go():
    """FULL AUTO: scrape → move to approved → process → serve → show review button."""
    query = request.form.get("query", "").strip()
    limit_str = request.form.get("limit", "5").strip()
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 5

    if not query:
        return render_template_string(HOME_TEMPLATE, error="Please enter a search keyword.",
                                      results=None, show_review=False, meta=None)

    gender = detect_gender(query)
    if gender == "unknown":
        return render_template_string(HOME_TEMPLATE,
            error="Please include 'pria' or 'wanita' in your search keyword.",
            results=None, show_review=False, meta=None)

    log = []
    state = load_state()
    state["gender"] = gender
    state["phase"] = "scraping"
    state["process_start"] = _time.time()  # timestamp untuk filter review
    save_state(state)

    # ── Step 1: Scrape ────────────────────────────────────────────────────
    log.append({"status": "run", "icon": "1.", "msg": f"Scraping '{query}' (limit={limit})..."})
    print(f"\n[SCRAPE START] query='{query}' limit={limit} gender={gender}")

    # Build scraper command with optional Pexels key
    scrape_cmd = [sys.executable, "scraper.py", "--query", query, "--limit", str(limit)]
    pexels_key = os.environ.get("PEXELS_API_KEY", "5fRPEq4K5fQrVvxSKeOLoma6icKFxxqTRPIlkvaLxSTwMLv1YiTxZDlM")
    if pexels_key:
        scrape_cmd += ["--pexels-key", pexels_key]

    try:
        print(f"[DASHBOARD] Sending limit={limit} to scraper")
        result = subprocess.run(
            scrape_cmd,
            capture_output=True, text=True, timeout=3600,  # 1 jam max
        )
        print(result.stdout)
        if result.returncode != 0:
            log[-1] = {"status": "err", "icon": "1.", "msg": f"Scrape failed: {result.stderr[-100:]}"}
            return render_template_string(HOME_TEMPLATE, error=None, results=log,
                                          show_review=False, meta=None)
    except subprocess.TimeoutExpired:
        log[-1] = {"status": "err", "icon": "1.", "msg": "Scrape timed out"}
        return render_template_string(HOME_TEMPLATE, error=None, results=log,
                                      show_review=False, meta=None)

    scraped = count_images(os.path.join(BASE, "data", "raw_images"))
    log[-1] = {"status": "ok", "icon": "1.", "msg": f"Scraped images → data/raw_images/ ({scraped} files)"}
    print("[SCRAPE DONE]")

    # ── Step 2: Auto move to raw_approved/{gender}/{age_group}/ ─────────
    import pandas as pd
    src_dir = os.path.join(BASE, "data", "raw_images")
    registry_path = os.path.join(BASE, "registry.xlsx")
    age_map = {}
    if os.path.exists(registry_path):
        reg_df = pd.read_excel(registry_path)
        if "age_group" in reg_df.columns:
            for _, row in reg_df.iterrows():
                fn = str(row.get("filename", ""))
                ag = str(row.get("age_group", "25-39"))
                if fn and ag in ("25-39", "40-65"):
                    age_map[fn] = ag

    moved = 0
    age_counts = {"25-39": 0, "40-65": 0}
    if os.path.exists(src_dir):
        for f in os.listdir(src_dir):
            if f.lower().endswith(EXTENSIONS):
                age_group = age_map.get(f, "25-39")
                dst_dir = os.path.join(BASE, "data", "raw_approved", gender, age_group)
                os.makedirs(dst_dir, exist_ok=True)
                src = os.path.join(src_dir, f)
                dst = os.path.join(dst_dir, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                    moved += 1
                    age_counts[age_group] = age_counts.get(age_group, 0) + 1

    log.append({"status": "ok", "icon": "2.",
                "msg": f"Moved {moved} → {gender}/ (25-39: {age_counts.get('25-39',0)}, 40-65: {age_counts.get('40-65',0)})"})
    print(f"[MOVE] {moved} images → raw_approved/{gender}/ (ages: {age_counts})")

    # Store age info in state for pipeline routing
    state["age_counts"] = age_counts

    # ── Step 3: Process pipelines ─────────────────────────────────────────
    state["phase"] = "processing"
    save_state(state)

    pipelines = find_pipelines(gender)
    ok = 0
    for p in pipelines:
        print(f"[PROCESS] {p['name']}")
        try:
            r = subprocess.run(
                [sys.executable, p["path"], "--process"],
                capture_output=True, text=True, timeout=600,
            )
            summary = ""
            for line in r.stdout.split("\n"):
                if "success=" in line:
                    summary = line.strip()
            if r.returncode == 0:
                ok += 1
                print(f"[DONE] {p['name']} {summary}")
        except Exception as e:
            print(f"[ERROR] {p['name']} {e}")

    log.append({"status": "ok", "icon": "3.", "msg": f"Processed {ok}/{len(pipelines)} pipelines for {gender}"})

    # ── Step 4: Start serve for this gender ───────────────────────────────
    state["phase"] = "serving"
    save_state(state)

    for p in pipelines:
        start_pipeline_serve(p["name"])
        _time.sleep(0.3)

    log.append({"status": "ok", "icon": "4.", "msg": f"Started {len(pipelines)} pipeline servers"})

    # ── Step 5: Smart Filter — ML auto approve/reject ──────────────────
    state["phase"] = "filtering"
    save_state(state)

    print("[SMART FILTER] Running ML filter on staging...")
    sf_result = filter_all_staging(gender)
    auto_a = sf_result["auto_approve"]
    auto_r = sf_result["auto_reject"]
    need_review = sf_result["review"]
    log.append({"status": "ok", "icon": "5.",
                "msg": f"Smart Filter: {auto_a} auto-approved, {auto_r} auto-rejected, {need_review} need review"})

    # Simpan summary ke state agar bisa ditampilkan di review page
    state["sf_summary"] = {"auto_approve": auto_a, "auto_reject": auto_r, "review": need_review}
    save_state(state)

    # ── Step 6: Clean raw_approved ────────────────────────────────────────
    clean_dir = os.path.join(BASE, "data", "raw_approved", gender)
    if os.path.exists(clean_dir):
        for root, dirs, files in os.walk(clean_dir):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp) and not f.startswith("."):
                    os.remove(fp)

    # ── Done ──────────────────────────────────────────────────────────────
    state["phase"] = "review"
    state["last_process"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)

    staging_total = count_staging(gender)
    log.append({"status": "ok", "icon": "✓",
                "msg": f"Ready! {staging_total} crops need manual review (rest handled by ML)"})
    print(f"[ALL DONE] {staging_total} need review, {auto_a} auto-approved, {auto_r} auto-rejected")

    return render_template_string(HOME_TEMPLATE, error=None, results=log,
                                  show_review=True, meta=None)


# ═══════════════════════════════════════════════════════════════════════════════
# REVIEW UI — 3 tabs (approved/rejected/review) x 7 body parts
# ═══════════════════════════════════════════════════════════════════════════════

def get_staging_images(gender, part):
    """Gambar di staging (belum di-approve/reject)."""
    d = os.path.join(BASE, "data", "pipelines", gender, f".staging_{part}")
    if not os.path.exists(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTENSIONS))


def get_dataset_images(gender, part, decision, since=None):
    """
    Gambar dari dataset batches berdasarkan decision (approved/reject).
    Jika since (timestamp) diberikan, hanya return file yang mtime >= since.
    Returns list of filenames.
    """
    images = []
    for root, _dirs, files in os.walk(DATASET_DIR):
        if f"/{decision}/{part}" in root and f"/{gender}/" in root:
            for f in files:
                if f.lower().endswith(EXTENSIONS):
                    if since is not None:
                        fpath = os.path.join(root, f)
                        if os.path.getmtime(fpath) < since:
                            continue
                    images.append(f)
    return sorted(images)


def _find_dataset_file(gender, part, decision, filename):
    """Cari full path file di dataset batches."""
    for root, _dirs, files in os.walk(DATASET_DIR):
        if f"/{decision}/{part}" in root and f"/{gender}/" in root:
            if filename in files:
                return os.path.join(root, filename)
    return None


def count_by_tab(gender, since=None):
    """Hitung total per tab: approved, rejected, review (staging)."""
    counts = {"approved": 0, "rejected": 0, "review": 0}
    for part in PARTS:
        counts["review"] += len(get_staging_images(gender, part))
        counts["approved"] += len(get_dataset_images(gender, part, "approved", since))
        counts["rejected"] += len(get_dataset_images(gender, part, "reject", since))
    return counts


def parts_for_tab(gender, tab, since=None):
    """Bangun sub-nav info: list of {name, count} per part untuk tab tertentu."""
    info = []
    for p in PARTS:
        if tab == "review":
            c = len(get_staging_images(gender, p))
        elif tab == "approved":
            c = len(get_dataset_images(gender, p, "approved", since))
        else:
            c = len(get_dataset_images(gender, p, "reject", since))
        info.append({"name": p, "count": c})
    return info


# ── Review main route ────────────────────────────────────────────────────────

@app.route("/review")
@app.route("/review/<tab>")
@app.route("/review/<tab>/<part>")
def review(tab=None, part=None):
    state = load_state()
    gender = state.get("gender", "wanita")

    if tab not in ("approved", "rejected", "review"):
        tab = "review"
    if part not in PARTS:
        part = PARTS[0]

    # Hanya tampilkan gambar dari sesi proses saat ini
    since = state.get("process_start")

    # Tab totals
    tab_totals = count_by_tab(gender, since)

    # Sub-nav per part
    sub_parts = parts_for_tab(gender, tab, since)

    # Get images for current tab + part
    if tab == "review":
        images = get_staging_images(gender, part)
    elif tab == "approved":
        images = get_dataset_images(gender, part, "approved", since)
    else:
        images = get_dataset_images(gender, part, "reject", since)

    total = len(images)
    per_page = 12
    page = request.args.get("page", 1, type=int)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)
    page_images = images[(page - 1) * per_page: page * per_page]

    sf_summary = state.get("sf_summary")

    return render_template_string(REVIEW_TEMPLATE,
        gender=gender, active_tab=tab, active_part=part,
        sub_parts=sub_parts, tab_totals=tab_totals,
        images=page_images, total=total, page=page, total_pages=total_pages,
        sf_summary=sf_summary)


# ── Image serving ────────────────────────────────────────────────────────────

@app.route("/img/review/<part>/<filename>")
def serve_staging(part, filename):
    state = load_state()
    gender = state.get("gender", "wanita")
    d = os.path.join(BASE, "data", "pipelines", gender, f".staging_{part}")
    return send_from_directory(d, filename)


@app.route("/img/approved/<part>/<filename>")
def serve_approved(part, filename):
    state = load_state()
    gender = state.get("gender", "wanita")
    fpath = _find_dataset_file(gender, part, "approved", filename)
    if fpath:
        return send_from_directory(os.path.dirname(fpath), filename)
    return "Not found", 404


@app.route("/img/rejected/<part>/<filename>")
def serve_rejected(part, filename):
    state = load_state()
    gender = state.get("gender", "wanita")
    fpath = _find_dataset_file(gender, part, "reject", filename)
    if fpath:
        return send_from_directory(os.path.dirname(fpath), filename)
    return "Not found", 404


# ── Move helpers ─────────────────────────────────────────────────────────────

def _extract_age(filename):
    """Extract age group dari filename."""
    for age in ["25-39", "40-65"]:
        if f"_{age}" in filename:
            return age
    return "25-39"


def _extract_part_from_path(fpath):
    """Extract part name dari path di dataset."""
    for part in PARTS:
        if f"/{part}/" in fpath or fpath.endswith(f"/{part}"):
            return part
    return None


def move_staging_file(gender, part, filename, decision):
    """Move file dari staging ke batch (approved/reject)."""
    src = os.path.join(BASE, "data", "pipelines", gender, f".staging_{part}", filename)
    if not os.path.exists(src):
        return
    age_group = _extract_age(filename)
    decision_folder = "approved" if decision == "approve" else "reject"
    dst_path, batch_num = batch_move(src, gender, age_group, part, decision_folder,
                                     original_filename=filename)
    if dst_path:
        print(f"[REVIEW] {filename} → {decision_folder} (batch_{batch_num})")


def move_dataset_file(gender, part, filename, from_decision, to_decision):
    """Move file antar approved <-> reject di dataset."""
    fpath = _find_dataset_file(gender, part, from_decision, filename)
    if not fpath:
        return
    age_group = _extract_age(filename)
    dst_path, batch_num = batch_move(fpath, gender, age_group, part, to_decision,
                                     original_filename=filename)
    if dst_path:
        print(f"[MOVE] {filename}: {from_decision} → {to_decision} (batch_{batch_num})")


# ── Action routes ────────────────────────────────────────────────────────────

@app.route("/review/<tab>/<part>/action", methods=["POST"])
def review_action(tab, part):
    state = load_state()
    gender = state.get("gender", "wanita")
    filename = request.form.get("filename")
    decision = request.form.get("decision")
    page = request.form.get("page", 1, type=int)

    if filename and decision:
        if tab == "review":
            move_staging_file(gender, part, filename, decision)
        elif tab == "approved" and decision == "reject":
            move_dataset_file(gender, part, filename, "approved", "reject")
        elif tab == "rejected" and decision == "approve":
            move_dataset_file(gender, part, filename, "reject", "approved")

    return redirect(url_for("review", tab=tab, part=part, page=page))


@app.route("/review/<tab>/<part>/bulk", methods=["POST"])
def review_bulk(tab, part):
    state = load_state()
    gender = state.get("gender", "wanita")
    decision = request.form.get("decision")
    page = request.form.get("page", 1, type=int)
    if not decision:
        return redirect(url_for("review", tab=tab, part=part, page=page))

    for fname in request.form.getlist("selected"):
        if tab == "review":
            move_staging_file(gender, part, fname, decision)
        elif tab == "approved" and decision == "reject":
            move_dataset_file(gender, part, fname, "approved", "reject")
        elif tab == "rejected" and decision == "approve":
            move_dataset_file(gender, part, fname, "reject", "approved")

    return redirect(url_for("review", tab=tab, part=part, page=page))


# ═══════════════════════════════════════════════════════════════════════════════
# FINISH / ULTIMATE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/finish", methods=["POST"])
def finish():
    """Stop all serves, go home."""
    stop_all_serves()
    state = load_state()
    state["phase"] = "idle"
    state["gender"] = None
    save_state(state)
    print("[FINISH] All servers stopped, back to home")

    # Auto re-train ML jika cukup data baru
    maybe_retrain()

    return redirect(url_for("index"))


@app.route("/ultimate", methods=["POST"])
def ultimate():
    """Approve ALL remaining staging images across all parts, then finish."""
    state = load_state()
    gender = state.get("gender", "wanita")
    total = 0
    for part in PARTS:
        images = get_staging_images(gender, part)
        for fname in images:
            move_staging_file(gender, part, fname, "approve")
            total += 1
    print(f"[ULTIMATE] Approved {total} images for {gender}")

    stop_all_serves()
    state = load_state()
    state["phase"] = "idle"
    state["gender"] = None
    save_state(state)

    # Auto re-train ML jika cukup data baru
    maybe_retrain()

    return redirect(url_for("index"))


if __name__ == "__main__":
    # ── Full cleanup saat startup ─────────────────────────────────────────
    # Kill semua orphan pipeline processes dari sesi sebelumnya
    state = load_state()
    for name, ps in list(state.get("pipelines", {}).items()):
        pid = ps.get("pid")
        port = ps.get("port", 0)
        if pid and is_process_alive(pid):
            try:
                os.kill(pid, 9)
                print(f"[STARTUP CLEANUP] Killed orphan {name} pid={pid}")
            except Exception:
                pass
        if port:
            kill_port(port)

    # Reset state bersih
    state["processing"] = False
    state["phase"] = "idle"
    state["pipelines"] = {}
    save_state(state)

    # ── Signal handler — cleanup saat Ctrl+C atau kill ────────────────────
    def _signal_handler(sig, frame):
        print(f"\n[SIGNAL] Received signal {sig}, cleaning up...")
        _cleanup_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── Start ─────────────────────────────────────────────────────────────
    print("=" * 50)
    print("  Wajah AI Dashboard")
    print("  http://127.0.0.1:8000")
    print("=" * 50)

    # use_reloader=False → satu proses saja, tidak ada orphan dari restart
    # debug=True tetap aktif untuk error page yang informatif
    app.run(debug=True, use_reloader=False, port=8000)
