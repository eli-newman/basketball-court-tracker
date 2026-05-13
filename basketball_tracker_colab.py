# -*- coding: utf-8 -*-
"""basketball_tracker_colab.ipynb

# Basketball Court Tracker - Colab Edition

Thin Colab wrapper around the basketball-court-tracker repo. Clones the
repo, installs deps, and runs `main.py` so any code change in the repo
applies here automatically (no inlined drift).

## Setup
1. **Enable GPU**: Runtime > Change runtime type > T4 GPU
2. Add **`ROBOFLOW_API_KEY`** and (for private repo) **`GITHUB_TOKEN`** to
   Colab Secrets (left sidebar > Secrets), or paste them inline.
3. Run all cells in order.

## Backends
- `hosted`: Roboflow HTTP API. Works without GPU. ~0.5 fps. Good for
  validation; useless for full games.
- `local`:  runs Roboflow weights on the Colab T4 via `inference-gpu`.
  ~30+ fps. Use this for real Knicks footage.
"""

#@title 1. Install dependencies and clone the repo
#@markdown Installs the pipeline + (optional) GPU inference, then clones
#@markdown this notebook's source repo into /content/tracker.
#@markdown
#@markdown ⚠️ Default branch is `claude/pedantic-wu-bb2183` — it contains
#@markdown identity + per-player stats + made/missed shot detection that
#@markdown haven't merged to `main` yet. Flip to `main` only if you want
#@markdown the older codepath.
REPO_URL = "https://github.com/eli-newman/basketball-court-tracker.git"  #@param {type:"string"}
REPO_BRANCH = "claude/pedantic-wu-bb2183"  #@param {type:"string"}
INSTALL_GPU_INFERENCE = True  #@param {type:"boolean"}

import os, subprocess, sys

# Core deps (always)
subprocess.run(["pip", "install", "-q",
    "supervision>=0.19.0", "opencv-python-headless>=4.8.0",
    "numpy>=1.24.0", "requests>=2.28.0", "scikit-learn>=1.3.0"], check=True)

# GPU inference (optional but recommended on T4)
if INSTALL_GPU_INFERENCE:
    subprocess.run(["pip", "install", "-q",
        "--extra-index-url", "https://download.pytorch.org/whl/cu124",
        "inference-gpu"], check=True)

# yt-dlp (for the optional YouTube ingest cell)
subprocess.run(["pip", "install", "-q", "-U", "yt-dlp"], check=True)

# Clone the repo. For private repos, set GITHUB_TOKEN in Colab Secrets.
def _clone_repo(url: str, branch: str, dest: str):
    if os.path.isdir(dest):
        print(f"Repo already at {dest}, pulling latest…")
        subprocess.run(["git", "-C", dest, "fetch", "--all"], check=True)
        subprocess.run(["git", "-C", dest, "checkout", branch], check=True)
        subprocess.run(["git", "-C", dest, "pull", "origin", branch], check=True)
        return
    auth_url = url
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("GITHUB_TOKEN")
        if token and url.startswith("https://github.com/"):
            auth_url = url.replace("https://github.com/", f"https://{token}@github.com/")
    except Exception:
        pass
    subprocess.run(["git", "clone", "-b", branch, auth_url, dest], check=True)

_clone_repo(REPO_URL, REPO_BRANCH, "/content/tracker")
os.chdir("/content/tracker")
sys.path.insert(0, "/content/tracker")

# GPU sanity check
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"ONNX providers: {providers}")
    print("✅ CUDA available" if "CUDAExecutionProvider" in providers
          else "⚠️  No CUDA — local backend will fall back to CPU (slow). "
               "Make sure runtime is set to T4 GPU.")
except Exception:
    print("(onnxruntime not yet imported — fine if INSTALL_GPU_INFERENCE was False)")

print(f"\nRepo at: /content/tracker (branch: {REPO_BRANCH})")
print(subprocess.run(["git", "log", "--oneline", "-3"],
                     capture_output=True, text=True).stdout)

#@title 2. Mount Google Drive (optional)
#@markdown Mount only if your clips live in Drive or you want outputs persisted.
MOUNT_DRIVE = False  #@param {type:"boolean"}
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')

#@title 2a. Ingest: Fetch a Knicks clip from YouTube (yt-dlp)
#@markdown Paste a URL. Optionally trim to a single possession.
YOUTUBE_URL = ""  #@param {type:"string"}
LABEL = "knicks_clip"  #@param {type:"string"}
START_TIME = ""  #@param {type:"string"}
END_TIME = ""  #@param {type:"string"}
MAX_HEIGHT = 720  #@param {type:"integer"}
RUN_YT_INGEST = False  #@param {type:"boolean"}

INGESTED_PATH = None
if RUN_YT_INGEST and YOUTUBE_URL:
    os.makedirs("/content/clips", exist_ok=True)
    out_path = f"/content/clips/{LABEL}.mp4"
    section_arg = ""
    if START_TIME and END_TIME:
        section_arg = f'--download-sections "*{START_TIME}-{END_TIME}"'
    fmt = f"bestvideo[height<={MAX_HEIGHT}][ext=mp4]+bestaudio/best[height<={MAX_HEIGHT}]"
    cmd = f'yt-dlp -f "{fmt}" --merge-output-format mp4 {section_arg} -o "{out_path}" "{YOUTUBE_URL}"'
    subprocess.run(cmd, shell=True, check=True)
    print(f"Saved: {out_path}")
    INGESTED_PATH = out_path
else:
    print("Skipping YouTube ingest. Toggle RUN_YT_INGEST + paste a URL to use.")

#@title 2b. Ingest: Upload a local clip from your computer
#@markdown Use this for laptop screen-recordings or pre-cleaned clips.
RUN_LOCAL_UPLOAD = False  #@param {type:"boolean"}

if RUN_LOCAL_UPLOAD:
    from google.colab import files
    import shutil
    os.makedirs("/content/clips", exist_ok=True)
    uploaded = files.upload()
    if uploaded:
        src_name = list(uploaded.keys())[0]
        ext = os.path.splitext(src_name)[1] or ".mp4"
        dst = f"/content/clips/uploaded{ext}"
        shutil.move(src_name, dst)
        print(f"Saved: {dst}")
        INGESTED_PATH = dst

#@title 2c. Optional: Clean an ingested clip (crop chrome, drop fps, trim)
#@markdown Only needed for hand-recorded YouTube tabs. yt-dlp output is clean.
SOURCE_PATH = ""  #@param {type:"string"}
TRIM_START_SEC = 0  #@param {type:"number"}
CROP_BOTTOM_PX = 0  #@param {type:"integer"}
CROP_TOP_PX = 0  #@param {type:"integer"}
TARGET_FPS = 30  #@param {type:"integer"}
TARGET_WIDTH = 1280  #@param {type:"integer"}
RUN_CLEAN = False  #@param {type:"boolean"}

if RUN_CLEAN and SOURCE_PATH:
    os.makedirs("/content/clips", exist_ok=True)
    out_path = "/content/clips/cleaned.mp4"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", SOURCE_PATH],
        capture_output=True, text=True,
    )
    w, h = [int(x) for x in probe.stdout.strip().split(",")]
    new_h = h - CROP_TOP_PX - CROP_BOTTOM_PX
    vf = f"crop={w}:{new_h}:0:{CROP_TOP_PX},scale={TARGET_WIDTH}:-2"
    cmd = (
        f'ffmpeg -y -ss {TRIM_START_SEC} -i "{SOURCE_PATH}" '
        f'-vf "{vf}" -r {TARGET_FPS} '
        f'-c:v libx264 -preset medium -crf 23 -c:a aac -b:a 96k '
        f'-movflags +faststart "{out_path}"'
    )
    subprocess.run(cmd, shell=True, check=True)
    print(f"Cleaned: {out_path}")
    INGESTED_PATH = out_path

#@title 3. Configuration
#@markdown Pick a backend, set your API key, and choose a video.
BACKEND = "local"  #@param ["local", "hosted"]
ROBOFLOW_API_KEY = ""  #@param {type:"string"}
VIDEO_PATH = ""  #@param {type:"string"}
OUTPUT_DIR = "/content/output"  #@param {type:"string"}

PLAYER_CONFIDENCE = 0.4  #@param {type:"slider", min:0.1, max:0.9, step:0.05}
COURT_CONFIDENCE = 0.3  #@param {type:"slider", min:0.1, max:0.9, step:0.05}
FRAME_SKIP = 1  #@param {type:"integer"}
MAX_FRAMES = 0  #@param {type:"integer"}
N_TEAMS = 2  #@param {type:"slider", min:2, max:3, step:1}
DEBUG_MODE = False  #@param {type:"boolean"}

#@markdown ### Team anchors (recommended)
#@markdown Names from `team_classifier.TEAM_COLOR_PROFILES`. With both set,
#@markdown the classifier locks teams to known jersey colors instead of
#@markdown clustering on the clip — way more robust on short clips and
#@markdown Knicks-blue vs Sixers-white in particular.
TEAM_A = "knicks"  #@param {type:"string"}
TEAM_B = "sixers"  #@param {type:"string"}

#@markdown ### Jersey OCR (unlocks per-player stats + cross-cut identity)
#@markdown When ON: the pipeline OCRs jersey numbers, collapses
#@markdown (team, number) → persistent player_id across camera cuts, and
#@markdown renders a TOP SCORERS panel in the bottom-left. Costs ~20%
#@markdown extra runtime per locked-jersey-search frame.
JERSEY_OCR = True  #@param {type:"boolean"}
JERSEY_SAMPLE_EVERY = 5  #@param {type:"integer"}

#@markdown ### Minimap view
#@markdown `both`: full-court (top) + half-court (bottom) stacked next to
#@markdown the broadcast frame. `full` and `half` show only one.
VIEW = "both"  #@param ["both", "full", "half"]

# Pull API key from Colab Secrets if not set inline
if not ROBOFLOW_API_KEY:
    try:
        from google.colab import userdata  # type: ignore
        ROBOFLOW_API_KEY = userdata.get("ROBOFLOW_API_KEY") or ""
    except Exception:
        pass

# Default video path to whatever the ingest cell produced
if not VIDEO_PATH and INGESTED_PATH:
    VIDEO_PATH = INGESTED_PATH

assert ROBOFLOW_API_KEY, "Set ROBOFLOW_API_KEY (in Colab Secrets or this cell)."
assert VIDEO_PATH, "Set VIDEO_PATH (or run an ingest cell first)."
print(f"Backend: {BACKEND}")
print(f"Video:   {VIDEO_PATH}")
print(f"Output:  {OUTPUT_DIR}")

#@title 4. Run the pipeline
#@markdown Calls `main.py` from the repo so all bug fixes apply automatically.
import time
os.makedirs(OUTPUT_DIR, exist_ok=True)
cmd = [
    sys.executable, "main.py", VIDEO_PATH,
    "--roboflow-key", ROBOFLOW_API_KEY,
    "--output", OUTPUT_DIR,
    "--backend", BACKEND,
    "--player-confidence", str(PLAYER_CONFIDENCE),
    "--court-confidence", str(COURT_CONFIDENCE),
    "--frame-skip", str(FRAME_SKIP),
    "--max-frames", str(MAX_FRAMES),
    "--n-teams", str(N_TEAMS),
    "--view", VIEW,
]
if TEAM_A:
    cmd += ["--team-a", TEAM_A]
if TEAM_B:
    cmd += ["--team-b", TEAM_B]
if JERSEY_OCR:
    cmd += ["--jersey-ocr",
            "--jersey-sample-every", str(JERSEY_SAMPLE_EVERY)]
if DEBUG_MODE:
    cmd.append("--debug")

t0 = time.time()
subprocess.run(cmd, check=True)
print(f"\nWall time: {time.time() - t0:.1f}s")

#@title 5. Summary: scoreboard + top scorers + event count
#@markdown Reads the JSON sidecars `main.py` wrote into OUTPUT_DIR and
#@markdown prints a compact stats summary. Useful before sitting through
#@markdown the full composite video.
import json
from IPython.display import display, Markdown

def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

sb = _read_json(os.path.join(OUTPUT_DIR, "output_scoreboard.json")) or {}
ps = _read_json(os.path.join(OUTPUT_DIR, "output_player_stats.json")) or {}
ev = _read_json(os.path.join(OUTPUT_DIR, "output_events.json")) or []

# Scoreboard line
score_by_team = sb.get("score_by_team", {})
team_labels = sb.get("team_labels", {})
if score_by_team:
    parts = []
    for tid, pts in sorted(score_by_team.items(), key=lambda kv: int(kv[0])):
        label = team_labels.get(str(tid)) or team_labels.get(tid) or f"T{tid}"
        parts.append(f"**{label}** {pts}")
    display(Markdown("### Final score\n" + "  •  ".join(parts)))

# Shot events
n_made = sum(1 for e in ev if e.get("made"))
display(Markdown(
    f"### Shot events\n"
    f"`{len(ev)}` attempts · `{n_made}` made · `{len(ev) - n_made}` missed"
))

# Top scorers table — only populated when --jersey-ocr identified shooters
players = ps.get("players", [])
if players:
    rows = sorted(players, key=lambda p: -p["points"])[:8]
    lines = ["### Top scorers", "",
             "| Team | # | PTS | FGM/FGA | FG% | 3PM/3PA |",
             "|------|---|-----|---------|-----|---------|"]
    for p in rows:
        lines.append(
            f"| {p.get('team') or 'T?'} | "
            f"#{p.get('jersey_number') or '?'} | "
            f"{p['points']} | "
            f"{p['fgm']}/{p['fga']} | "
            f"{int(p['fg_pct'] * 100)}% | "
            f"{p['fg3m']}/{p['fg3a']} |"
        )
    display(Markdown("\n".join(lines)))
else:
    display(Markdown(
        "### Top scorers\n"
        "_No identified shooters yet — either jersey OCR was off, or no "
        "jersey number locked before a shot resolved. Try enabling "
        "`--jersey-ocr` and re-running on a longer segment._"
    ))

#@title 6. Preview the composite
#@markdown Shows the side-by-side (broadcast + minimap) inline. For long
#@markdown clips this base64-embeds a big chunk — fine on Colab Pro, can
#@markdown OOM the browser tab on free tier. Use the Drive-sync cell below
#@markdown instead if it chokes.
from IPython.display import HTML
import base64
video_path = os.path.join(OUTPUT_DIR, "output_composite.mp4")
mp4 = open(video_path, "rb").read()
data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
HTML(f'<video controls width=900 src="{data_url}"></video>')

#@title 7. (Optional) Copy outputs to Google Drive so they survive runtime shutdown
#@markdown Colab VMs are ephemeral — when the runtime disconnects, /content
#@markdown is wiped. Run this to persist the run. Requires the Drive mount
#@markdown cell (2) to have run.
DRIVE_DEST = "/content/drive/MyDrive/basketball_output"  #@param {type:"string"}
RUN_SYNC_TO_DRIVE = False  #@param {type:"boolean"}

if RUN_SYNC_TO_DRIVE:
    import shutil, datetime
    assert os.path.ismount("/content/drive") or os.path.isdir("/content/drive/MyDrive"), \
        "Drive isn't mounted — run cell 2 first."
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(DRIVE_DEST, f"run_{stamp}")
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(OUTPUT_DIR):
        src = os.path.join(OUTPUT_DIR, name)
        dst = os.path.join(target, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"Synced {OUTPUT_DIR} → {target}")
else:
    print("Skipping Drive sync. Flip RUN_SYNC_TO_DRIVE to save outputs.")
