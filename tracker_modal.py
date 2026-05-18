"""Run the basketball court tracker on Modal (serverless GPU).

This is the harness that replaces the Colab + Cloudflare Tunnel +
SSH-from-Bash dance with a single `modal run` from any terminal:

    modal run tracker_modal.py \\
        --video-path ~/Desktop/knicks_test_01_clean.mp4 \\
        --roboflow-key $ROBOFLOW_API_KEY \\
        --team-a knicks --team-b sixers

Outputs land in `./output_modal/` (composite mp4 + JSON files), pulled
back over the wire from the remote container.

Why Modal:
  - No expiring tunnel hostnames.
  - $30/mo free credit ≈ ~375 A100-min/mo or ~750 A10-min/mo.
  - Cold-start ~30s, warm-pool reusable.
  - Container is reproducible — same image every run, no Colab drift.

Pivot note: this first version still uses the Roboflow `inference` package,
so it will continue to bill credits per frame. Once that's exhausted we
swap the detector backends to Ultralytics (player+court) and PaddleOCR
(jersey numbers) — both run inside this same Modal image, no harness
changes needed.
"""

from pathlib import Path
import json
import os

import modal


# ── Image ──────────────────────────────────────────────────────────────────
# Python 3.11 because Roboflow's `inference` package doesn't support 3.13
# (and 3.12 is sometimes flaky with their pinned transformers version).
# Once we pivot off the `inference` package this can move to 3.12.
#
# CUDA-DEVEL base (not just runtime) because `inference-gpu` transitively
# pulls `pycuda`, which compiles native code against `cuda.h` at install
# time. `debian_slim` has no CUDA headers → build fails. `nvidia/cuda:*-devel-*`
# ships the full toolkit so the wheel builds cleanly.
PROJECT_DIR = Path(__file__).parent

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "ffmpeg",          # video read/write
        "libgl1",          # opencv runtime dep
        "libglib2.0-0",    # opencv runtime dep
        "build-essential", # g++/gcc/make for pycuda's native build
        "clang",           # pycuda's setup.py shells out to clang++ by name
    )
    # Force pycuda to use g++ (some CUDA base images mis-detect compiler).
    .env({"CXX": "g++", "CC": "gcc"})
    .pip_install(
        # Project deps (mirrors requirements.txt)
        "supervision>=0.19.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "requests>=2.28.0",
        "scikit-learn>=1.3.0",
        # Roboflow inference. GPU variant pulls onnxruntime-gpu + pycuda;
        # the latter compiles against cuda.h shipped in the -devel- base.
        "inference-gpu",
    )
    # Ship the project source. `ignore` keeps the image small and avoids
    # uploading output dirs / virtualenvs that don't need to be in the
    # container.
    .add_local_dir(
        str(PROJECT_DIR),
        remote_path="/app",
        ignore=[
            ".venv*",
            ".claude",
            "output*",
            "*.mp4",
            "*.pyc",
            "__pycache__",
            ".git",
        ],
    )
)


app = modal.App("basketball-tracker", image=image)


# Modal Volume persists across function invocations — used to cache
# Roboflow's downloaded weights between runs. First run: ~30s of cold
# weight download. Subsequent runs: instant cache hit.
weights_cache = modal.Volume.from_name(
    "basketball-tracker-weights", create_if_missing=True,
)


@app.function(
    gpu="A10",              # $1.10/hr — about 4x faster than CPU, $0.04/run
    timeout=900,            # 15 min — clip up to ~10x our normal length
    volumes={"/cache": weights_cache},
    # Cache Roboflow weights here so `inference.get_model()` doesn't
    # re-download them every cold start.
)
def process_video(
    video_bytes: bytes,
    roboflow_api_key: str,
    team_a: str | None = None,
    team_b: str | None = None,
    enable_jersey_ocr: bool = False,
    view: str = "both",
    frame_skip: int = 1,
    max_frames: int = 0,
) -> dict:
    """Run the full pipeline on a video; return all output files as bytes.

    Returns a dict keyed by output filename. Caller writes them to disk.
    """
    import os
    import sys
    import tempfile

    # Run inference's weight cache in the Modal Volume (persisted across
    # cold starts so we only pay the download once).
    os.environ.setdefault("MODEL_CACHE_DIR", "/cache/inference")
    os.environ.setdefault("DISABLE_VERSION_CHECK", "true")

    # Make project code importable.
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from config import Config  # noqa: E402
    from pipeline import Pipeline  # noqa: E402

    # Stage the input video on local-tmp inside the container.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name
    output_dir = tempfile.mkdtemp(prefix="tracker_out_")

    config = Config(
        video_path=video_path,
        output_dir=output_dir,
        roboflow_api_key=roboflow_api_key,
        inference_backend="local",   # uses inference-gpu; reads /cache
        team_a=team_a,
        team_b=team_b,
        enable_jersey_ocr=enable_jersey_ocr,
        view=view,
        frame_skip=frame_skip,
        max_frames=max_frames,
    )
    Pipeline(config).run()

    # Collect every output file, return as {filename: bytes}.
    # main.py writes: output_composite.mp4, output_coordinates.{json,csv},
    # output_events.json, output_scoreboard.json, output_player_stats.json
    outputs: dict[str, bytes] = {}
    for path in sorted(Path(output_dir).iterdir()):
        if path.is_file():
            outputs[path.name] = path.read_bytes()

    return outputs


@app.local_entrypoint()
def main(
    video_path: str,
    roboflow_key: str,
    team_a: str | None = None,
    team_b: str | None = None,
    jersey_ocr: bool = False,
    view: str = "both",
    frame_skip: int = 1,
    max_frames: int = 0,
    output_dir: str = "./output_modal",
):
    """Local CLI: send a clip to Modal, write the results back locally."""
    src = Path(video_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"video not found: {src}")

    print(f"[modal] sending {src.name} ({src.stat().st_size / 1e6:.1f} MB) ...")
    video_bytes = src.read_bytes()

    outputs = process_video.remote(
        video_bytes=video_bytes,
        roboflow_api_key=roboflow_key,
        team_a=team_a,
        team_b=team_b,
        enable_jersey_ocr=jersey_ocr,
        view=view,
        frame_skip=frame_skip,
        max_frames=max_frames,
    )

    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (out_root / name).write_bytes(data)
        print(f"[modal] wrote {out_root / name}  ({len(data) / 1e6:.2f} MB)")

    # Print a tiny summary of the JSON outputs so the terminal user
    # sees the headline numbers without having to open files.
    score_path = out_root / "output_scoreboard.json"
    if score_path.exists():
        sb = json.loads(score_path.read_text())
        labels = sb.get("team_labels", {})
        scores = sb.get("score_by_team", {})
        print(f"[modal] final score: "
              f"{labels.get('0', 'A')} {scores.get('0', 0)} - "
              f"{labels.get('1', 'B')} {scores.get('1', 0)}")
    events_path = out_root / "output_events.json"
    if events_path.exists():
        events = json.loads(events_path.read_text())
        made = sum(1 for e in events if e.get("made"))
        print(f"[modal] {len(events)} shot events ({made} made)")
