"""Tests for CameraCutDetector."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cut_detector import CameraCutDetector


def _solid_frame(h: int, w: int, bgr: tuple) -> np.ndarray:
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def _gradient_frame(h: int, w: int, bgr: tuple, noise: int = 0) -> np.ndarray:
    """A solid-color frame with optional uniform noise.

    Useful when we want frames that are visually similar (low histogram
    distance) but not bit-identical, so we can simulate "continuous shot
    with some motion" frames.
    """
    f = np.full((h, w, 3), bgr, dtype=np.uint8)
    if noise > 0:
        rng = np.random.default_rng(0)
        # Same noise pattern every call so consecutive frames look the
        # same to the histogram even with noise.
        f = (f.astype(np.int16) + rng.integers(-noise, noise + 1, f.shape)).clip(0, 255).astype(np.uint8)
    return f


# ── First-frame behavior ────────────────────────────────────────────────────


def test_first_frame_never_a_cut():
    """There's nothing to compare against — the first frame can't be a cut."""
    det = CameraCutDetector()
    out = det.update(_solid_frame(200, 320, (40, 30, 30)))
    assert out is False


def test_two_identical_frames_not_a_cut():
    """Distance = 0, well below any reasonable threshold."""
    det = CameraCutDetector()
    f = _solid_frame(200, 320, (60, 80, 100))
    det.update(f)
    out = det.update(f)
    assert out is False
    assert det.last_distance == 0.0 or det.last_distance is not None


# ── Hard cut detection ──────────────────────────────────────────────────────


def test_drastic_scene_change_is_a_cut():
    """A frame full of warm tones followed by one full of cool tones is a cut."""
    det = CameraCutDetector(cut_threshold=0.45, refractory_frames=0)
    # Warm orange-ish frame
    det.update(_solid_frame(200, 320, (10, 80, 220)))
    # Cool blue-ish frame
    out = det.update(_solid_frame(200, 320, (220, 80, 10)))
    assert out is True
    assert det.last_distance is not None and det.last_distance > 0.45


def test_subtle_drift_below_threshold_not_a_cut():
    """Two similar-but-not-identical frames — well below the cut threshold."""
    det = CameraCutDetector(cut_threshold=0.45)
    f1 = _gradient_frame(200, 320, (60, 80, 100), noise=10)
    f2 = _gradient_frame(200, 320, (62, 82, 102), noise=10)
    det.update(f1)
    out = det.update(f2)
    assert out is False


# ── Refractory window ───────────────────────────────────────────────────────


def test_refractory_prevents_double_fire():
    """Two cut-worthy frames in a row should only fire once."""
    det = CameraCutDetector(cut_threshold=0.45, refractory_frames=5)
    warm = _solid_frame(200, 320, (10, 80, 220))
    cool = _solid_frame(200, 320, (220, 80, 10))
    green = _solid_frame(200, 320, (10, 220, 10))

    det.update(warm)
    assert det.update(cool) is True   # cut!
    # Immediately big change again — should be suppressed by refractory.
    assert det.update(green) is False
    assert det.update(warm) is False


def test_refractory_allows_cut_after_window():
    det = CameraCutDetector(cut_threshold=0.45, refractory_frames=2)
    warm = _solid_frame(200, 320, (10, 80, 220))
    cool = _solid_frame(200, 320, (220, 80, 10))

    det.update(warm)
    assert det.update(cool) is True
    # Stable cool for the refractory window
    assert det.update(cool) is False
    assert det.update(cool) is False
    # Now back to warm — distance large again, refractory passed
    assert det.update(warm) is True


# ── Reset ───────────────────────────────────────────────────────────────────


def test_reset_clears_previous_frame_so_next_call_is_first_frame():
    det = CameraCutDetector()
    det.update(_solid_frame(200, 320, (60, 80, 100)))
    det.reset()
    # After reset, the next frame is the "first frame" — no compare possible.
    out = det.update(_solid_frame(200, 320, (220, 80, 10)))
    assert out is False


# ── Robustness to single-frame flashes ─────────────────────────────────────

def test_single_frame_flash_does_not_trigger_cut():
    """A single anomalous frame embedded in a steady scene must not fire.

    Real-world case: replay flash / transition graphic. Previously the
    cut detector compared to the immediately-previous frame, so the
    NEXT real frame after the flash looked enormously different and
    triggered a false cut → tracker reset → team classifications got
    scrambled. The rolling-median baseline should absorb the 1-frame
    outlier.
    """
    det = CameraCutDetector(refractory_frames=1)
    scene = (60, 80, 90)
    flash = (240, 240, 240)  # near-white anomaly

    # Build a stable baseline window first.
    for _ in range(5):
        det.update(_solid_frame(200, 320, scene))

    # The anomalous frame itself can fire (it IS very different from
    # the median), but the recovery back to scene should NOT.
    det.update(_solid_frame(200, 320, flash))
    # 3 consecutive normal frames — none of them should fire even
    # though they sit right after a wildly-different flash frame.
    next1 = det.update(_solid_frame(200, 320, scene))
    next2 = det.update(_solid_frame(200, 320, scene))
    next3 = det.update(_solid_frame(200, 320, scene))
    assert next1 is False
    assert next2 is False
    assert next3 is False


def test_sustained_scene_change_still_fires():
    """A real cut (sustained change, not a 1-frame outlier) still fires.

    Guards against the rolling-median fix going too far and suppressing
    legitimate cuts.
    """
    det = CameraCutDetector(refractory_frames=1)
    scene_a = (60, 80, 90)
    scene_b = (10, 30, 220)  # very different palette

    for _ in range(5):
        det.update(_solid_frame(200, 320, scene_a))

    # Sustained switch to scene B should fire within a couple frames —
    # by the 2nd frame of scene B, the median is still mostly scene A
    # so distance is large.
    fires = [det.update(_solid_frame(200, 320, scene_b)) for _ in range(3)]
    assert any(fires), f"expected a cut in {fires}"
