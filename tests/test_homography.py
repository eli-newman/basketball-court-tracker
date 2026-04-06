"""Tests for homography engine with synthetic point correspondences."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from court import COURT_LENGTH, COURT_WIDTH, get_court_point, KEYPOINT_LABEL_MAP
from homography import HomographyEngine, CourtKeypoint


# Use real keypoint labels from the Roboflow model
_TEST_LABELS = ["01", "08", "34", "41", "19", "23", "21", "09"]
_TEST_COURT_POINTS = [get_court_point(label) for label in _TEST_LABELS]


def _simple_homography() -> np.ndarray:
    """Create a simple known homography (scaled court → pixel space)."""
    # Simple mapping: court feet → pixels at 10px/ft, offset by (50, 50)
    src = np.array([
        [0, 0], [COURT_LENGTH, 0],
        [0, COURT_WIDTH], [COURT_LENGTH, COURT_WIDTH],
    ], dtype=np.float64)
    dst = np.array([
        [50, 50], [50 + COURT_LENGTH * 10, 50],
        [50, 50 + COURT_WIDTH * 10], [50 + COURT_LENGTH * 10, 50 + COURT_WIDTH * 10],
    ], dtype=np.float64)
    # H maps pixels → court, so findHomography(dst, src)
    H, _ = cv2.findHomography(dst, src)
    return H


def _make_synthetic_keypoints(H_true: np.ndarray, labels: list[str]) -> list[CourtKeypoint]:
    """Generate synthetic keypoints by inverse-projecting court points to pixels."""
    H_inv = np.linalg.inv(H_true)
    keypoints = []
    for label in labels:
        court_pt = get_court_point(label)
        if court_pt is None:
            continue
        pt = np.array([[[court_pt[0], court_pt[1]]]], dtype=np.float64)
        pixel = cv2.perspectiveTransform(pt, H_inv)
        px, py = float(pixel[0, 0, 0]), float(pixel[0, 0, 1])
        keypoints.append(CourtKeypoint(name=label, pixel_x=px, pixel_y=py, confidence=0.9))
    return keypoints


def test_compute_with_good_keypoints():
    """Homography should be computed from 6+ well-distributed keypoints."""
    engine = HomographyEngine()
    H_true = _simple_homography()
    keypoints = _make_synthetic_keypoints(H_true, _TEST_LABELS)

    H = engine.compute(keypoints)
    assert H is not None

    # Transform a known pixel point and check accuracy
    result = engine.transform_point(H, 50, 50)
    assert result is not None
    x, y = result
    assert abs(x - 0.0) < 0.5, f"Expected x≈0, got {x}"
    assert abs(y - 0.0) < 0.5, f"Expected y≈0, got {y}"


def test_compute_with_noisy_keypoints():
    """Should handle moderate noise in keypoint positions."""
    engine = HomographyEngine()
    H_true = _simple_homography()
    keypoints = _make_synthetic_keypoints(H_true, _TEST_LABELS)

    np.random.seed(42)
    for kp in keypoints:
        kp.pixel_x += np.random.uniform(-3, 3)
        kp.pixel_y += np.random.uniform(-3, 3)

    H = engine.compute(keypoints)
    assert H is not None

    center_px = 50 + COURT_LENGTH / 2 * 10
    center_py = 50 + COURT_WIDTH / 2 * 10
    result = engine.transform_point(H, center_px, center_py)
    assert result is not None
    x, y = result
    assert abs(x - 47.0) < 2.0, f"Expected x≈47, got {x}"
    assert abs(y - 25.0) < 2.0, f"Expected y≈25, got {y}"


def test_compute_insufficient_keypoints():
    """Should return None when fewer than 4 keypoints (and no cache)."""
    engine = HomographyEngine(min_keypoints=4)

    keypoints = [
        CourtKeypoint("01", 100, 100, 0.9),
        CourtKeypoint("08", 500, 100, 0.9),
    ]

    H = engine.compute(keypoints)
    assert H is None


def test_fallback_to_cached_homography():
    """Should reuse last good homography when keypoints are insufficient."""
    engine = HomographyEngine(fallback_frames=5)
    H_true = _simple_homography()
    labels = ["01", "08", "34", "41", "19", "23"]
    keypoints = _make_synthetic_keypoints(H_true, labels)

    H1 = engine.compute(keypoints)
    assert H1 is not None

    H2 = engine.compute([keypoints[0]])
    assert H2 is not None
    assert np.allclose(H1, H2, atol=1e-6)


def test_fallback_expires():
    """Fallback should expire after fallback_frames."""
    engine = HomographyEngine(fallback_frames=2)
    H_true = _simple_homography()
    labels = ["01", "08", "34", "41", "19", "23"]
    keypoints = _make_synthetic_keypoints(H_true, labels)

    engine.compute(keypoints)

    bad = [keypoints[0]]
    assert engine.compute(bad) is not None  # frame 1
    assert engine.compute(bad) is not None  # frame 2
    assert engine.compute(bad) is None      # frame 3 — expired


def test_transform_point_out_of_bounds():
    """Points mapping outside court bounds should return None."""
    engine = HomographyEngine(court_margin=5.0)
    H_true = _simple_homography()
    result = engine.transform_point(H_true, -10000, -10000)
    assert result is None


def test_transform_point_clamping():
    """Points slightly outside court should be clamped to bounds."""
    engine = HomographyEngine(court_margin=5.0)
    H_true = _simple_homography()
    result = engine.transform_point(H_true, 45, 45)
    assert result is not None
    x, y = result
    assert x == 0.0
    assert y == 0.0


def test_reset_clears_cache():
    """Reset should clear the cached homography."""
    engine = HomographyEngine()
    H_true = _simple_homography()
    labels = ["01", "08", "34", "41", "19", "23"]
    keypoints = _make_synthetic_keypoints(H_true, labels)

    engine.compute(keypoints)
    assert engine.has_valid_homography

    engine.reset()
    assert not engine.has_valid_homography

    H = engine.compute([keypoints[0]])
    assert H is None


def test_round_trip_accuracy():
    """Court positions should round-trip through homography accurately."""
    engine = HomographyEngine()
    H_true = _simple_homography()
    keypoints = _make_synthetic_keypoints(H_true, _TEST_LABELS)
    H = engine.compute(keypoints)
    assert H is not None

    test_positions = [
        (0, 0, 50, 50),
        (47, 25, 50 + 470, 50 + 250),
        (94, 50, 50 + 940, 50 + 500),
    ]
    for cx, cy, px, py in test_positions:
        result = engine.transform_point(H, px, py)
        assert result is not None, f"Point ({px}, {py}) returned None"
        x, y = result
        assert abs(x - cx) < 1.0, f"For court ({cx},{cy}): got x={x}"
        assert abs(y - cy) < 1.0, f"For court ({cx},{cy}): got y={y}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
