"""Tests for homography engine with synthetic point correspondences."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from court import COURT_LENGTH, COURT_WIDTH
from homography import HomographyEngine, CourtKeypoint


def _make_synthetic_keypoints(H_true: np.ndarray, court_points: list) -> list:
    """Generate synthetic keypoints by inverse-projecting court points to pixels."""
    H_inv = np.linalg.inv(H_true)
    keypoints = []
    names = [
        "top-left", "top-right", "bottom-left", "bottom-right",
        "half-top", "half-bottom", "left-ft-center", "right-ft-center",
    ]
    for i, (cx, cy) in enumerate(court_points):
        pt = np.array([[[cx, cy]]], dtype=np.float64)
        pixel = cv2.perspectiveTransform(pt, H_inv)
        px, py = float(pixel[0, 0, 0]), float(pixel[0, 0, 1])
        name = names[i] if i < len(names) else f"point-{i}"
        keypoints.append(CourtKeypoint(name=name, pixel_x=px, pixel_y=py, confidence=0.9))
    return keypoints


import cv2


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
    # H maps pixels → court, so we need findHomography(dst, src)
    H, _ = cv2.findHomography(dst, src)
    return H


def test_compute_with_good_keypoints():
    """Homography should be computed from 6+ well-distributed keypoints."""
    engine = HomographyEngine()

    # Create a known pixel → court mapping
    H_true = _simple_homography()

    court_points = [
        (0, 0), (COURT_LENGTH, 0), (0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH),
        (47, 0), (47, COURT_WIDTH), (19, 25), (75, 25),
    ]
    keypoints = _make_synthetic_keypoints(H_true, court_points)

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

    court_points = [
        (0, 0), (COURT_LENGTH, 0), (0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH),
        (47, 0), (47, COURT_WIDTH), (19, 25), (75, 25),
    ]
    keypoints = _make_synthetic_keypoints(H_true, court_points)

    # Add noise to pixel positions (up to 3 pixels)
    np.random.seed(42)
    for kp in keypoints:
        kp.pixel_x += np.random.uniform(-3, 3)
        kp.pixel_y += np.random.uniform(-3, 3)

    H = engine.compute(keypoints)
    assert H is not None

    # Should still be reasonably accurate
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
        CourtKeypoint("top-left", 100, 100, 0.9),
        CourtKeypoint("top-right", 500, 100, 0.9),
    ]

    H = engine.compute(keypoints)
    assert H is None


def test_fallback_to_cached_homography():
    """Should reuse last good homography when keypoints are insufficient."""
    engine = HomographyEngine(fallback_frames=5)
    H_true = _simple_homography()

    court_points = [
        (0, 0), (COURT_LENGTH, 0), (0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH),
        (47, 0), (47, COURT_WIDTH),
    ]
    keypoints = _make_synthetic_keypoints(H_true, court_points)

    # First call — should compute and cache
    H1 = engine.compute(keypoints)
    assert H1 is not None

    # Second call with insufficient keypoints — should fallback
    H2 = engine.compute([keypoints[0]])
    assert H2 is not None
    assert np.allclose(H1, H2, atol=1e-6)


def test_fallback_expires():
    """Fallback should expire after fallback_frames."""
    engine = HomographyEngine(fallback_frames=2)
    H_true = _simple_homography()

    court_points = [
        (0, 0), (COURT_LENGTH, 0), (0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH),
        (47, 0), (47, COURT_WIDTH),
    ]
    keypoints = _make_synthetic_keypoints(H_true, court_points)

    engine.compute(keypoints)

    bad = [keypoints[0]]
    assert engine.compute(bad) is not None  # frame 1 of fallback
    assert engine.compute(bad) is not None  # frame 2 of fallback
    assert engine.compute(bad) is None      # frame 3 — expired


def test_transform_point_out_of_bounds():
    """Points mapping outside court bounds should return None."""
    engine = HomographyEngine(court_margin=5.0)
    H_true = _simple_homography()

    # A pixel way outside the court area
    result = engine.transform_point(H_true, -10000, -10000)
    assert result is None


def test_transform_point_clamping():
    """Points slightly outside court should be clamped to bounds."""
    engine = HomographyEngine(court_margin=5.0)
    H_true = _simple_homography()

    # Point slightly past the court edge (within margin)
    # Court top-left is at pixel (50, 50), so pixel (45, 45) maps to about (-0.5, -0.5)
    result = engine.transform_point(H_true, 45, 45)
    assert result is not None
    x, y = result
    assert x == 0.0  # clamped
    assert y == 0.0  # clamped


def test_reset_clears_cache():
    """Reset should clear the cached homography."""
    engine = HomographyEngine()
    H_true = _simple_homography()

    court_points = [
        (0, 0), (COURT_LENGTH, 0), (0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH),
        (47, 0), (47, COURT_WIDTH),
    ]
    keypoints = _make_synthetic_keypoints(H_true, court_points)

    engine.compute(keypoints)
    assert engine.has_valid_homography

    engine.reset()
    assert not engine.has_valid_homography

    # Should not fallback after reset
    H = engine.compute([keypoints[0]])
    assert H is None


def test_round_trip_accuracy():
    """Court center should round-trip through homography accurately."""
    engine = HomographyEngine()
    H_true = _simple_homography()

    court_points = [
        (0, 0), (COURT_LENGTH, 0), (0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH),
        (47, 0), (47, COURT_WIDTH), (19, 25), (75, 25),
    ]
    keypoints = _make_synthetic_keypoints(H_true, court_points)
    H = engine.compute(keypoints)
    assert H is not None

    # Test several known court positions
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
