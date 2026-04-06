"""Tests for court geometry and coordinate mapping."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from court import (
    COURT_LENGTH,
    COURT_WIDTH,
    HALF_COURT_X,
    KEYPOINT_LABEL_MAP,
    KEYPOINT_INDEX_MAP,
    court_to_minimap,
    get_court_point,
    get_all_vertices_ft,
)


def test_court_dimensions():
    assert COURT_LENGTH == 94.0
    assert COURT_WIDTH == 50.0


def test_half_court():
    assert HALF_COURT_X == 47.0


def test_keypoint_count():
    """Model has 33 keypoints."""
    assert len(KEYPOINT_LABEL_MAP) == 33
    assert len(KEYPOINT_INDEX_MAP) == 33


def test_all_keypoints_in_bounds():
    for label, (x, y) in KEYPOINT_LABEL_MAP.items():
        assert 0 <= x <= COURT_LENGTH + 0.5, f"Label {label}: x={x:.2f} out of bounds"
        assert 0 <= y <= COURT_WIDTH + 0.5, f"Label {label}: y={y:.2f} out of bounds"


def test_corner_keypoints():
    """Labels 01 and 41 should be opposite corners."""
    tl = get_court_point("01")  # top-left (0, 0)
    br = get_court_point("41")  # bottom-right (94, 50)
    assert tl is not None and br is not None
    assert abs(tl[0]) < 0.5  # near 0
    assert abs(tl[1]) < 0.5  # near 0
    assert abs(br[0] - COURT_LENGTH) < 0.5  # near 94
    assert abs(br[1] - COURT_WIDTH) < 0.5   # near 50


def test_half_court_center():
    """Label 21 is half court center."""
    pt = get_court_point("21")
    assert pt is not None
    assert abs(pt[0] - HALF_COURT_X) < 1.0
    assert abs(pt[1] - COURT_WIDTH / 2) < 1.0


def test_basket_positions():
    """Labels 09 (left basket) and 33 (right basket) should be symmetric."""
    left = get_court_point("09")
    right = get_court_point("33")
    assert left is not None and right is not None
    # Both centered on court width
    assert abs(left[1] - COURT_WIDTH / 2) < 1.0
    assert abs(right[1] - COURT_WIDTH / 2) < 1.0
    # Roughly symmetric in x
    assert abs(left[0] + right[0] - COURT_LENGTH) < 2.0


def test_left_right_symmetry():
    """Left and right side keypoints should be symmetric about half court."""
    symmetric_pairs = [
        ("01", "34"),  # top corners
        ("08", "41"),  # bottom corners
        ("04", "37"),  # baseline paint top
        ("05", "38"),  # baseline paint bottom
        ("12", "28"),  # free throw top
        ("14", "30"),  # free throw bottom
        ("10", "31"),  # 3pt straight top
        ("11", "32"),  # 3pt straight bottom
    ]
    for left_label, right_label in symmetric_pairs:
        lx, ly = get_court_point(left_label)
        rx, ry = get_court_point(right_label)
        assert abs(lx + rx - COURT_LENGTH) < 1.0, f"{left_label}/{right_label} not symmetric in x"
        assert abs(ly - ry) < 1.0, f"{left_label}/{right_label} not symmetric in y"


def test_get_court_point_by_label():
    pt = get_court_point("01")
    assert pt is not None
    assert abs(pt[0]) < 0.5
    assert abs(pt[1]) < 0.5


def test_get_court_point_by_index():
    pt = get_court_point(0)  # index 0 = label "01" = top-left corner
    assert pt is not None
    assert abs(pt[0]) < 0.5
    assert abs(pt[1]) < 0.5


def test_get_court_point_unknown():
    assert get_court_point("99") is None
    assert get_court_point(999) is None


def test_court_to_minimap_corners():
    w, h, pad = 940, 500, 10
    px, py = court_to_minimap(0, 0, w, h, pad)
    assert px == pad
    assert py == pad
    px, py = court_to_minimap(COURT_LENGTH, COURT_WIDTH, w, h, pad)
    assert px == w - pad
    assert py == h - pad


def test_court_to_minimap_center():
    w, h, pad = 940, 500, 10
    px, py = court_to_minimap(HALF_COURT_X, COURT_WIDTH / 2, w, h, pad)
    expected_px = pad + (w - 2 * pad) // 2
    expected_py = pad + (h - 2 * pad) // 2
    assert abs(px - expected_px) <= 1
    assert abs(py - expected_py) <= 1


def test_get_all_vertices():
    verts = get_all_vertices_ft()
    assert len(verts) == 33
    # First vertex should be near (0, 0)
    assert abs(verts[0][0]) < 0.5
    assert abs(verts[0][1]) < 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
