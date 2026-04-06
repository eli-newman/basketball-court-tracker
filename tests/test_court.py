"""Tests for court geometry and coordinate mapping."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from court import (
    COURT_LENGTH,
    COURT_WIDTH,
    COURT_LANDMARKS,
    FT_LINE_DIST,
    KEY_TOP,
    KEY_BOTTOM,
    HALF_COURT_X,
    THREE_PT_RADIUS,
    BASKET_OFFSET,
    court_to_minimap,
    get_court_point,
)


def test_court_dimensions():
    assert COURT_LENGTH == 94.0
    assert COURT_WIDTH == 50.0


def test_half_court():
    assert HALF_COURT_X == 47.0


def test_free_throw_distance():
    assert FT_LINE_DIST == 19.0


def test_key_width():
    assert KEY_BOTTOM - KEY_TOP == 16.0


def test_key_centered():
    center = (KEY_TOP + KEY_BOTTOM) / 2
    assert center == COURT_WIDTH / 2


def test_three_point_radius():
    assert THREE_PT_RADIUS == 23.75


def test_basket_positions():
    left = COURT_LANDMARKS["left_basket"]
    right = COURT_LANDMARKS["right_basket"]
    assert left == (BASKET_OFFSET, COURT_WIDTH / 2)
    assert right == (COURT_LENGTH - BASKET_OFFSET, COURT_WIDTH / 2)
    # Baskets are symmetric
    assert left[1] == right[1]
    assert left[0] + right[0] == COURT_LENGTH


def test_landmarks_all_in_bounds():
    for name, (x, y) in COURT_LANDMARKS.items():
        assert 0 <= x <= COURT_LENGTH, f"{name} x={x} out of bounds"
        assert 0 <= y <= COURT_WIDTH, f"{name} y={y} out of bounds"


def test_landmarks_symmetry():
    """Left and right landmarks should be symmetric about half court."""
    pairs = [
        ("left_key_top_right", "right_key_top_left"),
        ("left_key_bottom_right", "right_key_bottom_left"),
        ("left_ft_center", "right_ft_center"),
    ]
    for left_name, right_name in pairs:
        lx, ly = COURT_LANDMARKS[left_name]
        rx, ry = COURT_LANDMARKS[right_name]
        assert abs(lx + rx - COURT_LENGTH) < 0.01, f"{left_name}/{right_name} not symmetric in x"
        assert abs(ly - ry) < 0.01, f"{left_name}/{right_name} not symmetric in y"


def test_court_to_minimap_corners():
    w, h, pad = 940, 500, 10
    # Top-left corner
    px, py = court_to_minimap(0, 0, w, h, pad)
    assert px == pad
    assert py == pad
    # Bottom-right corner
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


def test_get_court_point_by_name():
    pt = get_court_point("top-left")
    assert pt == (0.0, 0.0)


def test_get_court_point_by_index():
    pt = get_court_point(0)  # index 0 = court_top_left
    assert pt == (0.0, 0.0)


def test_get_court_point_unknown_name():
    pt = get_court_point("nonexistent-class")
    assert pt is None


def test_get_court_point_unknown_index():
    pt = get_court_point(999)
    assert pt is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
