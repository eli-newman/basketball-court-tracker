"""Tests for jersey vote aggregation + response parsing."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from detector import NumberDetection
from jersey import (
    JerseyRead,
    JerseyVoter,
    crop_chest,
    crop_number_bbox,
    match_number_to_player,
    parse_jersey_response,
    preprocess_for_ocr,
)


# ── JerseyVoter ─────────────────────────────────────────────────────────────

def _read(track_id: int, number: str, conf: float = 0.9) -> JerseyRead:
    return JerseyRead(track_id=track_id, number=number, confidence=conf)


def test_locks_after_three_matching_reads():
    v = JerseyVoter(confirm_at=3)
    assert v.submit(_read(1, "11")).locked is False
    assert v.submit(_read(1, "11")).locked is False
    a = v.submit(_read(1, "11"))
    assert a.locked is True
    assert a.number == "11"
    assert a.samples == 3


def test_low_confidence_reads_ignored():
    v = JerseyVoter(confirm_at=2, min_confidence=0.5)
    v.submit(_read(1, "11", conf=0.1))
    v.submit(_read(1, "11", conf=0.2))
    # Two reads at 0.1/0.2 should be discarded; not locked.
    a = v.current(1)
    assert a is None


def test_majority_wins_locks_correctly():
    v = JerseyVoter(confirm_at=3)
    v.submit(_read(1, "11"))
    v.submit(_read(1, "11"))
    v.submit(_read(1, "8"))   # one bad read
    v.submit(_read(1, "11"))  # 11 reaches 3 votes
    a = v.current(1)
    assert a.locked is True
    assert a.number == "11"
    assert a.samples == 3


def test_tied_top_does_not_lock():
    """Two numbers tied at threshold-1 → can't lock until one pulls ahead."""
    v = JerseyVoter(confirm_at=3)
    # 2 votes each — both below threshold
    v.submit(_read(1, "11"))
    v.submit(_read(1, "11"))
    v.submit(_read(1, "8"))
    v.submit(_read(1, "8"))
    assert v.current(1).locked is False
    # Now 11 crosses to 3 while 8 stays at 2 — locks 11
    v.submit(_read(1, "11"))
    a = v.current(1)
    assert a.locked is True
    assert a.number == "11"


def test_lock_persists_against_later_contradictions():
    """Once locked, the assignment stays — later reads don't unlock it."""
    v = JerseyVoter(confirm_at=3)
    v.submit(_read(1, "11"))
    v.submit(_read(1, "11"))
    v.submit(_read(1, "11"))  # locks at 11
    assert v.is_locked(1)
    # Hammer with "8" reads — should not flip
    for _ in range(5):
        v.submit(_read(1, "8"))
    assert v.current(1).number == "11"
    assert v.is_locked(1)


def test_separate_tracks_dont_interfere():
    v = JerseyVoter(confirm_at=2)
    v.submit(_read(1, "11"))
    v.submit(_read(1, "11"))
    v.submit(_read(2, "9"))
    assert v.current(1).locked is True
    assert v.current(1).number == "11"
    assert v.current(2).locked is False  # only 1 vote


def test_reset_track_clears_state():
    v = JerseyVoter(confirm_at=2)
    v.submit(_read(1, "11"))
    v.submit(_read(1, "11"))
    assert v.is_locked(1)
    v.reset_track(1)
    assert v.current(1) is None
    assert not v.is_locked(1)


def test_assignments_dict_has_all_tracks():
    v = JerseyVoter(confirm_at=2)
    v.submit(_read(1, "11"))
    v.submit(_read(1, "11"))
    v.submit(_read(2, "9"))
    out = v.assignments()
    assert 1 in out and 2 in out
    assert out[1].locked
    assert not out[2].locked


# ── parse_jersey_response ───────────────────────────────────────────────────

def test_parse_classification_style():
    resp = {"predictions": [{"class": "11", "confidence": 0.92}]}
    r = parse_jersey_response(resp)
    assert r is not None
    assert r.number == "11"
    assert abs(r.confidence - 0.92) < 1e-6


def test_parse_classification_leading_zero():
    resp = {"predictions": [{"class": "07", "confidence": 0.8}]}
    r = parse_jersey_response(resp)
    assert r is not None
    assert r.number == "7"  # normalized


def test_parse_detection_style_two_digits():
    """Each digit is its own box; we sort by x and concat."""
    resp = {"predictions": [
        {"class": "1", "x": 200.0, "confidence": 0.9, "width": 20, "height": 30},
        {"class": "1", "x": 150.0, "confidence": 0.9, "width": 20, "height": 30},
    ]}
    r = parse_jersey_response(resp)
    assert r is not None
    assert r.number == "11"


def test_parse_rejects_non_numeric():
    resp = {"predictions": [{"class": "X", "confidence": 0.9}]}
    assert parse_jersey_response(resp) is None


def test_parse_rejects_out_of_range():
    resp = {"predictions": [{"class": "100", "confidence": 0.9}]}
    assert parse_jersey_response(resp) is None


def test_parse_empty():
    assert parse_jersey_response({"predictions": []}) is None
    assert parse_jersey_response(None) is None


# ── crop_chest ──────────────────────────────────────────────────────────────

def test_crop_chest_basic():
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    bbox = (500, 200, 600, 500)  # 100x300
    crop = crop_chest(frame, bbox)
    # Vertical 15-55% of 300 = rows 245..365 → 120 tall
    # Horizontal 20-80% of 100 = cols 520..580 → 60 wide
    assert crop.shape[0] == 120
    assert crop.shape[1] == 60


def test_crop_chest_clamped_to_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    bbox = (-50, -50, 100, 200)
    crop = crop_chest(frame, bbox)
    # Should not crash and should return something non-degenerate
    assert crop.size > 0


def test_crop_chest_empty_bbox():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = crop_chest(frame, (500, 500, 500, 500))  # zero area
    assert crop.size == 0


# ── number-bbox matching + cropping ────────────────────────────────────────

def _num(x1, y1, x2, y2, conf=0.8):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return NumberDetection(bbox=(x1, y1, x2, y2), center=(cx, cy), confidence=conf)


def test_match_number_to_player_empty():
    """No numbers detected → no match."""
    assert match_number_to_player((100, 100, 200, 400), []) is None


def test_match_number_to_player_inside():
    """Number whose center sits in the player bbox is returned."""
    player = (100, 100, 200, 400)
    n = _num(140, 180, 160, 220)  # center (150, 200) inside
    assert match_number_to_player(player, [n]) is n


def test_match_number_to_player_outside():
    """Number whose center is OUTSIDE the player bbox isn't matched."""
    player = (100, 100, 200, 400)
    n = _num(300, 180, 320, 220)  # center (310, 200) — different player
    assert match_number_to_player(player, [n]) is None


def test_match_number_picks_higher_confidence():
    """When two numbers fall inside the bbox, prefer higher confidence."""
    player = (100, 100, 200, 400)
    low = _num(110, 180, 130, 220, conf=0.4)
    high = _num(160, 180, 180, 220, conf=0.9)
    out = match_number_to_player(player, [low, high])
    assert out is high


def test_crop_number_bbox_includes_padding():
    """Crop is bigger than the raw bbox (padding margin)."""
    frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
    n = _num(500, 300, 540, 350)  # 40x50 bbox
    crop = crop_number_bbox(frame, n)
    # 25% padding each side → +25% width, +25% height
    assert crop.shape[0] > 50
    assert crop.shape[1] > 40


def test_crop_number_bbox_clamped_to_frame():
    """Cropping near frame edge doesn't go negative or out of bounds."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    n = _num(0, 0, 30, 30)  # bbox right at corner
    crop = crop_number_bbox(frame, n)
    assert crop.size > 0


def test_preprocess_upscales_tiny_crops():
    """Tiny crops are upscaled to at least the OCR min height (96px)."""
    tiny = np.zeros((20, 16, 3), dtype=np.uint8)
    out = preprocess_for_ocr(tiny)
    assert out.shape[0] >= 96
    # Aspect ratio preserved (within rounding)
    src_ratio = 16 / 20
    out_ratio = out.shape[1] / out.shape[0]
    assert abs(out_ratio - src_ratio) < 0.1


def test_preprocess_passes_through_large_crops():
    """Crops already large enough are returned unchanged."""
    big = np.zeros((150, 120, 3), dtype=np.uint8)
    out = preprocess_for_ocr(big)
    assert out.shape == big.shape


def test_preprocess_handles_empty_crop():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    out = preprocess_for_ocr(empty)
    assert out.size == 0
