"""Tests for jersey vote aggregation + response parsing."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from jersey import (
    JerseyRead,
    JerseyVoter,
    crop_chest,
    parse_jersey_response,
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
