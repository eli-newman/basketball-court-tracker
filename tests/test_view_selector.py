"""Tests for ActiveHalfSelector."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from homography import CourtKeypoint
from view_selector import ActiveHalfSelector


def _kp(idx: int, conf: float = 0.9) -> CourtKeypoint:
    return CourtKeypoint(name=idx, pixel_x=0.0, pixel_y=0.0, confidence=conf)


def test_left_dominant_returns_left():
    selector = ActiveHalfSelector(history_frames=1)
    # All 15 left keypoints at high confidence, no right
    kps = [_kp(i, 0.9) for i in range(15)]
    assert selector.update(kps) == "left"


def test_right_dominant_returns_right():
    selector = ActiveHalfSelector(history_frames=1)
    kps = [_kp(i, 0.9) for i in range(18, 33)]
    assert selector.update(kps) == "right"


def test_ambiguous_returns_none_initially():
    selector = ActiveHalfSelector(history_frames=1, ratio_threshold=1.5)
    # Equal confidence on both sides
    kps = [_kp(i, 0.5) for i in range(0, 15)] + [_kp(i, 0.5) for i in range(18, 33)]
    assert selector.update(kps) is None


def test_ambiguous_keeps_last_decision():
    selector = ActiveHalfSelector(history_frames=3, ratio_threshold=1.5)
    # Frame 1: clearly right
    selector.update([_kp(i, 0.9) for i in range(18, 33)])
    # Frame 2: ambiguous — should keep "right" (still in history)
    ambiguous = [_kp(i, 0.5) for i in range(0, 15)] + [_kp(i, 0.5) for i in range(18, 33)]
    assert selector.update(ambiguous) == "right"


def test_hysteresis_resists_single_frame_flip():
    selector = ActiveHalfSelector(history_frames=10, ratio_threshold=1.5)
    # 9 frames of right
    for _ in range(9):
        selector.update([_kp(i, 0.9) for i in range(18, 33)])
    # 1 frame of left — majority of last 10 is still right
    decision = selector.update([_kp(i, 0.9) for i in range(0, 15)])
    assert decision == "right"


def test_hysteresis_eventually_flips_when_majority_changes():
    selector = ActiveHalfSelector(history_frames=5, ratio_threshold=1.5)
    # 5 frames of right
    for _ in range(5):
        selector.update([_kp(i, 0.9) for i in range(18, 33)])
    assert selector._last_decision == "right"
    # 5 frames of left — now majority of last 5 is left
    for _ in range(5):
        selector.update([_kp(i, 0.9) for i in range(0, 15)])
    assert selector._last_decision == "left"


def test_no_keypoints_returns_none():
    selector = ActiveHalfSelector(history_frames=1)
    assert selector.update([]) is None


def test_only_bridge_keypoints_returns_none():
    """Keypoints 15-17 are half-court markers, ignored from the vote."""
    selector = ActiveHalfSelector(history_frames=1)
    kps = [_kp(15, 0.9), _kp(16, 0.9), _kp(17, 0.9)]
    assert selector.update(kps) is None


def test_string_name_class_id_works():
    """Ensure string class_ids ('5', '20') still resolve."""
    selector = ActiveHalfSelector(history_frames=1)
    kps = [
        CourtKeypoint(name="5", pixel_x=0, pixel_y=0, confidence=0.9),
        CourtKeypoint(name="6", pixel_x=0, pixel_y=0, confidence=0.9),
    ]
    assert selector.update(kps) == "left"


def test_unparseable_string_name_skipped():
    """Garbage names ("foo", "bar") shouldn't crash."""
    selector = ActiveHalfSelector(history_frames=1)
    kps = [CourtKeypoint(name="foo", pixel_x=0, pixel_y=0, confidence=0.9)]
    assert selector.update(kps) is None


def test_reset_clears_history():
    selector = ActiveHalfSelector(history_frames=5)
    selector.update([_kp(i, 0.9) for i in range(0, 15)])
    assert selector._last_decision == "left"
    selector.reset()
    assert selector._last_decision is None
    assert len(selector.history) == 0
