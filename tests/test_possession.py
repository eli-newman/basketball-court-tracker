"""Tests for PossessionTracker."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from detector import BallDetection, PlayerDetection
from possession import PossessionTracker, _bbox_to_point_distance


def _player(track_id: int, bbox: tuple) -> PlayerDetection:
    return PlayerDetection(
        bbox=bbox,
        bottom_center=((bbox[0] + bbox[2]) / 2, bbox[3]),
        center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        confidence=0.9,
        class_name="player",
        track_id=track_id,
    )


def _ball(cx: float, cy: float, size: int = 30) -> BallDetection:
    half = size / 2
    return BallDetection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        center=(cx, cy),
        bottom_center=(cx, cy + half),
        confidence=0.8,
    )


# ── _bbox_to_point_distance ────────────────────────────────────────────────


def test_bbox_distance_point_inside_is_zero():
    assert _bbox_to_point_distance((0, 0, 100, 100), 50, 50) == 0.0


def test_bbox_distance_point_on_edge_is_zero():
    assert _bbox_to_point_distance((0, 0, 100, 100), 0, 50) == 0.0


def test_bbox_distance_point_right_of_bbox():
    # Bbox is x=[0,100], point at x=120 → dx = 20, dy = 0 → dist = 20
    assert _bbox_to_point_distance((0, 0, 100, 100), 120, 50) == 20.0


def test_bbox_distance_point_diagonal():
    # Bbox is x=[0,100], y=[0,100]; point at (103, 104) → dx=3, dy=4 → dist=5
    assert _bbox_to_point_distance((0, 0, 100, 100), 103, 104) == 5.0


# ── PossessionTracker basic behaviour ───────────────────────────────────────


def test_no_ball_no_possessor():
    pt = PossessionTracker(confirm_at=1)
    out = pt.update(None, [_player(1, (100, 100, 200, 300))])
    assert out.possessor_track_id is None
    assert out.candidate_track_id is None
    assert out.distance_px is None


def test_no_players_no_possessor():
    pt = PossessionTracker(confirm_at=1)
    out = pt.update(_ball(150, 200), [])
    assert out.possessor_track_id is None
    assert out.candidate_track_id is None


def test_ball_inside_player_bbox_immediately_commits_with_confirm_at_1():
    pt = PossessionTracker(confirm_at=1, history_len=1)
    p = _player(1, (100, 100, 200, 300))
    ball = _ball(150, 200)  # inside the bbox
    out = pt.update(ball, [p])
    assert out.candidate_track_id == 1
    assert out.distance_px == 0.0
    assert out.possessor_track_id == 1


def test_requires_consecutive_frames_to_commit():
    """Track only commits possessor after confirm_at consecutive frames."""
    pt = PossessionTracker(confirm_at=3, history_len=5)
    p = _player(1, (100, 100, 200, 300))
    ball = _ball(150, 200)

    out = pt.update(ball, [p])
    assert out.candidate_track_id == 1
    assert out.possessor_track_id is None  # not yet committed

    out = pt.update(ball, [p])
    assert out.possessor_track_id is None  # still pending

    out = pt.update(ball, [p])
    assert out.possessor_track_id == 1  # committed


def test_flickering_candidates_block_commit():
    """If the nearest candidate flips frame-to-frame, possession never locks."""
    pt = PossessionTracker(confirm_at=3, history_len=5)
    p1 = _player(1, (100, 100, 200, 300))
    p2 = _player(2, (300, 100, 400, 300))

    # Frame 1: ball near p1 → candidate p1
    out = pt.update(_ball(150, 200), [p1, p2])
    assert out.candidate_track_id == 1
    # Frame 2: ball near p2
    out = pt.update(_ball(350, 200), [p1, p2])
    assert out.candidate_track_id == 2
    # Frame 3: back to p1
    out = pt.update(_ball(150, 200), [p1, p2])
    assert out.candidate_track_id == 1
    assert out.possessor_track_id is None  # never locked


def test_loose_ball_drops_possession_after_release_window():
    """Once held, possession drops to None after N frames of no ball."""
    pt = PossessionTracker(
        confirm_at=1, history_len=1, release_after_missing=3,
    )
    p = _player(1, (100, 100, 200, 300))

    # Lock possession
    pt.update(_ball(150, 200), [p])
    assert pt.current_possessor == 1

    # Ball goes missing
    pt.update(None, [p])
    assert pt.current_possessor == 1  # still holds
    pt.update(None, [p])
    assert pt.current_possessor == 1
    pt.update(None, [p])
    # 3rd missing frame triggers release
    assert pt.current_possessor is None


def test_ball_too_far_treated_as_loose():
    """Beyond max_distance_px, the candidate is suppressed."""
    pt = PossessionTracker(max_distance_px=50.0, confirm_at=1)
    p = _player(1, (0, 0, 50, 50))
    # Ball ~140px from the bbox edge → far beyond 50px
    ball = _ball(190, 190)
    out = pt.update(ball, [p])
    assert out.candidate_track_id is None
    # But distance is reported (so callers can debug)
    assert out.distance_px is not None and out.distance_px > 50.0


def test_untracked_players_cannot_be_possessor():
    """Track-id -1 detections are ignored — they have no stable identity."""
    pt = PossessionTracker(confirm_at=1)
    untracked = _player(-1, (100, 100, 200, 300))
    out = pt.update(_ball(150, 200), [untracked])
    assert out.candidate_track_id is None


def test_real_possession_change_eventually_flips_after_steal():
    """A clean change of possession to a new player commits after confirm_at."""
    pt = PossessionTracker(confirm_at=2, history_len=4)
    p1 = _player(1, (100, 100, 200, 300))
    p2 = _player(2, (500, 100, 600, 300))

    # Lock with p1
    pt.update(_ball(150, 200), [p1, p2])
    pt.update(_ball(150, 200), [p1, p2])
    assert pt.current_possessor == 1

    # Now ball moves to p2 — takes confirm_at frames to flip
    out = pt.update(_ball(550, 200), [p1, p2])
    assert out.candidate_track_id == 2
    assert out.possessor_track_id == 1  # not yet
    out = pt.update(_ball(550, 200), [p1, p2])
    assert out.possessor_track_id == 2  # committed


def test_history_len_must_be_at_least_confirm_at():
    """Misconfiguration is rejected at construction so the vote can succeed."""
    with pytest.raises(ValueError):
        PossessionTracker(confirm_at=5, history_len=3)
