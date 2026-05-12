"""Tests for GeometricShotDetector (ball-through-rim → ShotEvent)."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from detector import ActionObservation, BallDetection
from geo_shot import GeometricShotDetector
from mapper import MappedPlayer


# Rim sits at pixel (640, 100) — a small 30x20 box in the top half of frame.
_RIM_BBOX = (625, 90, 655, 110)
_RIM_CX = (_RIM_BBOX[0] + _RIM_BBOX[2]) / 2  # 640
_RIM_TOP = _RIM_BBOX[1]                       # 90


def _rim() -> ActionObservation:
    return ActionObservation(
        bbox=_RIM_BBOX,
        center=(_RIM_CX, (_RIM_BBOX[1] + _RIM_BBOX[3]) / 2),
        class_name="rim",
        confidence=0.9,
    )


def _ball(cx: float, cy: float) -> BallDetection:
    return BallDetection(
        bbox=(cx - 8, cy - 8, cx + 8, cy + 8),
        center=(cx, cy),
        bottom_center=(cx, cy + 8),
        confidence=0.85,
    )


def _mapped(track_id: int, court_x: float = 50.0, court_y: float = 25.0,
            team_id: int = 0) -> MappedPlayer:
    return MappedPlayer(
        track_id=track_id, court_x=court_x, court_y=court_y,
        pixel_x=600, pixel_y=400,
        bbox=(550, 200, 650, 600), confidence=0.9, class_name="player",
        team_id=team_id,
    )


# ── Happy path ──────────────────────────────────────────────────────────────


def test_ball_descending_through_rim_fires_event():
    """Ball above rim → inside rim → ShotEvent emitted."""
    det = GeometricShotDetector()
    mp = [_mapped(track_id=7, team_id=0)]

    # Frame 1-3: ball above the rim (cy < rim_top)
    for i in range(1, 4):
        det.update(
            frame_idx=i, ball=_ball(_RIM_CX, _RIM_TOP - 40),
            actions=[_rim()], mapped_players=mp,
            possessor_track_id=7,
        )

    # Frame 4: ball drops INTO the rim bbox — fires
    out = det.update(
        frame_idx=4, ball=_ball(_RIM_CX, _RIM_TOP + 5),
        actions=[_rim()], mapped_players=mp,
        possessor_track_id=7,
    )
    assert len(out) == 1
    ev = out[0]
    assert ev.made is True
    assert ev.shooter_track_id == 7
    assert ev.team_id == 0
    assert ev.frame_end == 4


def test_ball_just_sitting_in_rim_does_not_fire():
    """If the ball was never above the rim, no event fires.

    Prevents false positives from a ball detected near the rim at the
    start of a possession or after a pass.
    """
    det = GeometricShotDetector()
    mp = [_mapped(track_id=7)]
    # Ball is inside the rim bbox immediately, never observed above it.
    for i in range(1, 10):
        out = det.update(
            frame_idx=i, ball=_ball(_RIM_CX, _RIM_TOP + 5),
            actions=[_rim()], mapped_players=mp,
            possessor_track_id=7,
        )
        assert out == []


def test_refractory_prevents_immediate_repeat():
    """After a shot fires, suppress further shots for refractory_frames."""
    det = GeometricShotDetector(refractory_frames=20)
    mp = [_mapped(track_id=7)]

    # Set up + fire shot at frame 4
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    out = det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    assert len(out) == 1

    # Frames 5-23: even with valid "above then through" pattern, no events.
    for f_offset in range(1, 21):
        if f_offset <= 3:
            ball = _ball(_RIM_CX, _RIM_TOP - 40)  # above
        else:
            ball = _ball(_RIM_CX, _RIM_TOP + 5)   # inside
        out = det.update(4 + f_offset, ball, [_rim()], mp, 7)
        assert out == []


def test_no_rim_observations_no_event():
    """Without rim detections, the detector can't fire."""
    det = GeometricShotDetector()
    mp = [_mapped(track_id=7)]
    for i in range(1, 10):
        out = det.update(
            i, _ball(_RIM_CX, _RIM_TOP + 5), [], mp, 7,
        )
        assert out == []


def test_no_ball_no_event():
    det = GeometricShotDetector()
    mp = [_mapped(track_id=7)]
    out = det.update(1, None, [_rim()], mp, 7)
    assert out == []


def test_ball_horizontally_far_from_rim_does_not_qualify_as_above():
    """A ball at far-left while rim is at center should NOT register as
    "above" the rim. Without this gate, a ball moving along the floor
    that happens to cross the rim's x-range could trigger.
    """
    det = GeometricShotDetector()
    mp = [_mapped(track_id=7)]
    # Ball at x=10 (way left of rim at x=640), low cy
    for i in range(1, 4):
        det.update(i, _ball(10, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frame 4: ball "warps" into the rim — but no above-rim history nearby
    out = det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    assert out == []


# ── Shooter attribution ─────────────────────────────────────────────────────


def test_uses_most_recent_possessor_when_possessor_release_before_made():
    """Possession often releases the moment the ball leaves a shooter's
    hand. The geo detector walks backwards through the possessor buffer
    to find the most recent non-None possessor.
    """
    det = GeometricShotDetector()
    mp = [_mapped(track_id=7, team_id=0)]
    # Frame 1-3: above rim, possessor=7
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frame 4: ball is INSIDE rim — but possession was already released
    # (None for the last frame).
    det.update(4, _ball(_RIM_CX, _RIM_TOP - 5), [_rim()], mp, None)
    # Frame 5: ball drops into rim
    out = det.update(5, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, None)
    assert len(out) == 1
    # Should attribute to track 7 (the last known possessor).
    assert out[0].shooter_track_id == 7


def test_records_shooter_team_and_court_pos_from_mapped_players():
    det = GeometricShotDetector()
    mp = [_mapped(track_id=11, court_x=85.0, court_y=28.0, team_id=1)]
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 11)
    out = det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 11)
    assert len(out) == 1
    assert out[0].team_id == 1
    assert out[0].court_x == 85.0
    assert out[0].court_y == 28.0


def test_no_possessor_emits_event_with_unknown_shooter():
    """The ball can pass through the rim with no recorded possessor — we
    still want the event so the scoreboard counts it as `unattributed`.
    """
    det = GeometricShotDetector()
    mp = [_mapped(track_id=7)]
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, None)
    out = det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, None)
    assert len(out) == 1
    assert out[0].shooter_track_id == -1
    assert out[0].team_id == -1


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_above_rim_too_long_ago_does_not_count():
    """A ball that was above the rim 60 frames ago, then bounces around,
    and finally enters the rim, doesn't count — the "shot" connection is
    too stale.
    """
    det = GeometricShotDetector(
        max_shot_duration_frames=20, refractory_frames=5,
    )
    mp = [_mapped(track_id=7)]
    # Frame 1: above the rim
    det.update(1, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frames 2-50: ball wandering far from rim (NOT above)
    for i in range(2, 51):
        det.update(
            i, _ball(_RIM_CX + 500, _RIM_TOP + 200),
            [_rim()], mp, 7,
        )
    # Frame 51: ball enters rim — but it's been 50 frames since "above"
    out = det.update(51, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    assert out == []
