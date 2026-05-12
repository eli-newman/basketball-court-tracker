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
_RIM_BOTTOM = _RIM_BBOX[3]                    # 110


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


def _run_through_resolution(
    det: GeometricShotDetector,
    start_frame: int,
    n_frames: int,
    ball_at: callable,
    mp: list,
    possessor: int = 7,
):
    """Run the detector for n_frames starting at start_frame.

    `ball_at(frame_idx)` returns a BallDetection (or None) for that frame.
    Returns the flat list of ShotEvents emitted across all frames.
    """
    emitted = []
    for f in range(start_frame, start_frame + n_frames):
        b = ball_at(f)
        out = det.update(f, b, [_rim()], mp, possessor)
        emitted.extend(out)
    return emitted


# ── Happy path: made shot ───────────────────────────────────────────────────


def test_ball_above_then_through_then_below_fires_made():
    """Through-the-net: ball above → enters rim → ends up below = MADE."""
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7, team_id=0)]

    # Frames 1-3: ball above the rim
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frame 4: ball enters rim bbox (trigger)
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    # Frames 5-9: ball drops through and emerges below the rim
    for i in range(5, 10):
        det.update(i, _ball(_RIM_CX, _RIM_BOTTOM + 20), [_rim()], mp, 7)
    # Frame 14: resolution window (10 frames after trigger=4) elapsed
    out = det.update(14, _ball(_RIM_CX, _RIM_BOTTOM + 60), [_rim()], mp, 7)
    assert len(out) == 1
    assert out[0].made is True
    assert out[0].shooter_track_id == 7
    assert out[0].team_id == 0


# ── Missed: rim-bounce that never goes below ────────────────────────────────


def test_ball_bounces_off_rim_back_up_fires_missed():
    """Front-iron: ball enters rim from above, bounces straight back up,
    never goes below the rim. Should resolve as MISSED, not MADE.
    """
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7, team_id=0)]

    # Frames 1-3: ball above rim
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frame 4: ball enters rim bbox (trigger)
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    # Frames 5-14: ball bounces back up — never below rim
    for i in range(5, 14):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 30), [_rim()], mp, 7)
    out = det.update(14, _ball(_RIM_CX, _RIM_TOP - 50), [_rim()], mp, 7)
    assert len(out) == 1
    assert out[0].made is False
    assert out[0].shooter_track_id == 7


def test_ball_clips_rim_to_the_side_fires_missed():
    """Ball brushes the rim, deflects sideways without going through."""
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7, team_id=0)]

    # Approach from above
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frame 4: clips rim
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    # Frames 5-14: ball flies off horizontally, never below rim
    for i in range(5, 14):
        det.update(i, _ball(_RIM_CX + 200, _RIM_TOP + 10), [_rim()], mp, 7)
    out = det.update(14, _ball(_RIM_CX + 300, _RIM_TOP + 20), [_rim()], mp, 7)
    assert len(out) == 1
    assert out[0].made is False


def test_ball_below_rim_but_not_through_rim_first_does_not_fire():
    """Just having the ball appear below the rim isn't enough — the trigger
    (ball entering the rim from above) is what starts the resolution.
    """
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7)]
    # Ball is always BELOW the rim — never enters the bbox.
    for i in range(1, 30):
        out = det.update(
            i, _ball(_RIM_CX, _RIM_BOTTOM + 30), [_rim()], mp, 7,
        )
        assert out == []


# ── Trigger gating ──────────────────────────────────────────────────────────


def test_ball_just_sitting_in_rim_does_not_trigger():
    """Without observed "above rim" history, no trigger and no event."""
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7)]
    # Ball starts inside the rim bbox; never observed above.
    for i in range(1, 20):
        out = det.update(
            i, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7,
        )
        assert out == []


def test_no_rim_observations_no_event():
    """Without rim detections, no trigger possible."""
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7)]
    for i in range(1, 20):
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
    """Ball at x=10 while rim is at center — not "above" the rim even
    though it's at low cy.
    """
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7)]
    # Ball above-but-far-left for several frames
    for i in range(1, 4):
        det.update(i, _ball(10, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frame 4: warps to inside the rim. No nearby above-rim history.
    out = det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    # Even if a resolution window started, finish it out and ensure
    # nothing erroneously emits.
    for i in range(5, 20):
        out = det.update(i, _ball(_RIM_CX, _RIM_BOTTOM + 30), [_rim()], mp, 7)
        if out:
            break
    assert out == []


# ── Refractory ──────────────────────────────────────────────────────────────


def test_refractory_prevents_immediate_repeat():
    """After a shot fires, suppress new triggers for refractory_frames."""
    det = GeometricShotDetector(resolution_frames=10, refractory_frames=20)
    mp = [_mapped(track_id=7)]

    # Fire one MADE shot at frame ~14 (4 trigger + 10 resolution)
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    for i in range(5, 14):
        det.update(i, _ball(_RIM_CX, _RIM_BOTTOM + 20), [_rim()], mp, 7)
    out = det.update(14, _ball(_RIM_CX, _RIM_BOTTOM + 30), [_rim()], mp, 7)
    assert len(out) == 1
    assert out[0].made is True

    # Try to trigger another shot immediately — refractory blocks it.
    for i in range(15, 18):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    out = det.update(18, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    # Within refractory of frame 14 → no new pending shot
    assert out == []


# ── Shooter attribution ─────────────────────────────────────────────────────


def test_uses_most_recent_possessor_when_release_before_made():
    """Possession often releases the moment the ball leaves a shooter's
    hand. The detector walks backwards through the possessor buffer to
    find the most recent non-None possessor at trigger time.
    """
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7, team_id=0)]
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    # Frame 4: ball enters rim — possession already released this frame
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, None)
    # Make-style descent
    for i in range(5, 14):
        det.update(i, _ball(_RIM_CX, _RIM_BOTTOM + 20), [_rim()], mp, None)
    out = det.update(14, _ball(_RIM_CX, _RIM_BOTTOM + 30), [_rim()], mp, None)
    assert len(out) == 1
    assert out[0].shooter_track_id == 7  # Last known possessor.


def test_records_team_and_court_pos_from_mapped_players():
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=11, court_x=85.0, court_y=28.0, team_id=1)]
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 11)
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 11)
    for i in range(5, 14):
        det.update(i, _ball(_RIM_CX, _RIM_BOTTOM + 20), [_rim()], mp, 11)
    out = det.update(14, _ball(_RIM_CX, _RIM_BOTTOM + 30), [_rim()], mp, 11)
    assert len(out) == 1
    assert out[0].team_id == 1
    assert out[0].court_x == 85.0
    assert out[0].court_y == 28.0


def test_no_possessor_emits_event_with_unknown_shooter():
    """Ball through rim with no recorded possessor → shooter=-1."""
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7)]
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, None)
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, None)
    for i in range(5, 14):
        det.update(i, _ball(_RIM_CX, _RIM_BOTTOM + 20), [_rim()], mp, None)
    out = det.update(14, _ball(_RIM_CX, _RIM_BOTTOM + 30), [_rim()], mp, None)
    assert len(out) == 1
    assert out[0].shooter_track_id == -1
    assert out[0].team_id == -1


# ── Reset behavior ──────────────────────────────────────────────────────────


def test_reset_drops_pending_shot():
    """A pending shot at the moment of reset is discarded — no event after."""
    det = GeometricShotDetector(resolution_frames=10)
    mp = [_mapped(track_id=7)]
    for i in range(1, 4):
        det.update(i, _ball(_RIM_CX, _RIM_TOP - 40), [_rim()], mp, 7)
    det.update(4, _ball(_RIM_CX, _RIM_TOP + 5), [_rim()], mp, 7)
    det.reset()
    # Drive past the resolution window — nothing should emit.
    for i in range(5, 25):
        out = det.update(i, _ball(_RIM_CX, _RIM_BOTTOM + 30), [_rim()], mp, 7)
        assert out == []
