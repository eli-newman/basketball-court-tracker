"""Tests for EventDetector (shot attempts + makes from action classes)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from detector import ActionObservation, PlayerDetection
from events import EventDetector, _bbox_iou
from mapper import MappedPlayer


def _player(track_id: int, bbox: tuple) -> PlayerDetection:
    return PlayerDetection(
        bbox=bbox,
        bottom_center=((bbox[0] + bbox[2]) / 2, bbox[3]),
        center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        confidence=0.9,
        class_name="player",
        track_id=track_id,
    )


def _action(bbox: tuple, cls: str, conf: float = 0.8) -> ActionObservation:
    return ActionObservation(
        bbox=bbox,
        center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        class_name=cls,
        confidence=conf,
    )


def _mapped(track_id: int, court_x: float = 70.0, court_y: float = 25.0,
            team_id: int = 0) -> MappedPlayer:
    return MappedPlayer(
        track_id=track_id, court_x=court_x, court_y=court_y,
        pixel_x=600, pixel_y=400,
        bbox=(550, 200, 650, 600), confidence=0.9, class_name="player",
        team_id=team_id,
    )


# ── bbox IoU helper ─────────────────────────────────────────────────────────


def test_bbox_iou_identical_is_one():
    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_bbox_iou_disjoint_is_zero():
    assert _bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_bbox_iou_half_overlap():
    # 10x10 boxes that share a 10x5 strip → IoU = 50 / 150 = 1/3
    iou = _bbox_iou((0, 0, 10, 10), (0, 5, 10, 15))
    assert abs(iou - (50.0 / 150.0)) < 1e-6


# ── EventDetector basic behaviour ───────────────────────────────────────────


def test_no_actions_no_events():
    ed = EventDetector(shot_window=3, shot_confirm_at=2)
    out = ed.update(
        frame_idx=1, actions=[], tracked_players=[_player(1, (100, 100, 200, 400))],
    )
    assert out == []
    assert ed.events == []


def test_single_frame_action_does_not_commit():
    """One frame of jump-shot is noise — never emit an event."""
    ed = EventDetector(shot_window=3, shot_confirm_at=2, end_after_idle_frames=2,
                       made_window_frames=3)
    p = _player(1, (100, 100, 200, 400))
    # One frame of jump-shot
    ed.update(1, [_action((100, 100, 200, 400), "player-jump-shot")], [p])
    # Then idle frames
    for f in range(2, 10):
        ed.update(f, [], [p])
    assert ed.events == []  # never confirmed → never emitted


def test_confirmed_missed_jump_shot_emits_event():
    """Three consecutive jump-shot frames → confirmed; no basket → missed."""
    ed = EventDetector(
        shot_window=4, shot_confirm_at=3, end_after_idle_frames=2,
        made_window_frames=5,
    )
    p = _player(1, (100, 100, 200, 400))
    mp = [_mapped(1, court_x=70.0, court_y=20.0, team_id=0)]

    # 3 shot frames
    for f in range(1, 4):
        ed.update(f, [_action((100, 100, 200, 400), "player-jump-shot")], [p], mp)
    # 2 idle frames close out the shot
    ed.update(4, [], [p], mp)
    ed.update(5, [], [p], mp)
    # 5 more idle frames drain the made-window → emit miss
    for f in range(6, 12):
        ed.update(f, [], [p], mp)

    assert len(ed.events) == 1
    ev = ed.events[0]
    assert ev.shooter_track_id == 1
    assert ev.shot_type == "jump-shot"
    assert ev.made is False
    assert ev.court_x == 70.0
    assert ev.team_id == 0


def test_confirmed_made_shot_when_basket_within_window():
    """Shot ends → ball-in-basket fires next frame → made."""
    ed = EventDetector(
        shot_window=4, shot_confirm_at=3, end_after_idle_frames=2,
        made_window_frames=5,
    )
    p = _player(1, (100, 100, 200, 400))
    mp = [_mapped(1)]

    for f in range(1, 4):
        ed.update(f, [_action((100, 100, 200, 400), "player-jump-shot")], [p], mp)
    # Close out (idle)
    ed.update(4, [], [p], mp)
    ed.update(5, [], [p], mp)
    # On frame 6, ball-in-basket fires → MADE
    ed.update(6, [_action((400, 50, 460, 110), "ball-in-basket")], [p], mp)

    assert len(ed.events) == 1
    assert ed.events[0].made is True


def test_basket_after_window_treats_shot_as_missed():
    """Ball-in-basket *after* made_window_frames doesn't credit the shot."""
    ed = EventDetector(
        shot_window=4, shot_confirm_at=3, end_after_idle_frames=2,
        made_window_frames=3,
    )
    p = _player(1, (100, 100, 200, 400))
    mp = [_mapped(1)]

    for f in range(1, 4):
        ed.update(f, [_action((100, 100, 200, 400), "player-jump-shot")], [p], mp)
    # Close out + drain made-window
    for f in range(4, 12):
        ed.update(f, [], [p], mp)
    # Now a basket — but too late to count for the shot above
    ed.update(13, [_action((400, 50, 460, 110), "ball-in-basket")], [p], mp)

    assert len(ed.events) == 1
    assert ed.events[0].made is False


def test_layup_dunk_classifies_as_layup_dunk():
    ed = EventDetector(
        shot_window=3, shot_confirm_at=2, end_after_idle_frames=2,
        made_window_frames=3,
    )
    p = _player(1, (100, 100, 200, 400))
    for f in range(1, 4):
        ed.update(f, [_action((100, 100, 200, 400), "player-layup-dunk")], [p],
                  [_mapped(1)])
    for f in range(4, 10):
        ed.update(f, [], [p], [_mapped(1)])
    assert len(ed.events) == 1
    assert ed.events[0].shot_type == "layup-dunk"


def test_action_unattributable_to_any_player_is_ignored():
    """Action bbox with no player IoU > min_action_iou → no shot."""
    ed = EventDetector(
        shot_window=3, shot_confirm_at=2, min_action_iou=0.10,
        end_after_idle_frames=2, made_window_frames=3,
    )
    # Player at top-left, action bbox at bottom-right — zero overlap.
    p = _player(1, (0, 0, 100, 100))
    for f in range(1, 5):
        ed.update(f, [_action((500, 500, 600, 600), "player-jump-shot")], [p])
    for f in range(5, 12):
        ed.update(f, [], [p])
    assert ed.events == []


def test_block_track_id_stamped_on_event():
    """player-shot-block during a shot is attached to the eventual event."""
    ed = EventDetector(
        shot_window=3, shot_confirm_at=2, end_after_idle_frames=2,
        made_window_frames=3,
    )
    shooter = _player(1, (100, 100, 200, 400))
    blocker = _player(2, (180, 100, 280, 400))
    mp = [_mapped(1, team_id=0), _mapped(2, team_id=1)]

    for f in range(1, 4):
        actions = [
            _action((100, 100, 200, 400), "player-jump-shot"),
            _action((180, 100, 280, 400), "player-shot-block"),
        ]
        ed.update(f, actions, [shooter, blocker], mp)
    for f in range(4, 12):
        ed.update(f, [], [shooter, blocker], mp)

    assert len(ed.events) == 1
    assert ed.events[0].block_track_id == 2


def test_two_separate_shots_emit_two_events():
    """A second shot well after the first should produce its own event."""
    ed = EventDetector(
        shot_window=3, shot_confirm_at=2, end_after_idle_frames=2,
        made_window_frames=3,
    )
    p = _player(1, (100, 100, 200, 400))
    mp = [_mapped(1)]
    # First shot: 3 frames, then idle to close it out and drain
    for f in range(1, 4):
        ed.update(f, [_action((100, 100, 200, 400), "player-jump-shot")], [p], mp)
    for f in range(4, 12):
        ed.update(f, [], [p], mp)
    # Second shot
    for f in range(13, 16):
        ed.update(f, [_action((100, 100, 200, 400), "player-jump-shot")], [p], mp)
    for f in range(16, 25):
        ed.update(f, [], [p], mp)
    assert len(ed.events) == 2


def test_constructor_rejects_bad_window():
    with pytest.raises(ValueError):
        EventDetector(shot_window=2, shot_confirm_at=3)
