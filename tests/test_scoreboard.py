"""Tests for Scoreboard (ShotEvent stream → cumulative team scores)."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from events import ShotEvent
from scoreboard import (
    LEFT_RIM,
    RIGHT_RIM,
    THREE_POINT_DISTANCE_FT,
    Scoreboard,
)


def _shot(
    team_id: int = 0,
    made: bool = True,
    court_x: float = 10.0,
    court_y: float = 25.0,
    shooter: int = 1,
    frame_end: int = 30,
) -> ShotEvent:
    return ShotEvent(
        shooter_track_id=shooter,
        shot_type="jump-shot",
        frame_start=frame_end - 5,
        frame_end=frame_end,
        made=made,
        court_x=court_x,
        court_y=court_y,
        team_id=team_id,
    )


# ── Initial state ───────────────────────────────────────────────────────────


def test_initial_score_is_zero_for_both_teams():
    sb = Scoreboard(n_teams=2)
    assert sb.get(0) == 0
    assert sb.get(1) == 0
    assert sb.state.score_by_team == {0: 0, 1: 0}


def test_no_events_no_history():
    sb = Scoreboard(n_teams=2)
    sb.update([])
    assert sb.history == []


# ── 2 vs 3 classification by court geometry ─────────────────────────────────


def test_layup_at_left_rim_counts_as_2():
    """Shot from the rim is well inside the 3-point arc → 2 points."""
    sb = Scoreboard(n_teams=2)
    sb.update([_shot(team_id=0, court_x=LEFT_RIM[0] + 1, court_y=LEFT_RIM[1])])
    assert sb.get(0) == 2


def test_deep_three_from_top_of_key_counts_as_3():
    """A shot from beyond 22ft of the rim → 3 points."""
    sb = Scoreboard(n_teams=2)
    # Center of court at half-court — definitely beyond 22ft from either rim.
    sb.update([_shot(team_id=1, court_x=47.0, court_y=25.0)])
    assert sb.get(1) == 3


def test_long_two_just_inside_arc():
    """A shot from 20ft (inside arc) counts as 2."""
    sb = Scoreboard(n_teams=2)
    # 20ft from left rim, along the X axis
    sb.update([_shot(team_id=0, court_x=LEFT_RIM[0] + 20, court_y=LEFT_RIM[1])])
    assert sb.get(0) == 2


def test_threshold_exactly_at_arc_counts_as_3():
    """Distance exactly THREE_POINT_DISTANCE_FT counts as 3 (>= threshold)."""
    sb = Scoreboard(n_teams=2)
    sb.update([
        _shot(team_id=0,
              court_x=LEFT_RIM[0] + THREE_POINT_DISTANCE_FT,
              court_y=LEFT_RIM[1])
    ])
    assert sb.get(0) == 3


def test_right_rim_3_pointer():
    """Symmetry: same threshold applies to the right rim."""
    sb = Scoreboard(n_teams=2)
    sb.update([
        _shot(team_id=1,
              court_x=RIGHT_RIM[0] - 25,  # 25ft from right rim
              court_y=RIGHT_RIM[1])
    ])
    assert sb.get(1) == 3


# ── Made vs missed ──────────────────────────────────────────────────────────


def test_missed_shot_does_not_score():
    sb = Scoreboard(n_teams=2)
    sb.update([_shot(team_id=0, made=False, court_x=10.0)])
    assert sb.get(0) == 0
    # But the miss IS recorded in history with 0 points.
    history = sb.history
    assert len(history) == 1
    assert history[0].points == 0
    assert history[0].event.made is False


def test_missing_court_coords_defaults_to_two():
    """A made shot with no court position can't be classified — default 2."""
    sb = Scoreboard(n_teams=2)
    ev = _shot(team_id=0)
    ev.court_x = None
    ev.court_y = None
    sb.update([ev])
    assert sb.get(0) == 2
    assert sb.history[0].rim_side == "unknown"


# ── Team attribution ────────────────────────────────────────────────────────


def test_unattributed_made_shot_does_not_change_team_scores():
    """A make by a track with team_id=-1 increments unattributed, not a team."""
    sb = Scoreboard(n_teams=2)
    sb.update([_shot(team_id=-1, court_x=10.0)])  # 2-pointer, no team
    assert sb.get(0) == 0
    assert sb.get(1) == 0
    assert sb.state.unattributed_points == 2


def test_two_teams_separate_totals():
    sb = Scoreboard(n_teams=2)
    sb.update([
        _shot(team_id=0, court_x=10.0),                # +2
        _shot(team_id=1, court_x=47.0),                # +3
        _shot(team_id=0, court_x=47.0),                # +3
        _shot(team_id=1, made=False, court_x=10.0),    # miss
    ])
    assert sb.get(0) == 5
    assert sb.get(1) == 3


# ── Deduplication ───────────────────────────────────────────────────────────


def test_replaying_same_event_list_is_idempotent():
    """The pipeline replays the full events list every frame; calling update
    repeatedly must not double-count.
    """
    sb = Scoreboard(n_teams=2)
    events = [_shot(team_id=0, court_x=10.0), _shot(team_id=1, court_x=47.0)]
    sb.update(events)
    sb.update(events)
    sb.update(events)
    assert sb.get(0) == 2
    assert sb.get(1) == 3
    # History is also not duplicated.
    assert len(sb.history) == 2


def test_appending_new_events_picked_up_on_next_update():
    sb = Scoreboard(n_teams=2)
    events = [_shot(team_id=0, court_x=10.0)]
    sb.update(events)
    assert sb.get(0) == 2
    events.append(_shot(team_id=1, court_x=47.0))
    sb.update(events)
    assert sb.get(0) == 2
    assert sb.get(1) == 3


# ── Rim-side detection ──────────────────────────────────────────────────────


def test_left_rim_side_recorded():
    sb = Scoreboard(n_teams=2)
    sb.update([_shot(team_id=0, court_x=LEFT_RIM[0] + 1, court_y=LEFT_RIM[1])])
    assert sb.history[0].rim_side == "left"


def test_right_rim_side_recorded():
    sb = Scoreboard(n_teams=2)
    sb.update([_shot(team_id=1, court_x=RIGHT_RIM[0] - 1, court_y=RIGHT_RIM[1])])
    assert sb.history[0].rim_side == "right"
