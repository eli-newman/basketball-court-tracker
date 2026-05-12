"""Tests for PlayerStatsAggregator — per-player rollup from ShotEvents."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from events import ShotEvent
from player_stats import PlayerStatsAggregator
from scoreboard import LEFT_RIM, THREE_POINT_DISTANCE_FT


# A spot close to the left rim — counts as a 2.
_TWO_SPOT = (LEFT_RIM[0] + 5.0, LEFT_RIM[1])
# A spot well beyond the 3-pt threshold from the left rim.
_THREE_SPOT = (LEFT_RIM[0] + THREE_POINT_DISTANCE_FT + 5.0, LEFT_RIM[1])


def _shot(
    *,
    player_id=1,
    team_id=0,
    made=True,
    court_xy=None,
    frame_end=30,
) -> ShotEvent:
    cx, cy = court_xy if court_xy is not None else _TWO_SPOT
    return ShotEvent(
        shooter_track_id=7,
        shot_type="jump-shot",
        frame_start=frame_end - 5,
        frame_end=frame_end,
        made=made,
        court_x=cx,
        court_y=cy,
        team_id=team_id,
        shooter_player_id=player_id,
    )


def test_empty_aggregator_has_no_rows():
    agg = PlayerStatsAggregator()
    assert agg.all() == []
    assert agg.top_scorers() == []


def test_single_made_two_pointer_credits_two_points():
    agg = PlayerStatsAggregator()
    agg.update([_shot(player_id=1, made=True, court_xy=_TWO_SPOT)])
    row = agg.get(1)
    assert row is not None
    assert row.points == 2
    assert row.fga == 1
    assert row.fgm == 1
    assert row.fg2a == 1
    assert row.fg2m == 1
    assert row.fg3a == 0
    assert row.fg3m == 0
    assert row.fg_pct == 1.0


def test_made_three_pointer_credits_three_points():
    agg = PlayerStatsAggregator()
    agg.update([_shot(player_id=1, made=True, court_xy=_THREE_SPOT)])
    row = agg.get(1)
    assert row.points == 3
    assert row.fg3a == 1
    assert row.fg3m == 1
    assert row.fg2a == 0


def test_missed_shot_credits_attempt_not_points():
    agg = PlayerStatsAggregator()
    agg.update([_shot(player_id=1, made=False, court_xy=_TWO_SPOT)])
    row = agg.get(1)
    assert row.points == 0
    assert row.fga == 1
    assert row.fgm == 0
    assert row.fg2a == 1
    assert row.fg2m == 0


def test_unidentified_shooter_is_dropped():
    """Shots with shooter_player_id=None must not show up in the per-player
    table at all — they go to the team scoreboard's unattributed bucket."""
    ev = _shot(player_id=1)
    ev.shooter_player_id = None
    agg = PlayerStatsAggregator()
    agg.update([ev])
    assert agg.all() == []


def test_dedupe_by_id_no_double_counting():
    """Pipeline rebuilds the events list every frame; the aggregator must
    dedupe by object identity so the same event isn't counted twice."""
    agg = PlayerStatsAggregator()
    ev = _shot(player_id=1, made=True, court_xy=_TWO_SPOT)
    agg.update([ev])
    agg.update([ev])  # same object, second time
    agg.update([ev, ev])
    row = agg.get(1)
    assert row.fga == 1
    assert row.points == 2


def test_top_scorers_orders_by_points_then_fgm():
    """Events kept in a list so id() doesn't get reused via GC — matches
    the pipeline's reality where ShotEvents live in EventDetector.events
    for the lifetime of the run.
    """
    agg = PlayerStatsAggregator()
    events = [
        # Player 1: 6 PTS (3 made 2s out of 5 attempts)
        _shot(player_id=1, made=True, court_xy=_TWO_SPOT, frame_end=10),
        _shot(player_id=1, made=True, court_xy=_TWO_SPOT, frame_end=20),
        _shot(player_id=1, made=True, court_xy=_TWO_SPOT, frame_end=30),
        _shot(player_id=1, made=False, frame_end=40),
        _shot(player_id=1, made=False, frame_end=50),
        # Player 2: 6 PTS (2 made 3s, only 2 FGM)
        _shot(player_id=2, team_id=1, made=True,
              court_xy=_THREE_SPOT, frame_end=60),
        _shot(player_id=2, team_id=1, made=True,
              court_xy=_THREE_SPOT, frame_end=70),
        # Player 3: 3 PTS
        _shot(player_id=3, team_id=0, made=True,
              court_xy=_THREE_SPOT, frame_end=80),
    ]
    agg.update(events)

    top = agg.top_scorers(n=3)
    assert [s.player_id for s in top] == [1, 2, 3]
    # Player 1 and 2 are tied on points but 1 has more FGM, so 1 ranks first.
    assert top[0].points == 6 and top[0].fgm == 3
    assert top[1].points == 6 and top[1].fgm == 2


def test_team_id_pinned_to_latest_shot():
    agg = PlayerStatsAggregator()
    events = [
        _shot(player_id=1, team_id=0, frame_end=10),
        # Team classifier flips mid-clip (rare but possible). Latest shot wins.
        _shot(player_id=1, team_id=1, frame_end=200),
    ]
    agg.update(events)
    assert agg.get(1).team_id == 1


def test_negative_team_id_does_not_overwrite_known_team():
    agg = PlayerStatsAggregator()
    events = [
        _shot(player_id=1, team_id=0, frame_end=10),
        # Subsequent shot has unknown team — must not clobber the known value.
        _shot(player_id=1, team_id=-1, frame_end=100),
    ]
    agg.update(events)
    assert agg.get(1).team_id == 0
