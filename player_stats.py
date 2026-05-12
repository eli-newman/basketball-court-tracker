"""Per-player stat aggregation from the ShotEvent stream.

This is the payoff of the persistent-identity layer: with stable
player_ids, we can compute meaningful per-player numbers (PTS, FGM,
FGA, FG%) across camera cuts, instead of fragmenting them across
every ByteTrack id reissue.

Mechanics
---------
- We consume the same ShotEvent stream the Scoreboard consumes.
- We dedupe by `id()` of each event (same trick as Scoreboard — events
  are mutable dataclasses without unique field ids).
- We aggregate per `shooter_player_id` when present; events with
  `shooter_player_id is None` are dropped (we don't pollute the
  per-player table with the unattributed bucket — the team scoreboard
  already shows that signal).
- 2-vs-3 classification reuses Scoreboard's helpers so the per-player
  point totals always agree with the team scoreboard.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from events import ShotEvent
from scoreboard import LEFT_RIM, RIGHT_RIM, THREE_POINT_DISTANCE_FT, _euclid


@dataclass
class PlayerStats:
    """Counting stats for a single persistent player_id."""
    player_id: int
    team_id: int
    points: int = 0
    fga: int = 0          # field goal attempts (made + missed)
    fgm: int = 0          # field goal makes
    fg3a: int = 0
    fg3m: int = 0
    fg2a: int = 0
    fg2m: int = 0

    @property
    def fg_pct(self) -> float:
        return (self.fgm / self.fga) if self.fga else 0.0

    @property
    def fg3_pct(self) -> float:
        return (self.fg3m / self.fg3a) if self.fg3a else 0.0


class PlayerStatsAggregator:
    """Cumulative per-player stats keyed by persistent player_id.

    Mirrors Scoreboard's contract: call `update(events)` each frame
    with the full event list, and we'll process anything new.
    """

    def __init__(self):
        self._stats: Dict[int, PlayerStats] = {}
        self._processed_event_ids: set = set()

    def update(self, events: List[ShotEvent]) -> List[PlayerStats]:
        """Process any unseen events; return the updated rows for new shots."""
        touched: List[PlayerStats] = []
        for ev in events:
            key = id(ev)
            if key in self._processed_event_ids:
                continue
            self._processed_event_ids.add(key)
            if ev.shooter_player_id is None:
                continue   # un-identified shooter — skip per-player rollup

            row = self._stats.setdefault(
                ev.shooter_player_id,
                PlayerStats(player_id=ev.shooter_player_id, team_id=ev.team_id),
            )
            # team_id can change frame-to-frame in pathological cases (team
            # classifier ambiguity). Pin to the value at the latest shot —
            # cheap and almost always right.
            if ev.team_id >= 0:
                row.team_id = ev.team_id

            points = _points_for(ev)
            is_three = points == 3
            row.fga += 1
            if is_three:
                row.fg3a += 1
            else:
                row.fg2a += 1
            if ev.made:
                row.fgm += 1
                row.points += points
                if is_three:
                    row.fg3m += 1
                else:
                    row.fg2m += 1
            touched.append(row)
        return touched

    def get(self, player_id: int) -> Optional[PlayerStats]:
        return self._stats.get(player_id)

    def all(self) -> List[PlayerStats]:
        """Every player with at least one attempt, in player_id order."""
        return sorted(self._stats.values(), key=lambda s: s.player_id)

    def top_scorers(self, n: int = 5) -> List[PlayerStats]:
        """Highest-scoring players first. Ties broken by FGM, then attempts."""
        return sorted(
            self._stats.values(),
            key=lambda s: (-s.points, -s.fgm, -s.fga),
        )[:n]


def _points_for(ev: ShotEvent) -> int:
    """How many points this shot would be if made. Returns 2 or 3.

    Always returns a non-zero value: callers want to know whether the
    attempt was *worth* 2 or 3, not whether it went in. The aggregator
    decides whether to credit points based on ev.made separately.
    """
    if ev.court_x is None or ev.court_y is None:
        return 2   # match Scoreboard's default for unknown positions
    d_left = _euclid(ev.court_x, ev.court_y, *LEFT_RIM)
    d_right = _euclid(ev.court_x, ev.court_y, *RIGHT_RIM)
    distance = min(d_left, d_right)
    return 3 if distance >= THREE_POINT_DISTANCE_FT else 2
