"""Live scoreboard — turns ShotEvent stream into cumulative team scores.

Reads:
  - ShotEvents from EventDetector (shooter_track_id, team_id, made,
    court_x, court_y, shot_type)

Outputs:
  - Cumulative score per team_id, updated whenever a made shot resolves.
  - Per-event point value (2 or 3) derived from court geometry.

Point classification
--------------------
We use a single 22ft threshold from the nearer rim. NBA arc is actually
23.75ft straight-on and 22ft in the corners, so 22ft slightly over-counts
3s near the corner break — that's the simplest fix that doesn't require
a polygon test. Free throws can't be cleanly distinguished from inside
shots without OCR on the scoreboard or referee-signal recognition, so
they currently count as 2 — accept the bias for now.

Inferring "which team scored"
-----------------------------
Possession is brittle on a contested layup, and the model doesn't emit
which rim the ball went through. So we use the shooter's team_id
directly: ShotEvent already records the shooter (via action-class IoU
attribution), and the shooter's team_id is stamped on the event from the
MappedPlayer at shot time. A confirmed make → the shooter's team gets
points. If the shooter's team_id is unknown (-1), the points are still
counted in `unattributed_points` but don't change either team's total.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from court import BASKET_OFFSET, COURT_LENGTH, COURT_WIDTH
from events import ShotEvent


# Rims in court coordinates (feet). Origin top-left, X = length, Y = width.
LEFT_RIM: Tuple[float, float] = (BASKET_OFFSET, COURT_WIDTH / 2)
RIGHT_RIM: Tuple[float, float] = (COURT_LENGTH - BASKET_OFFSET, COURT_WIDTH / 2)

# Threshold (feet) from the nearer rim that counts as a 3-pointer. NBA
# straight-on arc is 23.75ft; corners are 22ft. 22 over-counts a few
# break-of-arc shots as 3s — fine until we have time for a polygon test.
THREE_POINT_DISTANCE_FT: float = 22.0


@dataclass
class ScoredShot:
    """Same as ShotEvent but with the resolved point value attached."""
    event: ShotEvent
    points: int            # 0 (miss), 2, or 3
    rim_side: str          # "left" | "right" | "unknown"


@dataclass
class ScoreboardState:
    """Snapshot of the scoreboard at a point in time."""
    score_by_team: Dict[int, int] = field(default_factory=dict)
    last_scored: Optional[ScoredShot] = None
    unattributed_points: int = 0   # makes by tracks with team_id < 0


class Scoreboard:
    """Maintains cumulative team scores from the ShotEvent stream."""

    def __init__(self, n_teams: int = 2):
        self.n_teams = n_teams
        self._score: Dict[int, int] = {i: 0 for i in range(n_teams)}
        self._unattributed: int = 0
        self._history: List[ScoredShot] = []
        self._processed_event_ids: set = set()

    def update(self, events: List[ShotEvent]) -> List[ScoredShot]:
        """Process any events not yet seen; return the new ScoredShots.

        We dedupe by Python object identity rather than a field, because
        ShotEvent doesn't carry a unique ID — the EventDetector produces
        a fresh object per emit, so id() is stable for the lifetime of
        the run.
        """
        new_scored: List[ScoredShot] = []
        for ev in events:
            key = id(ev)
            if key in self._processed_event_ids:
                continue
            self._processed_event_ids.add(key)
            scored = self._score_event(ev)
            self._history.append(scored)
            if scored.points > 0:
                if 0 <= ev.team_id < self.n_teams:
                    self._score[ev.team_id] += scored.points
                else:
                    self._unattributed += scored.points
            new_scored.append(scored)
        return new_scored

    def _score_event(self, ev: ShotEvent) -> ScoredShot:
        """Determine point value and which rim the ball went through.

        Misses are 0 points but still recorded so the history is complete.
        """
        if not ev.made:
            return ScoredShot(event=ev, points=0, rim_side="unknown")

        if ev.court_x is None or ev.court_y is None:
            # No court position recorded — can't classify 2 vs 3.
            # Conservative default: count as 2.
            return ScoredShot(event=ev, points=2, rim_side="unknown")

        d_left = _euclid(ev.court_x, ev.court_y, *LEFT_RIM)
        d_right = _euclid(ev.court_x, ev.court_y, *RIGHT_RIM)
        if d_left <= d_right:
            rim_side, distance = "left", d_left
        else:
            rim_side, distance = "right", d_right

        points = 3 if distance >= THREE_POINT_DISTANCE_FT else 2
        return ScoredShot(event=ev, points=points, rim_side=rim_side)

    @property
    def state(self) -> ScoreboardState:
        return ScoreboardState(
            score_by_team=dict(self._score),
            last_scored=(
                self._history[-1] if self._history else None
            ),
            unattributed_points=self._unattributed,
        )

    @property
    def history(self) -> List[ScoredShot]:
        return list(self._history)

    def get(self, team_id: int) -> int:
        return self._score.get(team_id, 0)


def _euclid(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return (dx * dx + dy * dy) ** 0.5
