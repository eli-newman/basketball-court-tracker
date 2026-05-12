"""Geometric shot detection — emit ShotEvents from ball + rim geometry alone.

The action-class shot detector in events.py depends on the player model
emitting `player-jump-shot`, `player-layup-dunk`, `ball-in-basket`, etc.
Some model versions (including basketball-player-detection-3-ycjdo/6,
which we use by default) DON'T emit those classes, so EventDetector
never fires.

This module fills the gap. It only needs:
  - `ball` detections per frame (already in BallDetection)
  - `rim` detections per frame (already in ActionObservation)
  - The current possessor's track_id from PossessionTracker (so the
    shot can be attributed)

A made shot is detected when the ball center enters a rim's bbox after
having been ABOVE the rim in the recent past. That's the unambiguous
"swish/through" signal — it can't fire on a ball merely passing by or
sitting on the rim.

We deliberately don't try to detect MISSED shots here. The action-class
detector handles those when they're available; doing it geometrically
requires tracking the ball through a bounce arc and is brittle. Missed
attempts are still implicitly logged via possession changes — a future
extension can build a richer model.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from detector import ActionObservation, BallDetection
from events import ShotEvent
from mapper import MappedPlayer


# Maximum frames between "ball above rim" and "ball through rim" for a
# detection to fire. At 30fps, 30 frames ≈ 1s — covers the time from
# release through net.
_MAX_SHOT_DURATION_FRAMES = 30

# After firing a shot, suppress new emissions for this many frames so a
# single basket doesn't trigger repeatedly as the ball continues through.
_REFRACTORY_FRAMES = 20

# How many recent frames to look back when attributing a shot to a
# possessor. The possession tracker may release the ball once it leaves
# the shooter's hands, so we keep a memory of recent possessors.
_POSSESSOR_LOOKBACK_FRAMES = 30


@dataclass
class _BallSample:
    frame_idx: int
    cx: float
    cy: float
    # Whether this sample was strictly ABOVE every rim bbox we saw
    # this frame. Pre-computed because rim positions vary per frame.
    above_any_rim: bool


class GeometricShotDetector:
    """Emits a ShotEvent when the ball passes through a rim from above."""

    def __init__(
        self,
        max_shot_duration_frames: int = _MAX_SHOT_DURATION_FRAMES,
        refractory_frames: int = _REFRACTORY_FRAMES,
        possessor_lookback_frames: int = _POSSESSOR_LOOKBACK_FRAMES,
    ):
        self.max_shot_duration_frames = max_shot_duration_frames
        self.refractory_frames = refractory_frames
        self.possessor_lookback_frames = possessor_lookback_frames

        self._ball_history: Deque[_BallSample] = deque(
            maxlen=max(max_shot_duration_frames, possessor_lookback_frames) + 5,
        )
        # frame_idx → possessor_track_id (or None). Same window as history.
        self._possessor_history: Deque[Tuple[int, Optional[int]]] = deque(
            maxlen=possessor_lookback_frames + 5,
        )
        self._last_emit_frame: int = -10**9
        self._ball_was_inside_rim_last_frame: bool = False

    def update(
        self,
        frame_idx: int,
        ball: Optional[BallDetection],
        actions: List[ActionObservation],
        mapped_players: List[MappedPlayer],
        possessor_track_id: Optional[int],
    ) -> List[ShotEvent]:
        """Advance one frame; return any ShotEvents that just resolved.

        Args:
            frame_idx: Absolute frame number.
            ball: Current ball detection, or None.
            actions: Detector's action observations this frame — we use
                only the entries with class_name == "rim".
            mapped_players: Used to look up the shooter's court position
                + team_id at emission time.
            possessor_track_id: Current confirmed possessor from the
                PossessionTracker, or None.

        Returns:
            List of ShotEvents emitted this frame (usually empty; never
            more than 1 per frame in practice).
        """
        # Record possession for shooter attribution later.
        self._possessor_history.append((frame_idx, possessor_track_id))

        if ball is None:
            self._ball_was_inside_rim_last_frame = False
            return []

        rims = [a for a in actions if a.class_name == "rim"]
        if not rims:
            self._ball_was_inside_rim_last_frame = False
            return []

        # Compute "ball above any rim" for the current sample. We define
        # ABOVE as: ball_cy < rim_top - small_margin AND ball is roughly
        # horizontally aligned with the rim. The horizontal check avoids
        # counting a ball halfway down the court as "above" the rim.
        bx, by = ball.center
        above_any_rim = self._above_any_rim(bx, by, rims)
        sample = _BallSample(
            frame_idx=frame_idx, cx=bx, cy=by, above_any_rim=above_any_rim,
        )
        self._ball_history.append(sample)

        # Refractory: don't fire two events in quick succession.
        if frame_idx - self._last_emit_frame < self.refractory_frames:
            self._ball_was_inside_rim_last_frame = self._ball_inside_any_rim(
                bx, by, rims,
            )
            return []

        # Did the ball just enter a rim's bbox from outside?
        inside_now = self._ball_inside_any_rim(bx, by, rims)
        just_entered = inside_now and not self._ball_was_inside_rim_last_frame
        self._ball_was_inside_rim_last_frame = inside_now
        if not just_entered:
            return []

        # Was the ball ABOVE some rim recently? Looking back within the
        # max-shot-duration window keeps us honest — a ball bouncing
        # around the floor for a long time shouldn't suddenly count
        # because it crosses a rim bbox.
        if not self._was_above_rim_recently(frame_idx):
            return []

        # Resolve shooter from the recent-possessor history.
        shooter_id, team_id, court_pos = self._attribute_shot(
            mapped_players,
        )

        event = ShotEvent(
            shooter_track_id=shooter_id if shooter_id is not None else -1,
            shot_type="geometric",
            frame_start=max(
                frame_idx - self.max_shot_duration_frames, 0,
            ),
            frame_end=frame_idx,
            made=True,
            court_x=court_pos[0] if court_pos else None,
            court_y=court_pos[1] if court_pos else None,
            team_id=team_id if team_id is not None else -1,
        )
        self._last_emit_frame = frame_idx
        return [event]

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _above_any_rim(
        bx: float, by: float, rims: List[ActionObservation],
    ) -> bool:
        for r in rims:
            x1, y1, x2, y2 = r.bbox
            # Horizontally aligned within the rim's x range plus a buffer
            # equal to one rim-width on each side.
            rw = x2 - x1
            if x1 - rw <= bx <= x2 + rw:
                # ABOVE: ball-y is less than rim-y-top (y axis points down
                # in image coords).
                if by < y1:
                    return True
        return False

    @staticmethod
    def _ball_inside_any_rim(
        bx: float, by: float, rims: List[ActionObservation],
    ) -> bool:
        for r in rims:
            x1, y1, x2, y2 = r.bbox
            if x1 <= bx <= x2 and y1 <= by <= y2:
                return True
        return False

    def _was_above_rim_recently(self, frame_idx: int) -> bool:
        cutoff = frame_idx - self.max_shot_duration_frames
        for s in reversed(self._ball_history):
            if s.frame_idx < cutoff:
                break
            if s.above_any_rim:
                return True
        return False

    def _attribute_shot(
        self,
        mapped_players: List[MappedPlayer],
    ) -> Tuple[Optional[int], Optional[int], Optional[Tuple[float, float]]]:
        """Find (shooter, team, court_pos) from recent-possessor history.

        Walk backwards through the possessor buffer; pick the first
        non-None possessor. If that track is still in mapped_players,
        use their current court position; otherwise return court_pos =
        None (the scoreboard will fall back to a 2-pointer default).
        """
        shooter_id: Optional[int] = None
        for _frame, pid in reversed(self._possessor_history):
            if pid is not None:
                shooter_id = pid
                break

        if shooter_id is None:
            return None, None, None

        for mp in mapped_players:
            if mp.track_id == shooter_id:
                return shooter_id, mp.team_id, (mp.court_x, mp.court_y)

        # Possessor track has gone out of view — keep the shooter ID but
        # can't look up court position right now.
        return shooter_id, None, None
