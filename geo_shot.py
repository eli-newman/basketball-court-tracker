"""Geometric shot detection — emit ShotEvents from ball + rim geometry alone.

The action-class shot detector in events.py depends on the player model
emitting `player-jump-shot`, `player-layup-dunk`, `ball-in-basket`, etc.
Some model versions (including basketball-player-detection-3-ycjdo/6,
which we use by default) DON'T emit those classes, so EventDetector
never fires.

This module fills the gap using ball + rim geometry alone.

State machine
-------------
A shot has two phases here:

1. **Trigger**: the ball center enters a rim's bbox having been
   *above* that rim in the recent past. This is when we start
   tracking the outcome — but we don't emit yet.

2. **Resolution**: over the next `resolution_frames` (≈0.5s at 30fps),
   we watch the ball. There are exactly two outcomes that matter:
     - The ball is observed BELOW the rim bbox → MADE. Through the
       rim and through the net, the only way for the ball to end up
       below the rim from above.
     - The ball never goes below the rim within the window → MISSED.
       Covers the front-rim brick, the back-iron bounce, the rim-roll
       that comes back up. All of those leave the ball at or above
       the rim, never below.

The previous "any rim-bbox entry = made" was wrong because the rim
bbox catches missed shots too — front-iron rejects, bounces, balls
sitting on the rim. The "did it go BELOW" check is what
distinguishes a make from anything else.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from detector import ActionObservation, BallDetection
from events import ShotEvent
from mapper import MappedPlayer


# Maximum frames between "ball above rim" and "ball at rim" for the
# trigger to count. At 30fps, 30 frames ≈ 1s — covers release through
# rim contact.
_MAX_SHOT_DURATION_FRAMES = 30

# After triggering a shot, watch this many frames for the
# made-or-missed signal (ball seen below rim → made; otherwise missed).
# 15 frames ≈ 0.5s — long enough for a made ball to drop through the
# net and reappear below; short enough that we resolve before the next
# play starts.
_RESOLUTION_FRAMES = 15

# After EMITTING a shot, suppress new triggers for this many frames so
# a single basket doesn't repeatedly fire as the ball continues through.
_REFRACTORY_FRAMES = 30

# How many recent frames to look back when attributing a shot to a
# possessor. The possession tracker may release the ball once it leaves
# the shooter's hands, so we keep a memory of recent possessors.
_POSSESSOR_LOOKBACK_FRAMES = 30

# Pixel margin BELOW the rim bbox bottom that the ball must reach to be
# called a make. Reduces sensitivity to rim-roll bounces that briefly
# poke below the bbox without actually going through the net.
_BELOW_RIM_MARGIN_PX = 4


@dataclass
class _BallSample:
    frame_idx: int
    cx: float
    cy: float
    # Whether this sample was strictly ABOVE every rim bbox we saw
    # this frame. Pre-computed because rim positions vary per frame.
    above_any_rim: bool


@dataclass
class _PendingShot:
    """A shot whose outcome (made/missed) we're still waiting to resolve."""
    trigger_frame: int             # frame where ball entered a rim bbox
    rim_bbox: Tuple[float, float, float, float]
    shooter_track_id: Optional[int]
    team_id: Optional[int]
    court_pos: Optional[Tuple[float, float]]
    shooter_player_id: Optional[int] = None  # persistent identity at trigger
    saw_ball_below_rim: bool = False


class GeometricShotDetector:
    """Emits a ShotEvent (made OR missed) after watching ball + rim geometry.

    A shot is *triggered* when the ball enters a rim bbox having been
    above the rim recently. The shot is then *resolved* over the next
    `resolution_frames`:
      - if the ball is seen below the rim during that window → MADE
      - otherwise → MISSED
    """

    def __init__(
        self,
        max_shot_duration_frames: int = _MAX_SHOT_DURATION_FRAMES,
        resolution_frames: int = _RESOLUTION_FRAMES,
        refractory_frames: int = _REFRACTORY_FRAMES,
        possessor_lookback_frames: int = _POSSESSOR_LOOKBACK_FRAMES,
        below_rim_margin_px: float = _BELOW_RIM_MARGIN_PX,
    ):
        self.max_shot_duration_frames = max_shot_duration_frames
        self.resolution_frames = resolution_frames
        self.refractory_frames = refractory_frames
        self.possessor_lookback_frames = possessor_lookback_frames
        self.below_rim_margin_px = below_rim_margin_px

        self._ball_history: Deque[_BallSample] = deque(
            maxlen=max(max_shot_duration_frames, possessor_lookback_frames) + 5,
        )
        # frame_idx → possessor_track_id (or None). Same window as history.
        self._possessor_history: Deque[Tuple[int, Optional[int]]] = deque(
            maxlen=possessor_lookback_frames + 5,
        )
        self._last_emit_frame: int = -10**9
        self._ball_was_inside_rim_last_frame: bool = False
        # The currently-pending shot, if any. Only one at a time —
        # there's no realistic scenario where two attempts overlap at
        # the same rim.
        self._pending: Optional[_PendingShot] = None

    def update(
        self,
        frame_idx: int,
        ball: Optional[BallDetection],
        actions: List[ActionObservation],
        mapped_players: List[MappedPlayer],
        possessor_track_id: Optional[int],
    ) -> List[ShotEvent]:
        """Advance one frame; return any ShotEvents that just resolved."""
        # Record possession for shooter attribution later.
        self._possessor_history.append((frame_idx, possessor_track_id))

        rims = [a for a in actions if a.class_name == "rim"]

        # Track ball position (used both for triggering and for
        # resolving an in-progress shot).
        if ball is not None and rims:
            bx, by = ball.center
            above_any_rim = self._above_any_rim(bx, by, rims)
            self._ball_history.append(_BallSample(
                frame_idx=frame_idx, cx=bx, cy=by,
                above_any_rim=above_any_rim,
            ))

        # 1) If a shot is pending, check whether the ball just went
        # BELOW its rim — that's the make signal. We do this even when
        # the ball is None or no rim is in this frame; pending state
        # persists until the resolution window expires.
        emitted: List[ShotEvent] = []
        if self._pending is not None and ball is not None:
            _, _, _, rim_y2 = self._pending.rim_bbox
            if ball.center[1] > rim_y2 + self.below_rim_margin_px:
                self._pending.saw_ball_below_rim = True

        # 2) Resolve the pending shot once its window has elapsed.
        if self._pending is not None:
            frames_elapsed = frame_idx - self._pending.trigger_frame
            if frames_elapsed >= self.resolution_frames:
                emitted.append(self._emit_pending(frame_idx))

        # 3) Possibly trigger a new shot this frame.
        # Refractory + an existing pending shot both block new triggers.
        if (
            ball is not None
            and rims
            and self._pending is None
            and frame_idx - self._last_emit_frame >= self.refractory_frames
        ):
            bx, by = ball.center
            inside_now = self._ball_inside_any_rim(bx, by, rims)
            just_entered = (
                inside_now and not self._ball_was_inside_rim_last_frame
            )
            self._ball_was_inside_rim_last_frame = inside_now

            if just_entered and self._was_above_rim_recently(frame_idx):
                # Find the rim the ball is inside; record it for resolution.
                rim_bbox = self._first_rim_containing(bx, by, rims)
                shooter_id, team_id, court_pos, player_id = self._attribute_shot(
                    mapped_players,
                )
                self._pending = _PendingShot(
                    trigger_frame=frame_idx,
                    rim_bbox=rim_bbox,
                    shooter_track_id=shooter_id,
                    team_id=team_id,
                    court_pos=court_pos,
                    shooter_player_id=player_id,
                )
        else:
            # Keep this flag honest even when we couldn't trigger,
            # so the "just_entered" edge detection works correctly
            # on the next eligible frame.
            if ball is not None and rims:
                self._ball_was_inside_rim_last_frame = (
                    self._ball_inside_any_rim(ball.center[0], ball.center[1], rims)
                )
            elif ball is None:
                self._ball_was_inside_rim_last_frame = False

        return emitted

    def _emit_pending(self, frame_idx: int) -> ShotEvent:
        """Convert the pending shot into a ShotEvent and clear pending state."""
        assert self._pending is not None
        p = self._pending
        made = p.saw_ball_below_rim
        event = ShotEvent(
            shooter_track_id=(
                p.shooter_track_id if p.shooter_track_id is not None else -1
            ),
            shot_type="geometric",
            frame_start=max(
                p.trigger_frame - self.max_shot_duration_frames, 0,
            ),
            frame_end=frame_idx,
            made=made,
            court_x=p.court_pos[0] if p.court_pos else None,
            court_y=p.court_pos[1] if p.court_pos else None,
            team_id=p.team_id if p.team_id is not None else -1,
            shooter_player_id=p.shooter_player_id,
        )
        self._pending = None
        self._last_emit_frame = frame_idx
        return event

    def reset(self):
        """Forget recent ball/possessor history. Use on a camera cut —
        the previous shot's ball trajectory has nothing to do with the
        new shot's geometry. Pending shots are dropped (no resolution
        signal across cut boundaries).
        """
        self._ball_history.clear()
        self._possessor_history.clear()
        self._ball_was_inside_rim_last_frame = False
        self._pending = None
        # Don't reset _last_emit_frame — refractory is per-clip, not per-shot.

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

    @staticmethod
    def _first_rim_containing(
        bx: float, by: float, rims: List[ActionObservation],
    ) -> Tuple[float, float, float, float]:
        """The first rim bbox containing the point. Caller guarantees one
        exists (this helper is only called after _ball_inside_any_rim).
        """
        for r in rims:
            x1, y1, x2, y2 = r.bbox
            if x1 <= bx <= x2 and y1 <= by <= y2:
                return (x1, y1, x2, y2)
        # Fallback: should not be reached.
        x1, y1, x2, y2 = rims[0].bbox
        return (x1, y1, x2, y2)

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
    ) -> Tuple[
        Optional[int], Optional[int], Optional[Tuple[float, float]], Optional[int],
    ]:
        """Find (shooter_track, team, court_pos, player_id) from recent
        possessor history.

        Walk backwards through the possessor buffer; pick the first
        non-None possessor. If that track is still in mapped_players,
        use their current court position + persistent player_id; otherwise
        return court_pos = None and player_id = None.
        """
        shooter_id: Optional[int] = None
        for _frame, pid in reversed(self._possessor_history):
            if pid is not None:
                shooter_id = pid
                break

        if shooter_id is None:
            return None, None, None, None

        for mp in mapped_players:
            if mp.track_id == shooter_id:
                return (
                    shooter_id,
                    mp.team_id,
                    (mp.court_x, mp.court_y),
                    mp.player_id,
                )

        # Possessor track has gone out of view — keep the shooter ID but
        # can't look up court position or persistent identity right now.
        return shooter_id, None, None, None
