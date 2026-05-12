"""Ball possession: which tracked player currently has the ball.

The signal we use is simple — the ball center's distance to each player's
bbox. The closest player gets credit IF the distance is below a threshold;
otherwise the ball is "loose" (mid-pass, mid-shot, on the floor between
players, etc.).

Two failure modes the raw nearest-player signal has, and how we fix them:

1. **Pass / shot jitter.** During a pass, the ball is between two players,
   and the "nearest" can flip every frame. Without smoothing the
   possession label flickers.

   Fix: a track only becomes the *confirmed* possessor after K consecutive
   frames as the nearest candidate (`confirm_at`). Below that threshold we
   report the *previous* confirmed possessor or None.

2. **Ball clearly thrown.** When the ball is in the air mid-shot, every
   player is far from it — we should report "no one has the ball" rather
   than picking the nearest by default.

   Fix: a hard distance threshold (`max_distance_px`). Beyond it we report
   the candidate as None for this frame's vote.

Track-id-stable: the player ID we return is the same persistent track_id
the rest of the pipeline uses, so downstream code can join it against
mapped players, team_id, jersey number, etc.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from detector import BallDetection, PlayerDetection


@dataclass
class PossessionResult:
    """Per-frame possession output.

    Attributes:
        possessor_track_id: track_id of the confirmed possessor, or None.
        candidate_track_id: track_id of the *raw* nearest player this frame
            (before temporal smoothing). Useful for debug overlays and tests.
        distance_px: pixel distance from ball center to the candidate's
            bbox. None if no ball was detected this frame.
    """
    possessor_track_id: Optional[int]
    candidate_track_id: Optional[int]
    distance_px: Optional[float]


class PossessionTracker:
    """Maps (ball, players) → which track currently has the ball."""

    def __init__(
        self,
        max_distance_px: float = 120.0,
        confirm_at: int = 3,
        history_len: int = 6,
        release_after_missing: int = 8,
    ):
        """
        Args:
            max_distance_px: A player must be within this many pixels of the
                ball center for them to be a candidate. Beyond it the ball
                is "loose" — no one has it. 120px is roughly arm's reach in
                a 1080p broadcast frame.
            confirm_at: Number of consecutive frames a candidate must be
                nearest before we *commit* possession to them. Higher =
                more stable but slower to react to a real possession change
                (steal, rebound). 3 is a good default at ~30 fps.
            history_len: Size of the rolling candidate buffer. Must be ≥
                confirm_at for the vote to ever succeed.
            release_after_missing: How many frames the ball can be undetected
                (or every player is too far) before we drop the current
                possessor and report None. Smooths over single-frame
                detector misses without holding stale possession forever.
        """
        if history_len < confirm_at:
            raise ValueError(
                f"history_len ({history_len}) must be ≥ confirm_at ({confirm_at})"
            )
        self.max_distance_px = max_distance_px
        self.confirm_at = confirm_at
        self.release_after_missing = release_after_missing

        self._candidates: Deque[Optional[int]] = deque(maxlen=history_len)
        self._current_possessor: Optional[int] = None
        self._missing_frames = 0

    def update(
        self,
        ball: Optional[BallDetection],
        players: List[PlayerDetection],
    ) -> PossessionResult:
        """Advance one frame and return the current possession state."""
        candidate, distance = self._nearest_player(ball, players)
        self._candidates.append(candidate)

        if candidate is None:
            # No candidate this frame (no ball, or all players too far).
            self._missing_frames += 1
            if self._missing_frames >= self.release_after_missing:
                self._current_possessor = None
            return PossessionResult(
                possessor_track_id=self._current_possessor,
                candidate_track_id=None,
                distance_px=distance,
            )

        # Found a candidate → reset the miss counter.
        self._missing_frames = 0

        # Check if the last `confirm_at` candidates are all this same track.
        recent = list(self._candidates)[-self.confirm_at:]
        if (
            len(recent) >= self.confirm_at
            and all(tid == candidate for tid in recent)
        ):
            self._current_possessor = candidate

        return PossessionResult(
            possessor_track_id=self._current_possessor,
            candidate_track_id=candidate,
            distance_px=distance,
        )

    def _nearest_player(
        self,
        ball: Optional[BallDetection],
        players: List[PlayerDetection],
    ) -> Tuple[Optional[int], Optional[float]]:
        """Return (track_id, distance) of the player nearest the ball.

        Returns (None, None) if no ball, or (None, dist) if every player is
        beyond `max_distance_px` (so the caller can log the gap if needed).
        """
        if ball is None or not players:
            return None, None

        bx, by = ball.center
        best_tid: Optional[int] = None
        best_dist: Optional[float] = None

        for p in players:
            if p.track_id < 0:
                # Untracked detections don't get to claim possession — the
                # whole downstream pipeline keys on track_id.
                continue
            dist = _bbox_to_point_distance(p.bbox, bx, by)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_tid = p.track_id

        if best_dist is None:
            return None, None
        if best_dist > self.max_distance_px:
            return None, best_dist
        return best_tid, best_dist

    @property
    def current_possessor(self) -> Optional[int]:
        return self._current_possessor


def _bbox_to_point_distance(bbox: tuple, px: float, py: float) -> float:
    """Euclidean distance from point (px, py) to nearest edge of bbox.

    Point INSIDE the bbox returns 0. This is more forgiving than
    center-to-center, which would unfairly penalize tall players whose
    bbox extends well above the ball.
    """
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return float(np.hypot(dx, dy))
