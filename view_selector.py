"""Active-half detection from court keypoints.

The Roboflow court-keypoint model returns 33 points labeled by class_id 0-32.
Indices 0-14 are on the LEFT half of the court (left baseline → left side
extended). Indices 15-17 are on the half-court line. Indices 18-32 are on
the RIGHT half. Whichever side has higher mean keypoint confidence is the
side the broadcast camera is currently looking at.

This avoids guessing from player center-of-mass, which gets confused
during transition (5 players on each half).
"""

from collections import Counter, deque
from typing import Deque, List, Optional, Union

from homography import CourtKeypoint


_LEFT_INDICES = set(range(0, 15))    # 0..14
_RIGHT_INDICES = set(range(18, 33))  # 18..32
# 15..17 are half-court markers (bridge points), excluded from the vote.


class ActiveHalfSelector:
    """Picks the active half ("left" / "right") with hysteresis.

    Algorithm:
      1. For the current frame, sum keypoint confidence on each side.
      2. Decide "left" / "right" / None (ambiguous) by ratio threshold.
      3. Push the decision into a rolling history.
      4. Return the majority of recent non-None decisions in the window.

    Hysteresis prevents flicker when the camera briefly catches the other rim,
    or during transitions where neither side dominates.
    """

    def __init__(self, ratio_threshold: float = 1.5, history_frames: int = 15):
        """
        Args:
            ratio_threshold: A side wins this frame only if its confidence sum
                is ≥ ratio_threshold × the other side's sum.
            history_frames: Window of recent frames used for the majority vote.
        """
        self.ratio_threshold = ratio_threshold
        self.history: Deque[Optional[str]] = deque(maxlen=history_frames)
        self._last_decision: Optional[str] = None

    def update(self, keypoints: List[CourtKeypoint]) -> Optional[str]:
        """Process a frame's keypoints and return the active half.

        Returns "left", "right", or None (when there's no signal yet).
        """
        per_frame = self._per_frame_decision(keypoints)
        self.history.append(per_frame)

        recent = [d for d in self.history if d is not None]
        if not recent:
            return self._last_decision  # nothing yet — keep prior
        winner, _ = Counter(recent).most_common(1)[0]
        self._last_decision = winner
        return winner

    def _per_frame_decision(self, keypoints: List[CourtKeypoint]) -> Optional[str]:
        left = 0.0
        right = 0.0
        for kp in keypoints:
            idx = _to_index(kp.name)
            if idx is None:
                continue
            if idx in _LEFT_INDICES:
                left += kp.confidence
            elif idx in _RIGHT_INDICES:
                right += kp.confidence

        if left == 0 and right == 0:
            return None
        if right >= left * self.ratio_threshold:
            return "right"
        if left >= right * self.ratio_threshold:
            return "left"
        return None  # ambiguous — neither side dominates this frame

    def reset(self):
        """Clear hysteresis (e.g., on detected camera cut)."""
        self.history.clear()
        self._last_decision = None


def _to_index(name: Union[int, str]) -> Optional[int]:
    """Coerce a CourtKeypoint.name to an integer class_id if possible."""
    if isinstance(name, int):
        return name
    if isinstance(name, str):
        try:
            return int(name)
        except ValueError:
            return None
    return None
