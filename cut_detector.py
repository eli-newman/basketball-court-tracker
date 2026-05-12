"""Detect hard camera cuts between consecutive frames.

A camera cut is a sudden change of viewpoint — sideline switching to
baseline, live action to replay, broadcast to scoreboard graphic. The
state in the rest of the pipeline (tracker IDs, possession, homography
cache) doesn't transfer across these boundaries, so without a cut
signal the tracker tries to match new players to old IDs and produces
the 58-tracks-for-10-players problem.

Detection method
----------------
Compute a 2-D hue/saturation histogram of each frame, compare to the
previous frame's histogram by Bhattacharyya distance. Distance is in
[0, 1] where 0 means identical and 1 means totally different. Hard
cuts produce a sudden jump well above what continuous motion does.

We pick the histogram over hue+saturation instead of full BGR for two
reasons:
  - Court color (the warm tan that dominates basketball broadcasts)
    is the same across all camera positions within a game, so the
    hue distribution shifts modestly during a pan and a LOT on a cut
    to a different scene.
  - Brightness changes (lighting, exposure) within a continuous shot
    don't affect HSV's H+S channels much. Using V would make pans
    look like cuts when an overhead light catches the lens.

A short refractory window prevents two cuts firing in rapid succession
(e.g. a fade-to-black that the histogram sees as cut-out + cut-in).
"""

from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np


# Bhattacharyya distance between consecutive HSV-HS histograms above
# which we declare a cut. Empirical from broadcast clips:
#   - smooth pan / zoom:   ~0.05 – 0.20
#   - player crowd-shot:   ~0.25 – 0.40
#   - hard cut:            ~0.50 – 0.90
# 0.45 sits cleanly between "lots of camera motion" and "different scene."
_DEFAULT_CUT_THRESHOLD = 0.45

# Don't fire two cuts within this many frames. Catches double-edges
# (e.g. a 2-frame black flash between scenes that would otherwise
# register as cut-out + cut-in).
_DEFAULT_REFRACTORY_FRAMES = 5


class CameraCutDetector:
    """Per-frame cut detection from HSV histogram differences."""

    def __init__(
        self,
        cut_threshold: float = _DEFAULT_CUT_THRESHOLD,
        refractory_frames: int = _DEFAULT_REFRACTORY_FRAMES,
        h_bins: int = 24,
        s_bins: int = 24,
    ):
        self.cut_threshold = cut_threshold
        self.refractory_frames = refractory_frames
        self._h_bins = h_bins
        self._s_bins = s_bins

        self._prev_hist: Optional[np.ndarray] = None
        self._frames_since_last_cut: int = 10**6
        # Last few distances — exposed for debugging / tuning.
        self._distances: Deque[float] = deque(maxlen=30)

    def update(self, frame: np.ndarray) -> bool:
        """Return True iff this frame is the first frame of a new shot.

        Always returns False for the very first frame — there's no
        previous frame to compare against.
        """
        self._frames_since_last_cut += 1
        hist = self._compute_hist(frame)

        if self._prev_hist is None:
            self._prev_hist = hist
            return False

        dist = float(cv2.compareHist(
            self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA,
        ))
        self._distances.append(dist)
        self._prev_hist = hist

        if dist < self.cut_threshold:
            return False
        if self._frames_since_last_cut < self.refractory_frames:
            return False

        self._frames_since_last_cut = 0
        return True

    def _compute_hist(self, frame: np.ndarray) -> np.ndarray:
        """Normalized hue/saturation histogram of the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self._h_bins, self._s_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    @property
    def last_distance(self) -> Optional[float]:
        if not self._distances:
            return None
        return self._distances[-1]

    @property
    def distances(self) -> list:
        """Recent per-frame distances — handy for tuning the threshold."""
        return list(self._distances)

    def reset(self):
        """Forget previous-frame state. Use when the input source restarts."""
        self._prev_hist = None
        self._frames_since_last_cut = 10**6
        self._distances.clear()
