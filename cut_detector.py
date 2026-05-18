"""Detect hard camera cuts robust to single-frame flashes.

A camera cut is a sudden change of viewpoint — sideline switching to
baseline, live action to replay, broadcast to scoreboard graphic. The
state in the rest of the pipeline (tracker IDs, possession, homography
cache) doesn't transfer across these boundaries, so without a cut
signal the tracker tries to match new players to old IDs and produces
the 58-tracks-for-10-players problem.

Detection method
----------------
Compute a 2-D hue/saturation histogram of each frame and compare it to
the per-bin MEDIAN of the last few frames' histograms via Bhattacharyya
distance. Distance is in [0, 1] where 0 means identical and 1 means
totally different.

Why median-not-prev: a single anomalous frame (replay flash, transition
graphic, scoreboard overlay) makes the NEXT real frame look enormously
different from "the previous frame" → cut detector mis-fires → tracker
bindings reset → team classifications scrambled. The median of a small
rolling window absorbs one outlier without moving, so single-frame
flashes don't trigger; sustained scene changes still do.

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
# 0.35 catches angle-to-angle cuts (sideline → baseline) that 0.45
# missed — observed in the Knicks/Sixers clip where only 5 of an
# expected ~12 cuts fired, leaving stale tracker IDs and homography
# cached across real scene changes. Trade-off: a fast camera pan with
# a player crowding the lens can briefly cross 0.35; the refractory
# window keeps that to one cut per scene.
_DEFAULT_CUT_THRESHOLD = 0.35

# Don't fire two cuts within this many frames. Catches double-edges
# (e.g. a 2-frame black flash between scenes that would otherwise
# register as cut-out + cut-in).
_DEFAULT_REFRACTORY_FRAMES = 5

# Number of recent frame histograms to take the per-bin median of as the
# comparison baseline. 5 is the sweet spot:
#   - small enough to react to a real cut within ~5 frames (the median
#     shifts toward the new scene as the window fills with new frames),
#   - large enough that 1 outlier in 5 leaves the median dominated by
#     the 4 normal frames, so single-frame flashes don't move it.
# Below 3 we lose robustness; above ~10 we get sluggish recovery from
# legitimate cuts (the old scene keeps weighting the median).
_HIST_WINDOW = 5


class CameraCutDetector:
    """Per-frame cut detection from HSV histogram differences."""

    def __init__(
        self,
        cut_threshold: float = _DEFAULT_CUT_THRESHOLD,
        refractory_frames: int = _DEFAULT_REFRACTORY_FRAMES,
        h_bins: int = 24,
        s_bins: int = 24,
        hist_window: int = _HIST_WINDOW,
    ):
        self.cut_threshold = cut_threshold
        self.refractory_frames = refractory_frames
        self._h_bins = h_bins
        self._s_bins = s_bins
        self._hist_window = hist_window

        # Rolling buffer of recent frame histograms. Comparison baseline
        # is the per-bin median across this buffer, NOT just the last
        # frame's hist — so single-frame anomalies can't trigger a cut.
        self._recent_hists: Deque[np.ndarray] = deque(maxlen=hist_window)
        self._frames_since_last_cut: int = 10**6
        # Last few distances — exposed for debugging / tuning.
        self._distances: Deque[float] = deque(maxlen=30)

    def update(self, frame: np.ndarray) -> bool:
        """Return True iff this frame is the first frame of a new shot.

        Always returns False for the very first frame — there's no
        history to compare against.
        """
        self._frames_since_last_cut += 1
        hist = self._compute_hist(frame)

        if not self._recent_hists:
            self._recent_hists.append(hist)
            return False

        # Per-bin median across the buffer is robust to single outliers.
        # Stack into shape (N, H_BINS, S_BINS) and take element-wise
        # median over the first axis.
        stacked = np.stack(list(self._recent_hists), axis=0)
        median_hist = np.median(stacked, axis=0).astype(np.float32)

        dist = float(cv2.compareHist(
            median_hist, hist, cv2.HISTCMP_BHATTACHARYYA,
        ))
        self._distances.append(dist)

        # Always append AFTER measuring, so the current frame doesn't
        # bias its own comparison baseline. The new frame becomes part
        # of the baseline for the next frame's comparison.
        self._recent_hists.append(hist)

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
        self._recent_hists.clear()
        self._frames_since_last_cut = 10**6
        self._distances.clear()
