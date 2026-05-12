"""Team classification by jersey color, aggregated per track.

Per-frame classification has two failure modes that show up in real footage:

1. **Occlusion contamination.** When a defender stands tight on an offensive
   player, their bboxes overlap. The chest crop for player A then includes
   pixels from player B's jersey — both end up with the same median color,
   both get assigned the same team.

2. **Per-frame jitter.** A single bad frame (motion blur, weird angle,
   referee crossing through) flips a player's team_id back and forth.

This module fixes both by:
  - Masking out pixels that fall inside *other* players' bboxes when
    extracting a player's chest color (occlusion-aware extraction).
  - Accumulating chest color samples per `track_id` over time, and
    fitting KMeans on **one median color per track**, not per-frame.
  - Returning a stable team_id per track that's recomputed on every refit
    rather than on every frame.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from detector import PlayerDetection


# Sentinel for "no team yet" — used until enough data is collected to fit.
UNKNOWN_TEAM = -1


class TeamClassifier:
    """Classifies players into teams based on chest color aggregated per track."""

    def __init__(
        self,
        n_teams: int = 2,
        warmup_frames: int = 15,
        min_saturation: int = 60,
        samples_per_track: int = 30,
        refit_every: int = 10,
        min_samples_to_classify: int = 3,
    ):
        """
        Args:
            n_teams: Number of teams to cluster (2 = just teams, 3 = teams + refs).
            warmup_frames: Don't fit KMeans until at least this many frames have
                been seen. Bigger = more stable initial fit, slower to label.
            min_saturation: HSV-S threshold for "team color" pixels. Drops
                washed-out whites/grays so the team accent dominates the median.
            samples_per_track: Rolling window of per-frame color samples kept
                for each tracked player. Old samples drop off as new ones land.
            refit_every: Refit KMeans (re-cluster all tracks) every N frames
                once warmed up. Keeps assignments stable but adaptive to new
                tracks that appear later.
            min_samples_to_classify: A track needs at least this many color
                samples before it's eligible to be clustered.
        """
        self.n_teams = n_teams
        self._kmeans: Optional[KMeans] = None
        self._calibrated = False
        self._warmup_frames = warmup_frames
        self._min_saturation = min_saturation
        self._samples_per_track = samples_per_track
        self._refit_every = refit_every
        self._min_samples_to_classify = min_samples_to_classify

        # Per-track state
        self._track_samples: Dict[int, Deque[np.ndarray]] = {}
        self._track_assignments: Dict[int, int] = {}  # track_id -> team_id

        # Bookkeeping
        self._frame_count = 0
        self._frames_since_refit = 0

    def classify(
        self,
        frame: np.ndarray,
        players: List[PlayerDetection],
    ) -> List[int]:
        """Extract per-player chest color, accumulate, return current team_ids.

        Args:
            frame: BGR frame.
            players: tracked players (track_id should be set; -1 untracked is
                tolerated but those players are skipped from the aggregator).

        Returns:
            team_id per player in the same order as the input. -1 until the
            track has enough samples and KMeans has been fit.
        """
        if not players:
            return []
        self._frame_count += 1

        # 1) Collect this frame's chest colors, masking each player's other
        #    overlapping bboxes out so defender/offensive-player overlap can't
        #    contaminate the sample.
        all_bboxes = [p.bbox for p in players]
        for i, player in enumerate(players):
            if player.track_id < 0:
                continue
            others = all_bboxes[:i] + all_bboxes[i + 1:]
            color = self._extract_jersey_color(frame, player, others)
            if color is None:
                continue
            buf = self._track_samples.setdefault(
                player.track_id, deque(maxlen=self._samples_per_track),
            )
            buf.append(color)

        # 2) Refit KMeans periodically once warmed up.
        self._frames_since_refit += 1
        if (
            self._frame_count >= self._warmup_frames
            and self._frames_since_refit >= self._refit_every
        ):
            self._refit()
            self._frames_since_refit = 0

        # 3) Return current assignments in input order.
        return [
            self._track_assignments.get(p.track_id, UNKNOWN_TEAM)
            for p in players
        ]

    def _extract_jersey_color(
        self,
        frame: np.ndarray,
        player: PlayerDetection,
        other_bboxes: List[tuple],
    ) -> Optional[np.ndarray]:
        """Extract median HSV from the player's chest, excluding pixels that
        fall inside any other player's bbox (occlusion-aware).
        """
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        fh, fw = frame.shape[:2]
        x1 = max(0, min(x1, fw - 1))
        x2 = max(0, min(x2, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        y2 = max(0, min(y2, fh - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        # Number ROI: 30-58% vertical (where the JERSEY NUMBER sits — that's
        # the largest patch of true team color), center 50% horizontal.
        # Was 20-45% before; that landed on the upper chest, ABOVE the number,
        # so we were sampling mostly white for both teams in a white-on-white
        # matchup.
        bh = y2 - y1
        bw = x2 - x1
        cy1 = y1 + int(bh * 0.30)
        cy2 = y1 + int(bh * 0.58)
        cx1 = x1 + int(bw * 0.25)
        cx2 = x2 - int(bw * 0.25)
        if cy2 <= cy1 or cx2 <= cx1:
            cy1, cy2, cx1, cx2 = y1, y2, x1, x2

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Strict saturation gate (S≥100) — only TRUE accent-colored pixels.
        # White/gray jersey body (S near 0) is excluded; only number / trim
        # pixels survive. This is the only way to separate teams whose
        # jersey bodies are both mostly white.
        mask = (s >= self._min_saturation) & (v > 50) & (v < 240)

        # Occlusion mask: drop pixels inside any other player's bbox.
        if other_bboxes:
            H, W = crop.shape[:2]
            occlusion = np.ones((H, W), dtype=bool)
            for ox1, oy1, ox2, oy2 in other_bboxes:
                ix1 = max(0, int(ox1) - cx1)
                iy1 = max(0, int(oy1) - cy1)
                ix2 = min(W, int(ox2) - cx1)
                iy2 = min(H, int(oy2) - cy1)
                if ix2 > ix1 and iy2 > iy1:
                    occlusion[iy1:iy2, ix1:ix2] = False
            mask = mask & occlusion

        if mask.sum() < 10:
            # Try a wider region (full chest) with the same strict S gate.
            wider = frame[max(0, y1 + int(bh * 0.15)):max(0, y1 + int(bh * 0.65)),
                          x1:x2]
            if wider.size == 0:
                return None
            hsv2 = cv2.cvtColor(wider, cv2.COLOR_BGR2HSV)
            s2 = hsv2[:, :, 1]
            v2 = hsv2[:, :, 2]
            mask2 = (s2 >= self._min_saturation) & (v2 > 50) & (v2 < 240)
            if mask2.sum() < 10:
                return None
            hue_pixels = hsv2[mask2, 0]
        else:
            hue_pixels = hsv[mask, 0]

        # Cluster on HUE only (circular). OpenCV hue is 0-180; map to angle
        # then take mean of unit vectors (handles the 180/0 wrap).
        angles = hue_pixels.astype(np.float32) * (np.pi / 90.0)  # 0..180 → 0..2π
        sin_h = float(np.mean(np.sin(angles)))
        cos_h = float(np.mean(np.cos(angles)))
        return np.array([sin_h, cos_h], dtype=np.float32)

    def _refit(self):
        """Recluster tracks: one median sample per track, fit, assign."""
        # Build "track_id -> median color" using only tracks with enough samples.
        track_ids: List[int] = []
        track_colors: List[np.ndarray] = []
        for tid, samples in self._track_samples.items():
            if len(samples) < self._min_samples_to_classify:
                continue
            track_ids.append(tid)
            track_colors.append(np.median(np.stack(samples, axis=0), axis=0))

        if len(track_colors) < self.n_teams:
            return  # not enough distinct tracks yet to cluster

        X = np.stack(track_colors, axis=0)
        kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
        kmeans.fit(X)
        labels = kmeans.predict(X)

        self._kmeans = kmeans
        self._track_assignments = {
            int(tid): int(label) for tid, label in zip(track_ids, labels)
        }
        self._calibrated = True

        # Debug log — cluster centers are (sin h, cos h); reverse to hue degrees
        centers = kmeans.cluster_centers_
        for i, (sin_h, cos_h) in enumerate(centers):
            count = sum(1 for v in self._track_assignments.values() if v == i)
            angle = np.degrees(np.arctan2(sin_h, cos_h))
            if angle < 0:
                angle += 360
            hue_deg = angle / 2.0  # back to OpenCV's 0..180
            print(
                f"  Team {i}: hue≈{hue_deg:.0f}° (sinH={sin_h:.2f}, "
                f"cosH={cos_h:.2f})  [{count} tracks]"
            )

    def refit(self):
        """Public force-refit (e.g., after a detected scene change)."""
        self._refit()
        self._frames_since_refit = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def team_colors_bgr(self) -> List[Tuple[int, int, int]]:
        """Cluster centers in BGR for visualization.

        Cluster centers are (sin h, cos h); we recover hue degrees, set
        saturation and value to fixed mid-high values for visibility.
        """
        if not self._calibrated or self._kmeans is None:
            return [(200, 200, 200)] * self.n_teams
        out: List[Tuple[int, int, int]] = []
        for sin_h, cos_h in self._kmeans.cluster_centers_:
            angle = np.degrees(np.arctan2(sin_h, cos_h))
            if angle < 0:
                angle += 360
            hue = int(angle / 2.0)  # OpenCV hue is 0-180
            hsv_pixel = np.uint8([[[hue, 220, 220]]])
            bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            out.append((int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])))
        return out
