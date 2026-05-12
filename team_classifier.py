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

        # Chest ROI: 20-45% vertical (below chin, above waistband),
        # center 50% horizontal (skip sleeves).
        bh = y2 - y1
        bw = x2 - x1
        cy1 = y1 + int(bh * 0.20)
        cy2 = y1 + int(bh * 0.45)
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

        # Saturated, mid-brightness pixels only — team accent, not white trim.
        mask = (s >= self._min_saturation) & (v > 50) & (v < 240)

        # Occlusion mask: any pixel inside another player's bbox is excluded.
        # This is the main fix for "defender's color leaks into offensive
        # player's chest sample because their bboxes overlap."
        if other_bboxes:
            H, W = crop.shape[:2]
            occlusion = np.ones((H, W), dtype=bool)
            for ox1, oy1, ox2, oy2 in other_bboxes:
                # Intersect other bbox with our crop region (in crop-local coords).
                ix1 = max(0, int(ox1) - cx1)
                iy1 = max(0, int(oy1) - cy1)
                ix2 = min(W, int(ox2) - cx1)
                iy2 = min(H, int(oy2) - cy1)
                if ix2 > ix1 and iy2 > iy1:
                    occlusion[iy1:iy2, ix1:ix2] = False
            mask = mask & occlusion

        if mask.sum() < 10:
            # Fall back to saturation-only mask (lose occlusion guard, but
            # better than zero pixels — usually means heavy overlap).
            fallback = (s >= self._min_saturation) & (v > 50) & (v < 240)
            if fallback.sum() < 10:
                return None
            pixels = hsv[fallback]
        else:
            pixels = hsv[mask]

        return np.median(pixels, axis=0).astype(np.float32)

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

        # Debug log
        centers = kmeans.cluster_centers_
        for i, center in enumerate(centers):
            count = sum(1 for v in self._track_assignments.values() if v == i)
            h, s, v = center
            print(f"  Team {i}: HSV=({h:.0f}, {s:.0f}, {v:.0f})  [{count} tracks]")

    def refit(self):
        """Public force-refit (e.g., after a detected scene change)."""
        self._refit()
        self._frames_since_refit = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def team_colors_bgr(self) -> List[Tuple[int, int, int]]:
        """Cluster centers in BGR for visualization."""
        if not self._calibrated or self._kmeans is None:
            return [(200, 200, 200)] * self.n_teams
        out: List[Tuple[int, int, int]] = []
        for center in self._kmeans.cluster_centers_:
            hsv_pixel = np.uint8([[center]])
            bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            out.append((int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])))
        return out
