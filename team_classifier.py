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

        # ── Color-bucket proportions ────────────────────────────────────────
        # Instead of trying to summarize each player with a single hue/V, we
        # measure the **proportion** of chest pixels in each characteristic
        # team-color bucket. NBA teams have a few canonical accent colors;
        # different teams differ in WHICH BUCKETS they have non-zero
        # proportions:
        #
        #   - ORANGE  H 10-25   (Knicks numbers/trim)
        #   - RED     H 0-5 ∪ 170-180  (76ers, Bulls, Heat, Hawks)
        #   - BLUE    H 100-130 (Knicks body, 76ers trim, Mavs, Pistons)
        #   - GREEN   H 50-80   (Celtics, Bucks)
        #   - PURPLE  H 130-150 (Lakers, Kings)
        #   - YELLOW  H 25-35   (Pacers, Jazz, Lakers home)
        #
        # And we add **dark_body** — the fraction of pixels that are low-V
        # (saturated dark colors like Knicks navy blue body). This is the
        # one feature that uniquely separates teams in white from teams in
        # dark colors regardless of accent.
        h = hsv[:, :, 0]
        sat_mask = (s >= self._min_saturation) & (v > 50) & (v < 240)
        total_sat = int(sat_mask.sum()) + 1  # +1 to avoid /0

        def frac(lo, hi):
            return float(((h >= lo) & (h <= hi) & sat_mask).sum()) / total_sat

        orange = frac(10, 25)
        red_low = frac(0, 5)
        red_high = frac(170, 180)
        red = red_low + red_high
        blue = frac(100, 130)
        green = frac(50, 80)
        purple = frac(130, 150)
        yellow = frac(25, 35)

        # dark_body: heavily-saturated dark pixels (jersey body of teams
        # wearing dark colors). Knicks navy ≈ S>120, V<130.
        dark_body = float(((s >= 120) & (v < 130)).sum()) / (s.size + 1)

        return np.array(
            [orange, red, blue, green, purple, yellow, dark_body],
            dtype=np.float32,
        )

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

        # Debug log: show the strongest color signature in each cluster center
        names = ["orange", "red", "blue", "green", "purple", "yellow", "dark_body"]
        centers = kmeans.cluster_centers_
        for i, center in enumerate(centers):
            count = sum(1 for v in self._track_assignments.values() if v == i)
            top = sorted(enumerate(center), key=lambda x: -x[1])[:3]
            sig = " ".join(f"{names[j]}={frac:.2f}" for j, frac in top if frac > 0.02)
            print(f"  Team {i}: {sig}  [{count} tracks]")

    def refit(self):
        """Public force-refit (e.g., after a detected scene change)."""
        self._refit()
        self._frames_since_refit = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # Approx BGR for each color bucket (just for visualization)
    _BUCKET_BGR = [
        (0, 130, 255),    # orange (Knicks)
        (40, 40, 220),    # red (76ers, Bulls, Heat)
        (220, 100, 30),   # blue (Knicks body, Pistons, Mavs)
        (40, 180, 40),    # green (Celtics, Bucks)
        (200, 40, 160),   # purple (Lakers, Kings)
        (0, 230, 230),    # yellow (Pacers, Lakers home)
        (60, 30, 30),     # dark body (Knicks navy)
    ]

    @property
    def team_colors_bgr(self) -> List[Tuple[int, int, int]]:
        """Cluster centers as BGR — use the dominant color bucket per team."""
        if not self._calibrated or self._kmeans is None:
            return [(200, 200, 200)] * self.n_teams
        out: List[Tuple[int, int, int]] = []
        for center in self._kmeans.cluster_centers_:
            top_bucket = int(np.argmax(center))
            out.append(self._BUCKET_BGR[top_bucket])
        return out
