"""Team classification by jersey color using KMeans clustering.

Extracts the dominant jersey color from each player's bounding box crop,
then clusters all players into 2-3 groups (team A, team B, referees).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from detector import PlayerDetection


class TeamClassifier:
    """Classifies players into teams based on jersey color."""

    def __init__(self, n_teams: int = 2, warmup_frames: int = 15, min_saturation: int = 60):
        """
        Args:
            n_teams: Number of teams to cluster (2 = just teams, 3 = teams + refs).
            warmup_frames: Frames of samples to collect before fitting KMeans.
                Larger = more stable initial fit at the cost of late team_id labels.
            min_saturation: HSV saturation threshold for "this is a team color"
                pixels. Drops washed-out whites/grays in the torso crop, so
                Knicks blue and 76ers red are not diluted by jersey trim/white.
        """
        self.n_teams = n_teams
        self._kmeans: Optional[KMeans] = None
        self._calibrated = False
        self._color_samples: List[np.ndarray] = []
        self._warmup_frames = warmup_frames
        self._min_saturation = min_saturation
        self._frame_count = 0

    def classify(
        self,
        frame: np.ndarray,
        players: List[PlayerDetection],
    ) -> List[int]:
        """Assign team IDs to each player based on jersey color.

        Args:
            frame: Current video frame (BGR).
            players: Detected players with bounding boxes.

        Returns:
            List of team_ids (0, 1, or 2) in same order as players.
            Returns [-1, -1, ...] if not enough data to classify yet.
        """
        if not players:
            return []

        # Extract jersey colors for all players
        colors = [self._extract_jersey_color(frame, p) for p in players]
        color_array = np.array(colors)

        self._frame_count += 1

        # Warmup phase: collect samples before fitting
        if not self._calibrated:
            self._color_samples.append(color_array)
            if self._frame_count >= self._warmup_frames:
                self._fit_kmeans()
            else:
                return [-1] * len(players)

        # Predict team assignments
        return self._kmeans.predict(color_array).tolist()

    def _extract_jersey_color(
        self, frame: np.ndarray, player: PlayerDetection
    ) -> np.ndarray:
        """Extract the dominant jersey color from a player's bounding box.

        Crops the jersey chest region (upper torso, between the chin and the
        waistband, center horizontal strip — no shoulders/arms either since
        sleeve color often differs from chest color). Then computes the median
        HSV across "team color" pixels — those with saturation above
        `min_saturation`, so white/gray jersey areas don't drown out the team
        accent. Median (not mean) is robust to skin tone leakage.
        """
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 128], dtype=np.float32)

        # Jersey chest ROI: 20-45% vertical (below chin, above waistband),
        # center 50% horizontal (skip sleeves & background).
        bh = y2 - y1
        bw = x2 - x1
        cy1 = y1 + int(bh * 0.20)
        cy2 = y1 + int(bh * 0.45)
        cx1 = x1 + int(bw * 0.25)
        cx2 = x2 - int(bw * 0.25)
        if cy2 <= cy1 or cx2 <= cx1:
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.array([0, 0, 128], dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Keep saturated, mid-brightness pixels. These are the team-colored
        # pixels (jersey accent / numbers), not skin, shadow, or white trim.
        mask = (s >= self._min_saturation) & (v > 50) & (v < 240)

        if mask.sum() >= 10:
            pixels = hsv[mask]
        else:
            # Fall back to full crop (player wearing nearly-white jersey)
            pixels = hsv.reshape(-1, 3)

        # Median is robust against the few skin/court pixels that slip in.
        color = np.median(pixels, axis=0)
        return color.astype(np.float32)

    def _fit_kmeans(self):
        """Fit KMeans on collected color samples."""
        all_colors = np.vstack(self._color_samples)

        if len(all_colors) < self.n_teams:
            return

        self._kmeans = KMeans(
            n_clusters=self.n_teams,
            n_init=10,
            random_state=42,
        )
        self._kmeans.fit(all_colors)
        self._calibrated = True

        # Log cluster centers for debugging
        centers = self._kmeans.cluster_centers_
        for i, center in enumerate(centers):
            h, s, v = center
            print(f"  Team {i}: HSV=({h:.0f}, {s:.0f}, {v:.0f})")

    def refit(self):
        """Force re-fitting from accumulated samples (e.g., after camera cut)."""
        if self._color_samples:
            self._fit_kmeans()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def team_colors_bgr(self) -> List[Tuple[int, int, int]]:
        """Return cluster center colors in BGR for visualization."""
        if not self._calibrated:
            return [(200, 200, 200)] * self.n_teams

        bgr_colors = []
        for center in self._kmeans.cluster_centers_:
            hsv_pixel = np.uint8([[center]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            b, g, r = int(bgr_pixel[0, 0, 0]), int(bgr_pixel[0, 0, 1]), int(bgr_pixel[0, 0, 2])
            bgr_colors.append((b, g, r))
        return bgr_colors
