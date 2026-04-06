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

    def __init__(self, n_teams: int = 2):
        """
        Args:
            n_teams: Number of teams to cluster (2 = just teams, 3 = teams + refs).
        """
        self.n_teams = n_teams
        self._kmeans: Optional[KMeans] = None
        self._calibrated = False
        self._color_samples: List[np.ndarray] = []
        self._warmup_frames = 5  # collect samples before first fit
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

        Crops the upper-middle portion of the bbox (torso area) to avoid
        shorts, shoes, and the court floor.
        """
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return np.array([128, 128, 128], dtype=np.float32)

        # Crop the torso: top 30-70% of bbox height, middle 20-80% width
        bh = y2 - y1
        bw = x2 - x1
        torso_y1 = y1 + int(bh * 0.2)
        torso_y2 = y1 + int(bh * 0.6)
        torso_x1 = x1 + int(bw * 0.2)
        torso_x2 = x2 - int(bw * 0.2)

        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        if crop.size == 0:
            return np.array([128, 128, 128], dtype=np.float32)

        # Convert to HSV for better color discrimination
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Filter out very dark pixels (shadows, dark court) and very bright (highlights)
        mask = (hsv[:, :, 2] > 40) & (hsv[:, :, 2] < 240)
        if mask.sum() < 10:
            # Not enough valid pixels, use full crop
            mean_color = hsv.reshape(-1, 3).mean(axis=0)
        else:
            mean_color = hsv[mask].mean(axis=0)

        return mean_color.astype(np.float32)

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
