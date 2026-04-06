"""Per-frame homography computation for court coordinate mapping.

Computes a 3x3 transformation matrix from detected court keypoints,
mapping pixel coordinates to real-world court coordinates in feet.
Handles fallback caching when keypoints are insufficient.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from court import COURT_LENGTH, COURT_WIDTH, get_court_point


@dataclass
class CourtKeypoint:
    """A detected court keypoint from the Roboflow model."""
    name: str           # Roboflow class name (e.g., "top-left")
    pixel_x: float      # x position in the frame (pixels)
    pixel_y: float      # y position in the frame (pixels)
    confidence: float


class HomographyEngine:
    """Computes and caches per-frame homography matrices."""

    def __init__(
        self,
        min_keypoints: int = 4,
        fallback_frames: int = 30,
        ransac_threshold: float = 5.0,
        max_reproj_error: float = 10.0,
        court_margin: float = 5.0,
    ):
        self.min_keypoints = min_keypoints
        self.fallback_frames = fallback_frames
        self.ransac_threshold = ransac_threshold
        self.max_reproj_error = max_reproj_error
        self.court_margin = court_margin

        self._last_good_H: Optional[np.ndarray] = None
        self._frames_since_good: int = 0

    def compute(self, keypoints: List[CourtKeypoint]) -> Optional[np.ndarray]:
        """Compute homography from detected keypoints.

        Args:
            keypoints: List of detected court keypoints with pixel positions.

        Returns:
            3x3 homography matrix (pixel → court feet), or None if unavailable.
        """
        # Build point correspondences
        src_pts = []  # pixel coordinates
        dst_pts = []  # court coordinates (feet)

        for kp in keypoints:
            court_pt = get_court_point(kp.name)
            if court_pt is None:
                continue
            src_pts.append([kp.pixel_x, kp.pixel_y])
            dst_pts.append([court_pt[0], court_pt[1]])

        if len(src_pts) < self.min_keypoints:
            return self._fallback()

        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)

        # Compute homography with RANSAC
        H, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, self.ransac_threshold
        )

        if H is None:
            return self._fallback()

        # Validate
        if not self._validate(H, src, dst, inlier_mask):
            return self._fallback()

        # Good homography — cache it
        self._last_good_H = H.copy()
        self._frames_since_good = 0
        return H

    def transform_point(
        self, H: np.ndarray, pixel_x: float, pixel_y: float
    ) -> Optional[Tuple[float, float]]:
        """Transform a single pixel coordinate to court coordinates.

        Args:
            H: 3x3 homography matrix.
            pixel_x, pixel_y: Pixel coordinates in the frame.

        Returns:
            (x_ft, y_ft) court coordinates, or None if out of bounds.
        """
        pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, H)
        x_ft = float(transformed[0, 0, 0])
        y_ft = float(transformed[0, 0, 1])

        # Bounds check with margin
        if (
            x_ft < -self.court_margin
            or x_ft > COURT_LENGTH + self.court_margin
            or y_ft < -self.court_margin
            or y_ft > COURT_WIDTH + self.court_margin
        ):
            return None

        # Clamp to court bounds
        x_ft = max(0.0, min(COURT_LENGTH, x_ft))
        y_ft = max(0.0, min(COURT_WIDTH, y_ft))
        return (x_ft, y_ft)

    def transform_points(
        self, H: np.ndarray, points: List[Tuple[float, float]]
    ) -> List[Optional[Tuple[float, float]]]:
        """Transform multiple pixel coordinates to court coordinates."""
        return [self.transform_point(H, px, py) for px, py in points]

    def reset(self):
        """Reset cached homography (e.g., on camera cut)."""
        self._last_good_H = None
        self._frames_since_good = 0

    @property
    def has_valid_homography(self) -> bool:
        return self._last_good_H is not None

    def _fallback(self) -> Optional[np.ndarray]:
        """Return cached homography if within fallback window."""
        self._frames_since_good += 1
        if (
            self._last_good_H is not None
            and self._frames_since_good <= self.fallback_frames
        ):
            return self._last_good_H.copy()
        return None

    def _validate(
        self,
        H: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        inlier_mask: Optional[np.ndarray],
    ) -> bool:
        """Validate a computed homography matrix."""
        # Check determinant — near zero means degenerate
        det = np.linalg.det(H)
        if abs(det) < 1e-6:
            return False

        # Check reprojection error on inliers
        if inlier_mask is not None:
            inliers_src = src[inlier_mask.ravel() == 1]
            inliers_dst = dst[inlier_mask.ravel() == 1]

            if len(inliers_src) < self.min_keypoints:
                return False

            # Transform inlier source points and check error
            src_h = np.hstack([inliers_src, np.ones((len(inliers_src), 1))])
            projected = (H @ src_h.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            errors = np.sqrt(np.sum((projected - inliers_dst) ** 2, axis=1))
            mean_error = np.mean(errors)

            if mean_error > self.max_reproj_error:
                return False

        # Check that court corners map to reasonable pixel positions
        # (inverse check — court corners should map to somewhere in a
        # reasonable image space)
        try:
            H_inv = np.linalg.inv(H)
            corners_court = np.array([
                [[0, 0]], [[COURT_LENGTH, 0]],
                [[COURT_LENGTH, COURT_WIDTH]], [[0, COURT_WIDTH]]
            ], dtype=np.float64)
            corners_pixel = cv2.perspectiveTransform(corners_court, H_inv)
            # Just check they don't go wildly negative or huge
            if np.any(np.abs(corners_pixel) > 1e5):
                return False
        except np.linalg.LinAlgError:
            return False

        return True
