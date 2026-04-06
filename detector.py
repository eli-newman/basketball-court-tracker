"""Detection layer: Roboflow API for players + court keypoints."""

from dataclasses import dataclass
from typing import List, Optional
import base64

import cv2
import numpy as np
import requests

from config import Config
from homography import CourtKeypoint


@dataclass
class PlayerDetection:
    """A detected basketball player."""
    bbox: tuple           # (x1, y1, x2, y2)
    bottom_center: tuple  # (x, y) — feet position for court mapping
    center: tuple         # (cx, cy) — bbox center
    confidence: float
    class_name: str       # model class (e.g., "player", "referee")


class PlayerDetector:
    """Roboflow API basketball player detection."""

    def __init__(self, config: Config):
        self.config = config
        self.api_url = f"https://detect.roboflow.com/{config.player_model_id}"

    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """Detect players in a frame.

        Returns list of PlayerDetection with bounding boxes and feet positions.
        """
        try:
            result = self._call_api(frame, self.config.player_confidence)
            if result is None:
                return []

            players = []
            for pred in result.get("predictions", []):
                x, y = pred["x"], pred["y"]
                w, h = pred["width"], pred["height"]
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2

                players.append(PlayerDetection(
                    bbox=(x1, y1, x2, y2),
                    bottom_center=(x, y2),  # bottom center = feet
                    center=(x, y),
                    confidence=pred.get("confidence", 0.0),
                    class_name=pred.get("class", "player"),
                ))
            return players
        except Exception:
            return []

    def _call_api(self, frame: np.ndarray, confidence: float) -> Optional[dict]:
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        response = requests.post(
            self.api_url,
            params={
                "api_key": self.config.roboflow_api_key,
                "confidence": confidence,
            },
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )

        if response.status_code != 200:
            return None
        return response.json()


class CourtKeypointDetector:
    """Roboflow API basketball court keypoint detection."""

    def __init__(self, config: Config):
        self.config = config
        self.api_url = f"https://detect.roboflow.com/{config.court_model_id}"

    def detect(self, frame: np.ndarray) -> List[CourtKeypoint]:
        """Detect court keypoints in a frame.

        Returns list of CourtKeypoint with pixel positions and class names.
        """
        try:
            result = self._call_api(frame, self.config.court_confidence)
            if result is None:
                return []

            keypoints = []
            for pred in result.get("predictions", []):
                keypoints.append(CourtKeypoint(
                    name=pred.get("class", ""),
                    pixel_x=pred["x"],
                    pixel_y=pred["y"],
                    confidence=pred.get("confidence", 0.0),
                ))
            return keypoints
        except Exception:
            return []

    def _call_api(self, frame: np.ndarray, confidence: float) -> Optional[dict]:
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        response = requests.post(
            self.api_url,
            params={
                "api_key": self.config.roboflow_api_key,
                "confidence": confidence,
            },
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )

        if response.status_code != 200:
            return None
        return response.json()

    def inspect_classes(self, frame: np.ndarray) -> List[str]:
        """Run detection and return all unique class names found.

        Useful for calibrating the ROBOFLOW_KEYPOINT_MAP in court.py.
        """
        keypoints = self.detect(frame)
        return sorted(set(kp.name for kp in keypoints))
