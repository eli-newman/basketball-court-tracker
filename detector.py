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
    """Roboflow API basketball court keypoint detection.

    The basketball-court-detection-2 model is a YOLOv11m-pose model.
    It returns ONE detection ("the court") with an ARRAY of keypoints,
    not separate object detection classes.

    The API response format for keypoint models is:
    {
      "predictions": [{
        "x": ..., "y": ..., "width": ..., "height": ...,
        "keypoints": [
          {"x": px, "y": py, "confidence": conf, "class_name": "0"},
          {"x": px, "y": py, "confidence": conf, "class_name": "1"},
          ...
        ]
      }]
    }

    Each keypoint index maps to a court landmark via KEYPOINT_INDEX_MAP in court.py.
    """

    def __init__(self, config: Config):
        self.config = config
        self.api_url = f"https://detect.roboflow.com/{config.court_model_id}"
        self._last_raw_response: Optional[dict] = None

    def detect(self, frame: np.ndarray) -> List[CourtKeypoint]:
        """Detect court keypoints in a frame.

        Handles both pose model (keypoints array) and object detection
        (separate predictions) response formats.
        """
        try:
            result = self._call_api(frame, self.config.court_confidence)
            if result is None:
                return []

            self._last_raw_response = result
            keypoints = []

            for pred in result.get("predictions", []):
                # Pose model format: prediction has a "keypoints" array
                if "keypoints" in pred:
                    for kp in pred["keypoints"]:
                        conf = kp.get("confidence", 0.0)
                        if conf < self.config.court_confidence:
                            continue
                        # class_name is the keypoint index as string (e.g., "0", "1")
                        # or a descriptive name
                        name = kp.get("class_name", kp.get("class", ""))
                        try:
                            name = int(name)  # convert "0" → 0 for index-based lookup
                        except (ValueError, TypeError):
                            pass  # keep as string for name-based lookup
                        keypoints.append(CourtKeypoint(
                            name=name,
                            pixel_x=kp["x"],
                            pixel_y=kp["y"],
                            confidence=conf,
                        ))
                else:
                    # Object detection format fallback: each prediction is a keypoint
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

    def inspect_keypoints(self, frame: np.ndarray) -> dict:
        """Run detection and return raw response for calibration.

        Use this to inspect the actual keypoint indices/names and pixel
        positions returned by the model. Essential for setting up
        KEYPOINT_INDEX_MAP in court.py.

        Returns the raw API response dict.
        """
        result = self._call_api(frame, 0.1)  # low confidence to see all keypoints
        return result or {}
