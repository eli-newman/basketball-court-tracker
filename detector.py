"""Detection layer: Roboflow models for players + court keypoints.

Two backends are available:
- "hosted" (default): POSTs each frame to detect.roboflow.com. ~0.5 fps,
  network-bound. Useful for quick local validation without GPU setup.
- "local": runs the same Roboflow models locally via the `inference`
  package (CPU) or `inference-gpu` (CUDA). ~30+ fps on a Colab T4. Same
  model_ids, same response shape — drop-in.

Pick the backend via `Config.inference_backend` ("hosted" or "local").
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import base64
import time

import cv2
import numpy as np
import requests

from config import Config
from homography import CourtKeypoint


_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


# ── Local inference backend (lazy) ───────────────────────────────────────────
# Models are heavy (weights, ONNX runtime) and only needed when backend="local".
# We cache one loaded model per model_id for the lifetime of the process.

_LOCAL_MODEL_CACHE: Dict[str, Any] = {}


def _get_local_model(model_id: str, api_key: str) -> Any:
    """Lazy-load a Roboflow model for local inference. Cached by model_id."""
    if model_id in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_id]
    try:
        from inference import get_model  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Local inference requires the `inference` package. "
            "Install with `pip install inference` (CPU) or "
            "`pip install inference-gpu` (CUDA)."
        ) from e
    model = get_model(model_id=model_id, api_key=api_key)
    _LOCAL_MODEL_CACHE[model_id] = model
    return model


def _local_infer(model_id: str, api_key: str, frame: np.ndarray, confidence: float) -> Optional[dict]:
    """Run a frame through a locally-loaded Roboflow model.

    Returns a dict in the same shape as the hosted API
    (``{"predictions": [...]}``) so downstream parsing is identical.
    """
    model = _get_local_model(model_id, api_key)
    try:
        responses = model.infer(frame, confidence=confidence)
    except Exception as e:  # pragma: no cover — surface the real error
        print(f"[detector] local infer failed for {model_id}: {type(e).__name__}: {e}")
        return None
    if not responses:
        return {"predictions": []}
    resp = responses[0]
    # InferenceResponse → dict; hosted API key is "predictions"
    if hasattr(resp, "dict"):
        return resp.dict(by_alias=True, exclude_none=True)
    if hasattr(resp, "model_dump"):
        return resp.model_dump(by_alias=True, exclude_none=True)
    if isinstance(resp, dict):
        return resp
    print(f"[detector] unexpected local infer response type: {type(resp).__name__}")
    return None


def _post_with_retry(
    url: str,
    *,
    api_key: str,
    confidence: float,
    img_b64: str,
    timeout: float,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
) -> Optional[dict]:
    """POST to Roboflow with retry on transient failures.

    Returns parsed JSON on success, or None after all attempts fail.
    Caller decides what to do with None (typically: return empty detection list).
    """
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                url,
                params={"api_key": api_key, "confidence": confidence},
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code not in _TRANSIENT_STATUSES:
                # Permanent error (auth, bad request, etc.) — don't retry.
                print(f"[detector] HTTP {response.status_code} from {url}: {response.text[:200]}")
                return None
            # else fall through to retry
            last_err = f"HTTP {response.status_code}"
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = f"{type(e).__name__}: {e}"

        if attempt < max_attempts - 1:
            time.sleep(backoff_base * (2 ** attempt))
    print(f"[detector] giving up after {max_attempts} attempts: {last_err}")
    return None


@dataclass
class PlayerDetection:
    """A detected basketball player."""
    bbox: tuple           # (x1, y1, x2, y2)
    bottom_center: tuple  # (x, y) — feet position for court mapping
    center: tuple         # (cx, cy) — bbox center
    confidence: float
    class_name: str       # model class (e.g., "player", "referee")
    track_id: int = -1    # persistent ID from tracker; -1 = untracked


class PlayerDetector:
    """Roboflow API basketball player detection."""

    def __init__(self, config: Config):
        self.config = config
        self.api_url = f"https://detect.roboflow.com/{config.player_model_id}"

    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """Detect players in a frame.

        Returns list of PlayerDetection with bounding boxes and feet positions.
        Only `player` and `player-in-possession` classes are returned;
        `referee`, `rim`, `ball`, `number` are filtered out.
        """
        result = self._call_api(frame, self.config.player_confidence)
        if result is None:
            return []

        players = []
        for pred in result.get("predictions", []):
            cls = pred.get("class", "player")
            if cls not in ("player", "player-in-possession"):
                continue
            x, y = pred["x"], pred["y"]
            w, h = pred["width"], pred["height"]
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2

            players.append(PlayerDetection(
                bbox=(x1, y1, x2, y2),
                bottom_center=(x, y2),  # bottom center = feet
                center=(x, y),
                confidence=pred.get("confidence", 0.0),
                class_name=cls,
            ))
        return players

    def _call_api(self, frame: np.ndarray, confidence: float) -> Optional[dict]:
        if self.config.inference_backend == "local":
            return _local_infer(
                self.config.player_model_id,
                self.config.roboflow_api_key,
                frame,
                confidence,
            )
        _, buffer = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        return _post_with_retry(
            self.api_url,
            api_key=self.config.roboflow_api_key,
            confidence=confidence,
            img_b64=img_b64,
            timeout=15,
        )


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

        Roboflow keypoint API returns each keypoint with both `class_id`
        (sequential 0-N matching our _KEYPOINTS_CM order) and `class`
        (sparse label string like "01","02","04" — gaps).
        We prefer `class_id` because it's a true index into our list.
        """
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
                    # Prefer class_id (sequential index). Fall back to class label.
                    name: Union[int, str]
                    if "class_id" in kp and kp["class_id"] is not None:
                        name = int(kp["class_id"])
                    else:
                        raw = kp.get("class_name") or kp.get("class") or ""
                        name = str(raw)
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

    def _call_api(self, frame: np.ndarray, confidence: float) -> Optional[dict]:
        if self.config.inference_backend == "local":
            return _local_infer(
                self.config.court_model_id,
                self.config.roboflow_api_key,
                frame,
                confidence,
            )
        _, buffer = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        return _post_with_retry(
            self.api_url,
            api_key=self.config.roboflow_api_key,
            confidence=confidence,
            img_b64=img_b64,
            timeout=15,
        )

    def inspect_keypoints(self, frame: np.ndarray) -> dict:
        """Run detection and return raw response for calibration.

        Use this to inspect the actual keypoint indices/names and pixel
        positions returned by the model. Essential for setting up
        KEYPOINT_INDEX_MAP in court.py.

        Returns the raw API response dict.
        """
        result = self._call_api(frame, 0.1)  # low confidence to see all keypoints
        return result or {}
