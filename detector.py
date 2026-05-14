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


@dataclass
class BallDetection:
    """A detected basketball.

    Unlike PlayerDetection there's no `bottom_center` distinction because the
    ball is small and often airborne — the bbox center is the most stable
    reference point. `bottom_center` is still exposed because the pipeline
    sometimes uses it to map onto the court (rough approximation when the
    ball is mid-flight, fine when it's near a player's hands at hip level).
    """
    bbox: tuple           # (x1, y1, x2, y2)
    center: tuple         # (cx, cy) — bbox center, the canonical ball position
    bottom_center: tuple  # (x, y) — bottom of bbox; used for court mapping
    confidence: float


@dataclass
class ActionObservation:
    """A model action prediction (player-jump-shot, player-layup-dunk, etc.).

    The Roboflow basketball-player-detection-3 model returns several action
    classes alongside players and balls. Each action comes back as its own
    bbox — typically tightly overlapping a player. We don't associate it
    with a track at detection time; the pipeline does that downstream via
    bbox IoU against tracked players.
    """
    bbox: tuple           # (x1, y1, x2, y2)
    center: tuple         # (cx, cy)
    class_name: str       # e.g. "player-jump-shot", "ball-in-basket"
    confidence: float


@dataclass
class NumberDetection:
    """A `number` bbox from the player-detection model.

    The detector emits a separate `number` class for each visible jersey
    number on a player's back/chest. These are way tighter than the
    heuristic chest-rectangle crop (`jersey.crop_chest`), so when a number
    bbox is available the jersey-OCR step uses it directly — typically
    cutting noise (shoulders, chest folds, background) out of the crop
    entirely and dramatically improving OCR accuracy.

    Association with a player happens downstream via bbox containment:
    a number whose center falls inside a player's bbox belongs to that
    player. The pipeline does that match in `_update_jersey_numbers`.
    """
    bbox: tuple           # (x1, y1, x2, y2)
    center: tuple         # (cx, cy) — used for containment test against player bbox
    confidence: float


# Classes the detector exposes via `detect_all` in addition to players/ball.
# Used both for shot-action observations and for the rim itself (used as a
# geometric sanity check on `ball-in-basket` events later).
_ACTION_CLASSES = {
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "ball-in-basket",
    "rim",
}


class PlayerDetector:
    """Roboflow API basketball player detection.

    The underlying model (basketball-player-detection-3) returns multiple
    classes per frame — `player`, `player-in-possession`, `referee`, `ball`,
    `rim`, `number`, plus several action classes. Historically this class
    discarded everything but the player rows; we now expose a
    `detect_with_ball` method that returns the ball detection from the same
    inference call (no extra API cost).
    """

    # Bbox sanity filters apply to PLAYERS only — ball/rim are tiny by nature.
    _PLAYER_CLASSES = {"player", "player-in-possession"}

    def __init__(self, config: Config):
        self.config = config
        self.api_url = f"{config.inference_host.rstrip('/')}/{config.player_model_id}"
        # Per-class detection counter for `--detector-class-histogram`. Counts
        # EVERY prediction the model returned, regardless of whether we kept
        # it downstream — so we can see if action classes are appearing in
        # the response at all on a given clip.
        self._class_counts: Dict[str, int] = {}

    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """Detect players in a frame.

        Returns list of PlayerDetection with bounding boxes and feet positions.
        Only `player` and `player-in-possession` classes are returned;
        `referee`, `rim`, `ball`, `number` are filtered out.
        """
        players, _ = self.detect_with_ball(frame)
        return players

    def print_class_histogram(self):
        """Print the cumulative class counter — useful for verifying whether
        the model is returning action classes (`player-jump-shot`,
        `ball-in-basket`, etc.) at all on a given clip. Zero counts mean
        the model never emitted that class, NOT that our parsing is
        filtering it out.
        """
        if not self._class_counts:
            print("[detector] no predictions seen across the run")
            return
        total = sum(self._class_counts.values())
        print(f"[detector] class histogram ({total} total predictions):")
        for cls, n in sorted(
            self._class_counts.items(), key=lambda kv: -kv[1]
        ):
            print(f"  {cls:<30}  {n:>6}  ({100 * n / total:.1f}%)")

    def detect_with_ball(
        self, frame: np.ndarray,
    ) -> tuple[List[PlayerDetection], Optional[BallDetection]]:
        """Detect players AND ball in one API call. Discards action classes.

        See `detect_all` for the version that also returns shot-action
        observations.
        """
        players, ball, _, _ = self.detect_all(frame)
        return players, ball

    def detect_all(
        self, frame: np.ndarray,
    ) -> tuple[
        List[PlayerDetection],
        Optional[BallDetection],
        List[ActionObservation],
        List[NumberDetection],
    ]:
        """One inference call → players, ball, action observations, numbers.

        The basketball-player-detection-3 model returns multiple classes per
        frame. We bucket them into four streams:

          - **Players** (`player`, `player-in-possession`): geometry-filtered
            for sane standing-player shapes.
          - **Ball**: highest-confidence kept if multiple.
          - **Actions**: every non-player non-ball detection of interest for
            event detection (`player-jump-shot`, `player-layup-dunk`,
            `player-shot-block`, `ball-in-basket`, `rim`).
          - **Numbers**: `number` bboxes — tight crops on the visible jersey
            digits. Consumed by the jersey-OCR step in the pipeline; was
            dropped on the floor before. Surfaces 25-30% of the model's
            predictions that we used to throw away.

        Everything else (`referee` etc.) is dropped.
        """
        result = self._call_api(frame, self.config.player_confidence)
        if result is None:
            return [], None, [], []

        players: List[PlayerDetection] = []
        ball_candidates: List[BallDetection] = []
        actions: List[ActionObservation] = []
        numbers: List[NumberDetection] = []

        for pred in result.get("predictions", []):
            cls = pred.get("class", "")
            x, y = pred["x"], pred["y"]
            w, h = pred["width"], pred["height"]
            conf = pred.get("confidence", 0.0)
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2

            # Tally every class the model returned, before any filtering.
            self._class_counts[cls] = self._class_counts.get(cls, 0) + 1

            if cls in self._PLAYER_CLASSES:
                # Sanity filter: drop boxes that don't look like a standing
                # player. Crowd / sideline false positives are usually small
                # or wide.
                if h < self.config.min_player_bbox_height:
                    continue
                if w > 0 and (h / w) < self.config.min_player_aspect_ratio:
                    continue
                players.append(PlayerDetection(
                    bbox=(x1, y1, x2, y2),
                    bottom_center=(x, y2),  # bottom center = feet
                    center=(x, y),
                    confidence=conf,
                    class_name=cls,
                ))
            elif cls == "ball":
                # No sanity filter — ball is tiny by design. We do require
                # the model's own confidence threshold (already applied
                # server-side via `player_confidence`) but nothing more.
                ball_candidates.append(BallDetection(
                    bbox=(x1, y1, x2, y2),
                    center=(x, y),
                    bottom_center=(x, y2),
                    confidence=conf,
                ))
            elif cls in _ACTION_CLASSES:
                actions.append(ActionObservation(
                    bbox=(x1, y1, x2, y2),
                    center=(x, y),
                    class_name=cls,
                    confidence=conf,
                ))
            elif cls == "number":
                numbers.append(NumberDetection(
                    bbox=(x1, y1, x2, y2),
                    center=(x, y),
                    confidence=conf,
                ))

        ball = max(ball_candidates, key=lambda b: b.confidence) if ball_candidates else None
        return players, ball, actions, numbers

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
        self.api_url = f"{config.inference_host.rstrip('/')}/{config.court_model_id}"
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
