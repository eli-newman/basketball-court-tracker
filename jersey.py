"""Jersey number recognition + per-track vote aggregator.

Uses Roboflow's purpose-trained `basketball-jersey-numbers-ocr` model. The
single-frame OCR is noisy (small numbers, motion blur, oblique angles), so
we accept a track's jersey number only after the same number wins K of the
last N samples. Sampling is throttled (every Nth frame, only for tracks
still without a locked number) so jersey OCR doesn't tank pipeline speed.

This also implicitly stabilizes identity across camera cuts: ByteTrack
issues a fresh track_id after every cut, but the jersey number is the
same human, so downstream code can collapse `track_id -> jersey_number`
to get a persistent per-player identity.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import base64

import cv2
import numpy as np

from config import Config
from detector import _local_infer, _post_with_retry


@dataclass
class JerseyRead:
    """A single-frame jersey number prediction for one player."""
    track_id: int
    number: str           # "0".."99"
    confidence: float


@dataclass
class JerseyAssignment:
    """A confirmed jersey number for a track, after vote aggregation."""
    track_id: int
    number: str
    samples: int          # number of frames it won
    locked: bool          # True once it crossed the confirm threshold


class JerseyNumberRecognizer:
    """Calls the Roboflow jersey OCR model on a player chest crop.

    Reuses the same backend abstraction as PlayerDetector — either an HTTP
    POST to `inference_host` or in-process via the `inference` package.
    """

    def __init__(self, config: Config):
        self.config = config
        self.api_url = (
            f"{config.inference_host.rstrip('/')}/{config.jersey_model_id}"
        )

    def read(self, chest_crop: np.ndarray) -> Optional[dict]:
        """Run one OCR call. Returns the raw {"predictions":[...]} dict."""
        if chest_crop.size == 0:
            return None
        if self.config.inference_backend == "local":
            return _local_infer(
                self.config.jersey_model_id,
                self.config.roboflow_api_key,
                chest_crop,
                self.config.jersey_confidence,
            )
        _, buffer = cv2.imencode(".jpg", chest_crop)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        return _post_with_retry(
            self.api_url,
            api_key=self.config.roboflow_api_key,
            confidence=self.config.jersey_confidence,
            img_b64=img_b64,
            timeout=15,
        )


def crop_chest(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """Cut the central chest region (where the number is) out of a player bbox.

    Vertical 15-55% of the bbox: below the chin, above the waistband.
    Horizontal 20-80% of the bbox: skip arms/sleeves where digits don't sit.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    x1 = max(0, min(x1, fw - 1))
    x2 = max(0, min(x2, fw - 1))
    y1 = max(0, min(y1, fh - 1))
    y2 = max(0, min(y2, fh - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    bh, bw = y2 - y1, x2 - x1
    cy1 = y1 + int(bh * 0.15)
    cy2 = y1 + int(bh * 0.55)
    cx1 = x1 + int(bw * 0.20)
    cx2 = x2 - int(bw * 0.20)
    if cy2 <= cy1 or cx2 <= cx1:
        return frame[y1:y2, x1:x2]
    return frame[cy1:cy2, cx1:cx2]


def parse_jersey_response(resp: Optional[dict]) -> Optional[JerseyRead]:
    """Extract (number, confidence) from a Roboflow OCR response.

    Accepts both classification-style predictions (top-1 number + confidence)
    and object-detection-style predictions (digit boxes; we concatenate by
    x position). Returns None when the response is empty or ambiguous.
    """
    if not resp:
        return None
    preds = resp.get("predictions") or []
    if not preds:
        return None

    # Classification style: first prediction is the top number
    p0 = preds[0]
    if "class" in p0 and not all("width" in p and "x" in p for p in preds):
        num = str(p0.get("class", "")).strip()
        conf = float(p0.get("confidence", 0.0))
        if num.isdigit() and 0 <= int(num) <= 99:
            return JerseyRead(track_id=-1, number=str(int(num)), confidence=conf)
        return None

    # Object-detection style: each pred is a digit box. Concatenate left→right.
    digits = []
    for p in preds:
        cls = str(p.get("class", "")).strip()
        if cls.isdigit() and len(cls) == 1:
            digits.append((float(p.get("x", 0.0)), cls, float(p.get("confidence", 0.0))))
    if not digits:
        return None
    digits.sort(key=lambda t: t[0])
    number = "".join(d[1] for d in digits[:2])  # cap at 2 digits
    if not number.isdigit() or int(number) > 99:
        return None
    conf = sum(d[2] for d in digits) / len(digits)
    return JerseyRead(track_id=-1, number=str(int(number)), confidence=conf)


class JerseyVoter:
    """Per-track vote aggregation.

    For each track_id we keep a Counter of {number -> votes}. A track's
    jersey is `locked` once one number has at least `confirm_at` votes AND
    is the unique top-1. Once locked, we stop sampling that track.

    On `reset_track(id)`, the vote history is dropped — used after a hard
    cut where ByteTrack issued the same id to a different person.
    """

    def __init__(self, confirm_at: int = 3, min_confidence: float = 0.5):
        self.confirm_at = confirm_at
        self.min_confidence = min_confidence
        self._votes: Dict[int, Counter] = {}
        self._locked: Dict[int, str] = {}

    def submit(self, read: JerseyRead) -> Optional[JerseyAssignment]:
        """Record a single-frame read. Returns the current assignment (if any)."""
        if read.confidence < self.min_confidence:
            return self.current(read.track_id)
        votes = self._votes.setdefault(read.track_id, Counter())
        votes[read.number] += 1
        return self.current(read.track_id)

    def current(self, track_id: int) -> Optional[JerseyAssignment]:
        """Return the current best assignment for a track, or None."""
        if track_id in self._locked:
            num = self._locked[track_id]
            samples = self._votes.get(track_id, Counter()).get(num, 0)
            return JerseyAssignment(track_id, num, samples, locked=True)
        votes = self._votes.get(track_id)
        if not votes:
            return None
        top_two = votes.most_common(2)
        top_num, top_count = top_two[0]
        if top_count >= self.confirm_at and (
            len(top_two) == 1 or top_count > top_two[1][1]
        ):
            self._locked[track_id] = top_num
            return JerseyAssignment(track_id, top_num, top_count, locked=True)
        # Provisional: return current top but not locked yet
        return JerseyAssignment(track_id, top_num, top_count, locked=False)

    def is_locked(self, track_id: int) -> bool:
        return track_id in self._locked

    def reset_track(self, track_id: int):
        self._votes.pop(track_id, None)
        self._locked.pop(track_id, None)

    def reset_all(self):
        """Clear all state. Use on a hard scene break where every id is stale."""
        self._votes.clear()
        self._locked.clear()

    def assignments(self) -> Dict[int, JerseyAssignment]:
        """All known assignments, locked + provisional."""
        out: Dict[int, JerseyAssignment] = {}
        for tid in set(self._votes) | set(self._locked):
            a = self.current(tid)
            if a is not None:
                out[tid] = a
        return out
