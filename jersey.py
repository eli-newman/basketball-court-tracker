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
from detector import NumberDetection, _local_infer, _post_with_retry

# Minimum height (px) of a jersey-number crop after upscaling. Broadcast
# numbers are often 15-30 px tall in the source frame, which is way below
# what the OCR model was trained on. We upscale tight `number` crops to at
# least this height before sending — INTER_CUBIC preserves digit edges
# better than bilinear. 96 px is a sweet spot: smaller and the model
# starts hallucinating, bigger and we just waste bytes on the wire.
_OCR_MIN_HEIGHT = 96
# Padding around the number bbox before cropping, as a fraction of the
# bbox's own dimensions. Roboflow's `number` class tends to crop tight
# right against the digits; the OCR model wants a little quiet zone
# around them. 25% on each side ≈ "give the digit room to breathe"
# without dragging in stitching/neighbouring numbers.
_NUMBER_BBOX_PADDING = 0.25


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
        """Run one OCR call. Returns the raw {"predictions":[...]} dict.

        Always uses the hosted HTTP API regardless of the player/court
        detection backend. Reason: the Roboflow jersey OCR model is a
        PeftModel with BFloat16 weights, which fails to load on Apple
        Silicon (MPS doesn't support BFloat16) — and forcing the model
        to CPU adds complexity for marginal benefit since OCR runs
        infrequently (every Nth frame per unlocked track) and chest
        crops are small. Hybrid: local for heavy player/court inference,
        hosted for cheap OCR. Caller's `inference_backend` setting still
        controls everything else.
        """
        if chest_crop.size == 0:
            return None
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

    Used as fallback only — prefer `crop_number_bbox` when the detector
    surfaced a `number` bbox for this player. The chest crop is much wider
    than the actual digits, so it gives the OCR model lots of jersey
    background to hallucinate from.
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


def _bbox_contains(player_bbox: tuple, point: tuple) -> bool:
    """True iff (px, py) falls inside player_bbox (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = player_bbox
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def match_number_to_player(
    player_bbox: tuple,
    numbers: List[NumberDetection],
) -> Optional[NumberDetection]:
    """Pick the best `number` bbox belonging to this player, or None.

    A `number` bbox belongs to the player if its CENTER falls inside the
    player's bbox. When several numbers fall inside (rare — happens with
    overlapping defenders), we pick the most confident one and break
    ties by preferring numbers nearer the upper half (chest/back face on
    a standing player, where the front/back digit lives).

    Returns None when no number was detected on this player this frame —
    in that case the caller should skip OCR, not fall back to the wider
    chest crop. A blind read on no-number-visible is worse than no read:
    it pollutes the vote aggregator and locks in wrong numbers.
    """
    if not numbers:
        return None
    x1, y1, x2, y2 = player_bbox
    bbox_mid_y = (y1 + y2) / 2
    candidates = [n for n in numbers if _bbox_contains(player_bbox, n.center)]
    if not candidates:
        return None
    # Higher confidence first; on tie, prefer numbers in the upper 2/3 of
    # the bbox (where torso digits actually sit).
    candidates.sort(
        key=lambda n: (-n.confidence, abs(n.center[1] - bbox_mid_y))
    )
    return candidates[0]


def crop_number_bbox(
    frame: np.ndarray,
    number: NumberDetection,
    padding: float = _NUMBER_BBOX_PADDING,
) -> np.ndarray:
    """Crop the `number` region with a small padding margin.

    The detector's `number` bbox is tight around the digits; the OCR model
    likes a small quiet zone on each side. 25% padding is enough to give
    the digit room without pulling in neighbouring numbers / stitching.
    """
    x1, y1, x2, y2 = number.bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding
    pad_y = bh * padding
    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y
    fh, fw = frame.shape[:2]
    x1i = max(0, int(round(x1)))
    y1i = max(0, int(round(y1)))
    x2i = min(fw, int(round(x2)))
    y2i = min(fh, int(round(y2)))
    if x2i <= x1i or y2i <= y1i:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return frame[y1i:y2i, x1i:x2i]


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Upscale tiny crops so the OCR model has enough pixels to chew on.

    Roboflow's jersey-OCR was trained on chunky crops, but broadcast
    numbers from sideline angles are often 15-30 px tall. Letting them
    through at native size produces blurry, ambiguous reads. We upscale
    with INTER_CUBIC (preserves digit edges better than bilinear) to at
    least `_OCR_MIN_HEIGHT` pixels tall, keeping aspect ratio. Crops that
    are already large enough pass through unchanged.
    """
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if h >= _OCR_MIN_HEIGHT:
        return crop
    scale = _OCR_MIN_HEIGHT / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(crop, (new_w, _OCR_MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)


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
