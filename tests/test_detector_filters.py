"""Tests for the detection-time sanity filters in PlayerDetector."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from detector import PlayerDetector


def _cfg(**overrides) -> Config:
    defaults = dict(
        roboflow_api_key="test",
        min_player_bbox_height=30,
        min_player_aspect_ratio=1.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _fake_response(*preds) -> dict:
    return {"predictions": list(preds)}


def _pred(x=500, y=300, w=40, h=100, cls="player", conf=0.9) -> dict:
    return {"x": x, "y": y, "width": w, "height": h, "class": cls, "confidence": conf}


def test_normal_player_passes():
    det = PlayerDetector(_cfg())
    with patch.object(det, "_call_api", return_value=_fake_response(_pred())):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = det.detect(frame)
    assert len(out) == 1


def test_too_short_dropped():
    """A 20px-tall detection is too small to be a real player on broadcast."""
    det = PlayerDetector(_cfg())
    with patch.object(det, "_call_api",
                      return_value=_fake_response(_pred(h=20))):
        out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert out == []


def test_wide_blob_dropped():
    """A horizontal box (aspect < 1.0) is a coach/photographer/crowd, not a player."""
    det = PlayerDetector(_cfg())
    with patch.object(det, "_call_api",
                      return_value=_fake_response(_pred(w=100, h=50))):
        out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert out == []


def test_referee_dropped():
    """Refs are filtered by class, not by geometry."""
    det = PlayerDetector(_cfg())
    with patch.object(det, "_call_api",
                      return_value=_fake_response(_pred(cls="referee"))):
        out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert out == []


def test_filters_disabled_keep_everything():
    """If thresholds are slack, even small boxes pass."""
    det = PlayerDetector(_cfg(min_player_bbox_height=1, min_player_aspect_ratio=0.0))
    with patch.object(det, "_call_api",
                      return_value=_fake_response(_pred(h=10), _pred(w=100, h=20))):
        out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(out) == 2


def test_mixed_predictions_only_real_players_kept():
    det = PlayerDetector(_cfg())
    preds = _fake_response(
        _pred(),                                # real player → keep
        _pred(h=15),                            # too short → drop
        _pred(cls="rim"),                       # not a player → drop
        _pred(cls="player-in-possession"),      # also a player → keep
        _pred(w=120, h=40),                     # wide blob → drop
    )
    with patch.object(det, "_call_api", return_value=preds):
        out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(out) == 2
    assert {p.class_name for p in out} == {"player", "player-in-possession"}
