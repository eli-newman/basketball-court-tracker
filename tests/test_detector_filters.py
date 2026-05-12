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


# ── Ball extraction ─────────────────────────────────────────────────────────


def test_detect_with_ball_returns_ball():
    det = PlayerDetector(_cfg())
    preds = _fake_response(
        _pred(),
        _pred(x=640, y=360, w=30, h=30, cls="ball", conf=0.75),
    )
    with patch.object(det, "_call_api", return_value=preds):
        players, ball = det.detect_with_ball(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(players) == 1
    assert ball is not None
    assert ball.center == (640, 360)
    assert ball.confidence == 0.75
    # bbox is (x-w/2, y-h/2, x+w/2, y+h/2) → (625, 345, 655, 375)
    assert ball.bbox == (625.0, 345.0, 655.0, 375.0)


def test_detect_with_ball_returns_none_when_no_ball():
    det = PlayerDetector(_cfg())
    with patch.object(det, "_call_api", return_value=_fake_response(_pred())):
        players, ball = det.detect_with_ball(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(players) == 1
    assert ball is None


def test_detect_with_ball_keeps_highest_confidence_ball():
    """Multiple ball detections → pick the most confident one."""
    det = PlayerDetector(_cfg())
    preds = _fake_response(
        _pred(x=100, y=100, w=20, h=20, cls="ball", conf=0.5),
        _pred(x=900, y=400, w=25, h=25, cls="ball", conf=0.91),
        _pred(x=500, y=300, w=30, h=30, cls="ball", conf=0.7),
    )
    with patch.object(det, "_call_api", return_value=preds):
        _, ball = det.detect_with_ball(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert ball is not None
    assert ball.center == (900, 400)
    assert ball.confidence == 0.91


def test_detect_with_ball_ignores_non_player_non_ball_classes():
    """rim, number, etc. should be filtered out — neither players nor balls."""
    det = PlayerDetector(_cfg())
    preds = _fake_response(
        _pred(cls="rim"),
        _pred(cls="number"),
        _pred(cls="ball-in-basket", w=25, h=25),
    )
    with patch.object(det, "_call_api", return_value=preds):
        players, ball = det.detect_with_ball(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert players == []
    assert ball is None


def test_detect_with_ball_api_failure_returns_empty():
    """API failure (None response) → empty players + no ball, no crash."""
    det = PlayerDetector(_cfg())
    with patch.object(det, "_call_api", return_value=None):
        players, ball = det.detect_with_ball(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert players == []
    assert ball is None


def test_legacy_detect_method_still_works():
    """detect() still returns just the players list, unchanged from before."""
    det = PlayerDetector(_cfg())
    preds = _fake_response(_pred(), _pred(cls="ball", w=20, h=20))
    with patch.object(det, "_call_api", return_value=preds):
        out = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].class_name == "player"


# ── Action class extraction ─────────────────────────────────────────────────


def test_detect_all_returns_action_observations():
    """detect_all surfaces jump-shot / basket / rim observations."""
    det = PlayerDetector(_cfg())
    preds = _fake_response(
        _pred(),                                                  # player
        _pred(cls="ball", x=400, y=200, w=20, h=20),              # ball
        _pred(cls="player-jump-shot", x=500, y=300, w=80, h=200),
        _pred(cls="ball-in-basket", x=600, y=100, w=30, h=30),
        _pred(cls="rim", x=600, y=100, w=50, h=20),
    )
    with patch.object(det, "_call_api", return_value=preds):
        players, ball, actions = det.detect_all(
            np.zeros((720, 1280, 3), dtype=np.uint8),
        )
    assert len(players) == 1
    assert ball is not None
    assert len(actions) == 3
    classes = {a.class_name for a in actions}
    assert classes == {"player-jump-shot", "ball-in-basket", "rim"}


def test_detect_all_drops_unknown_classes():
    """`number`, `referee`, etc. don't appear in actions."""
    det = PlayerDetector(_cfg())
    preds = _fake_response(
        _pred(cls="number"),
        _pred(cls="referee"),
        _pred(cls="player-jump-shot", x=500, y=300, w=80, h=200),
    )
    with patch.object(det, "_call_api", return_value=preds):
        _, _, actions = det.detect_all(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(actions) == 1
    assert actions[0].class_name == "player-jump-shot"


def test_detect_all_api_failure_returns_empty_triple():
    """API failure yields ([], None, []) — never crashes."""
    det = PlayerDetector(_cfg())
    with patch.object(det, "_call_api", return_value=None):
        players, ball, actions = det.detect_all(
            np.zeros((720, 1280, 3), dtype=np.uint8),
        )
    assert players == []
    assert ball is None
    assert actions == []
