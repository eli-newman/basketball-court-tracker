"""Tests for TeamClassifier (body-color feature, occlusion-aware aggregation)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from detector import PlayerDetection
from team_classifier import (
    TEAM_PROFILES,
    TeamClassifier,
    UNKNOWN_TEAM,
    _nearest_anchor,
    is_anomalous_frame,
    resolve_team_profile,
)


def _player(track_id: int, bbox: tuple) -> PlayerDetection:
    return PlayerDetection(
        bbox=bbox,
        bottom_center=((bbox[0] + bbox[2]) / 2, bbox[3]),
        center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        confidence=0.9,
        class_name="player",
        track_id=track_id,
    )


def _frame_with_navy_and_white_jerseys() -> np.ndarray:
    """800x1280 BGR frame: solid navy on the left half, solid white on the right.

    Navy = (50, 30, 30) BGR ≈ HSV (120, 152, 50) — low V, high S.
    White = (235, 235, 235) BGR ≈ HSV (0, 0, 235) — high V, low S.

    Realistic test for the body-lightness feature: this is the classic NBA
    home-vs-road matchup shape.
    """
    f = np.zeros((800, 1280, 3), dtype=np.uint8)
    f[:, :640] = (50, 30, 30)        # navy BGR (Knicks-like)
    f[:, 640:] = (235, 235, 235)     # white BGR (Sixers-like)
    return f


def test_classifies_two_distinct_jerseys():
    """Two players on light/dark backgrounds → different teams."""
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=2,
        refit_every=1,
        min_samples_to_classify=2,
    )
    frame = _frame_with_navy_and_white_jerseys()
    navy_player = _player(track_id=1, bbox=(100, 100, 200, 600))
    white_player = _player(track_id=2, bbox=(900, 100, 1000, 600))

    for _ in range(3):
        result = clf.classify(frame, [navy_player, white_player])

    assert clf.is_calibrated
    assert result[0] != result[1]
    assert set(result) == {0, 1}


def test_untracked_players_get_unknown_team():
    """Players with track_id == -1 don't participate in the aggregator."""
    clf = TeamClassifier(n_teams=2, warmup_frames=1, refit_every=1,
                          min_samples_to_classify=1)
    frame = _frame_with_navy_and_white_jerseys()
    untracked = _player(track_id=-1, bbox=(100, 100, 200, 600))
    out = clf.classify(frame, [untracked])
    assert out == [UNKNOWN_TEAM]


def test_returns_unknown_until_warmup_complete():
    clf = TeamClassifier(n_teams=2, warmup_frames=5, refit_every=1,
                          min_samples_to_classify=1)
    frame = _frame_with_navy_and_white_jerseys()
    p1 = _player(1, (100, 100, 200, 600))
    p2 = _player(2, (900, 100, 1000, 600))

    for _ in range(4):
        out = clf.classify(frame, [p1, p2])
        assert out == [UNKNOWN_TEAM, UNKNOWN_TEAM]
        assert not clf.is_calibrated

    out = clf.classify(frame, [p1, p2])
    assert clf.is_calibrated
    assert UNKNOWN_TEAM not in out


def test_overlap_does_not_collapse_two_teams_into_one():
    """Occlusion mask must keep an navy defender on a (mostly-white) overlap
    classified as navy — the white pixels inside the defender's bbox should
    not leak into the defender's body sample.
    """
    f = np.full((800, 1280, 3), (50, 30, 30), dtype=np.uint8)  # navy
    f[100:700, 500:900] = (235, 235, 235)  # white rectangle (offensive area)

    offensive = _player(track_id=1, bbox=(500, 100, 900, 700))  # bbox ON white
    defender = _player(track_id=2, bbox=(800, 100, 1100, 700))  # overlaps white

    clf = TeamClassifier(n_teams=2, warmup_frames=2, refit_every=1,
                          min_samples_to_classify=2)
    for _ in range(3):
        result = clf.classify(f, [offensive, defender])

    assert result[0] != result[1]


def test_one_track_one_team_assignment_after_many_frames():
    """A track that always sees the same body color → stable team_id."""
    clf = TeamClassifier(n_teams=2, warmup_frames=2, refit_every=1,
                          min_samples_to_classify=2)
    frame = _frame_with_navy_and_white_jerseys()
    navy = _player(1, (100, 100, 200, 600))
    white = _player(2, (900, 100, 1000, 600))

    assignments = []
    for _ in range(20):
        out = clf.classify(frame, [navy, white])
        assignments.append(tuple(out))
    final = assignments[-10:]
    assert all(a == final[0] for a in final)


def test_team_colors_bgr_returns_centers():
    clf = TeamClassifier(n_teams=2, warmup_frames=1, refit_every=1,
                          min_samples_to_classify=1)
    frame = _frame_with_navy_and_white_jerseys()
    navy = _player(1, (100, 100, 200, 600))
    white = _player(2, (900, 100, 1000, 600))
    for _ in range(2):
        clf.classify(frame, [navy, white])
    colors = clf.team_colors_bgr
    assert len(colors) == 2
    for b, g, r in colors:
        assert 0 <= b <= 255 and 0 <= g <= 255 and 0 <= r <= 255


def test_no_players_returns_empty():
    clf = TeamClassifier(n_teams=2, warmup_frames=1)
    assert clf.classify(_frame_with_navy_and_white_jerseys(), []) == []


# ── Supervised (anchored) mode ──────────────────────────────────────────────


def test_resolve_team_profile_case_insensitive():
    a = resolve_team_profile("Knicks")
    b = resolve_team_profile("  KNICKS  ")
    c = resolve_team_profile("knicks")
    assert a is not None
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_resolve_team_profile_unknown_returns_none():
    assert resolve_team_profile("not-a-team") is None
    assert resolve_team_profile("") is None


def test_all_team_profiles_have_consistent_4d_shape():
    """Every profile must be 4-D: [body_v, body_s, accent_red, accent_blue]."""
    for name, profile in TEAM_PROFILES.items():
        assert profile.shape == (4,), f"{name} has wrong shape: {profile.shape}"


def test_home_away_variants_resolve():
    """Both bare and variant forms are registered."""
    assert resolve_team_profile("knicks") is not None
    assert resolve_team_profile("knicks-road") is not None
    assert resolve_team_profile("knicks-home") is not None
    assert resolve_team_profile("sixers-home") is not None


def test_home_and_road_have_opposite_lightness():
    """For teams with home/road registered, the V values should differ
    meaningfully — at least one variant is light, one is dark.
    Anchors are calibrated to real broadcast samples (which compress the
    V range), so the threshold is 50 not 100.
    """
    pairs = [
        ("knicks-road", "knicks-home"),
        ("celtics-road", "celtics-home"),
        ("bulls-road", "bulls-home"),
    ]
    for road, home in pairs:
        r = resolve_team_profile(road)
        h = resolve_team_profile(home)
        assert r is not None and h is not None
        assert abs(r[0] - h[0]) >= 50, f"{road} vs {home}: V's too close"


def test_nearest_anchor_navy_vs_white():
    """Navy track → navy anchor, white track → white anchor (V dominates)."""
    anchors = np.stack(
        [
            np.array([55, 200, 0.0, 0.4], dtype=np.float32),   # navy
            np.array([215, 35, 0.3, 0.2], dtype=np.float32),   # white
        ],
        axis=0,
    )
    tracks = np.array([
        [60, 180, 0.05, 0.30],    # dark, navy-ish → anchor 0
        [220, 40, 0.20, 0.10],    # bright, white-ish → anchor 1
    ], dtype=np.float32)
    labels = _nearest_anchor(tracks, anchors)
    assert labels[0] == 0
    assert labels[1] == 1


def test_nearest_anchor_accent_breaks_tie_for_two_dark_teams():
    """Two teams with same body lightness → accent differentiates."""
    anchors = np.stack(
        [
            np.array([85, 200, 0.50, 0.0], dtype=np.float32),  # red-bodied
            np.array([85, 200, 0.0, 0.50], dtype=np.float32),  # blue-bodied
        ],
        axis=0,
    )
    tracks = np.array([
        [80, 210, 0.45, 0.05],
        [90, 195, 0.05, 0.55],
    ], dtype=np.float32)
    labels = _nearest_anchor(tracks, anchors)
    assert labels[0] == 0
    assert labels[1] == 1


def test_anchored_classifier_skips_kmeans():
    """Anchored mode uses cosine-replacement L2; _kmeans stays None."""
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=1,
        refit_every=1,
        min_samples_to_classify=1,
        team_anchors=[
            TEAM_PROFILES["knicks"],  # navy (V=55)
            TEAM_PROFILES["sixers"],  # white (V=215)
        ],
        team_names=["knicks", "sixers"],
    )
    frame = _frame_with_navy_and_white_jerseys()
    navy_p = _player(1, (100, 100, 200, 600))
    white_p = _player(2, (900, 100, 1000, 600))

    for _ in range(2):
        out = clf.classify(frame, [navy_p, white_p])

    assert clf.is_calibrated
    assert clf._kmeans is None
    # Navy → team 0 (knicks), white → team 1 (sixers)
    assert out[0] == 0
    assert out[1] == 1


def test_anchored_classifier_works_with_single_track():
    """KMeans needs n_teams tracks to fit; anchored mode does not."""
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=1,
        refit_every=1,
        min_samples_to_classify=1,
        team_anchors=[TEAM_PROFILES["knicks"], TEAM_PROFILES["sixers"]],
        team_names=["knicks", "sixers"],
    )
    frame = _frame_with_navy_and_white_jerseys()
    navy_only = _player(1, (100, 100, 200, 600))

    for _ in range(2):
        out = clf.classify(frame, [navy_only])

    assert clf.is_calibrated
    assert out[0] == 0  # navy → knicks


def test_anchored_classifier_team_colors_bgr_uses_canonical():
    """team_colors_bgr returns canonical NBA BGRs when names are provided."""
    from team_classifier import TEAM_DISPLAY_BGR
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=1,
        refit_every=1,
        min_samples_to_classify=1,
        team_anchors=[TEAM_PROFILES["knicks"], TEAM_PROFILES["sixers"]],
        team_names=["knicks", "sixers"],
    )
    colors = clf.team_colors_bgr
    assert colors[0] == TEAM_DISPLAY_BGR["knicks"]
    assert colors[1] == TEAM_DISPLAY_BGR["sixers"]


def test_team_colors_bgr_resolves_home_road_variants():
    """`knicks-home` falls back to `knicks` in the BGR display map."""
    from team_classifier import TEAM_DISPLAY_BGR
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=1, refit_every=1, min_samples_to_classify=1,
        team_anchors=[
            TEAM_PROFILES["knicks-home"], TEAM_PROFILES["sixers-road"],
        ],
        team_names=["knicks-home", "sixers-road"],
    )
    colors = clf.team_colors_bgr
    assert colors[0] == TEAM_DISPLAY_BGR["knicks"]
    assert colors[1] == TEAM_DISPLAY_BGR["sixers"]


def test_anchored_classifier_rejects_mismatched_anchor_count():
    with pytest.raises(ValueError):
        TeamClassifier(
            n_teams=2,
            team_anchors=[TEAM_PROFILES["knicks"]],
        )


def test_anchored_classifier_rejects_mismatched_names_count():
    with pytest.raises(ValueError):
        TeamClassifier(
            n_teams=2,
            team_anchors=[TEAM_PROFILES["knicks"], TEAM_PROFILES["sixers"]],
            team_names=["knicks"],
        )


# ── Flash-frame gate ────────────────────────────────────────────────────────


def test_anomalous_frame_detects_bright_flash():
    """A near-white frame (replay flash / transition) is flagged."""
    flash = np.full((600, 800, 3), 235, dtype=np.uint8)
    assert is_anomalous_frame(flash) is True


def test_anomalous_frame_passes_normal_broadcast_palette():
    """A realistic broadcast palette (court tan + crowd + jerseys) is NOT flagged."""
    # Roughly the HSV distribution of a real broadcast frame: ~half
    # court (warm tan, moderately saturated), ~quarter dark crowd,
    # ~quarter mid-saturation jerseys.
    f = np.zeros((600, 800, 3), dtype=np.uint8)
    f[:, :400] = (60, 110, 160)   # tan court (B,G,R)
    f[:, 400:600] = (40, 40, 40)  # dark crowd
    f[:, 600:] = (180, 80, 50)    # blue-ish jerseys
    assert is_anomalous_frame(f) is False


def test_anomalous_frame_handles_empty():
    """Defensive: empty/None frame must not crash."""
    assert is_anomalous_frame(np.zeros((0, 0, 3), dtype=np.uint8)) is False
    assert is_anomalous_frame(None) is False


def test_flash_frame_skips_sample_accumulation():
    """Anomalous-lighting frames must not add to the per-track sample buffer.

    Before this guard, a bright frame's washed-out chest crops would
    pollute each track's rolling median feature and the next normal
    frame's anchored re-assignment would flip players to the wrong team.
    """
    clf = TeamClassifier(
        n_teams=2,
        team_anchors=[TEAM_PROFILES["knicks"], TEAM_PROFILES["sixers"]],
        team_names=["knicks", "sixers"],
    )
    flash = np.full((600, 800, 3), 235, dtype=np.uint8)
    player = _player(track_id=1, bbox=(100, 100, 300, 500))

    initial_skips = clf.n_flash_frames_skipped
    clf.classify(flash, [player])
    assert clf.n_flash_frames_skipped == initial_skips + 1
    # No sample should have been added since the frame was flagged.
    assert 1 not in clf._track_samples or len(clf._track_samples[1]) == 0
