"""Tests for TeamClassifier (occlusion-aware, per-track aggregation)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from detector import PlayerDetection
from team_classifier import (
    TEAM_PROFILES,
    TeamClassifier,
    UNKNOWN_TEAM,
    _nearest_anchor_cosine,
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


def _solid_color_frame(h: int, w: int, bgr: tuple) -> np.ndarray:
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def _frame_with_two_jerseys() -> np.ndarray:
    """A 800x1280 frame: solid red on the left half, solid blue on the right."""
    f = np.zeros((800, 1280, 3), dtype=np.uint8)
    f[:, :640] = (0, 0, 200)   # red BGR
    f[:, 640:] = (200, 0, 0)   # blue BGR
    return f


def test_classifies_two_distinct_jerseys():
    """Two players on different-colored backgrounds → different teams."""
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=2,
        refit_every=1,
        min_samples_to_classify=2,
    )
    frame = _frame_with_two_jerseys()
    # red side: bbox (100, 100, 200, 600). Chest ROI roughly (125, 200, 175, 325).
    # blue side: bbox (900, 100, 1000, 600).
    red_player = _player(track_id=1, bbox=(100, 100, 200, 600))
    blue_player = _player(track_id=2, bbox=(900, 100, 1000, 600))

    # Run several frames to accumulate samples & trigger refit
    for _ in range(3):
        result = clf.classify(frame, [red_player, blue_player])

    assert clf.is_calibrated
    # Two distinct team ids — order isn't guaranteed (KMeans labels arbitrary)
    assert result[0] != result[1]
    assert set(result) == {0, 1}


def test_untracked_players_get_unknown_team():
    """Players with track_id == -1 don't participate in the aggregator."""
    clf = TeamClassifier(n_teams=2, warmup_frames=1, refit_every=1,
                          min_samples_to_classify=1)
    frame = _frame_with_two_jerseys()
    untracked = _player(track_id=-1, bbox=(100, 100, 200, 600))
    out = clf.classify(frame, [untracked])
    assert out == [UNKNOWN_TEAM]


def test_returns_unknown_until_warmup_complete():
    clf = TeamClassifier(n_teams=2, warmup_frames=5, refit_every=1,
                          min_samples_to_classify=1)
    frame = _frame_with_two_jerseys()
    p1 = _player(1, (100, 100, 200, 600))
    p2 = _player(2, (900, 100, 1000, 600))

    for i in range(4):
        out = clf.classify(frame, [p1, p2])
        assert out == [UNKNOWN_TEAM, UNKNOWN_TEAM]
        assert not clf.is_calibrated

    # 5th frame triggers calibration
    out = clf.classify(frame, [p1, p2])
    assert clf.is_calibrated
    assert UNKNOWN_TEAM not in out


def test_overlap_does_not_collapse_two_teams_into_one():
    """The key test: defender (red) with bbox overlapping the offensive player
    (blue) must still be classified as red — the blue pixels inside the
    defender's bbox should NOT leak into the defender's chest sample.
    """
    # Frame: solid red EVERYWHERE except a blue rectangle that fully covers
    # the offensive player's bbox. Defender's bbox includes the right edge
    # of the blue zone (overlap) but most of the defender is still on red.
    f = np.full((800, 1280, 3), (0, 0, 200), dtype=np.uint8)  # red
    f[100:700, 500:900] = (200, 0, 0)  # blue rectangle (offensive player area)

    offensive = _player(track_id=1, bbox=(500, 100, 900, 700))  # bbox ON blue
    # Defender bbox overlaps from x=800 (still in blue) to x=1100 (red)
    defender = _player(track_id=2, bbox=(800, 100, 1100, 700))

    clf = TeamClassifier(n_teams=2, warmup_frames=2, refit_every=1,
                          min_samples_to_classify=2)
    for _ in range(3):
        result = clf.classify(f, [offensive, defender])

    # Each should land in different teams because the occlusion mask kicks
    # the blue overlap pixels out of the defender's chest sample.
    assert result[0] != result[1]


def test_one_track_one_team_assignment_after_many_frames():
    """A track that always sees the same color should get a stable team_id."""
    clf = TeamClassifier(n_teams=2, warmup_frames=2, refit_every=1,
                          min_samples_to_classify=2)
    frame = _frame_with_two_jerseys()
    red = _player(1, (100, 100, 200, 600))
    blue = _player(2, (900, 100, 1000, 600))

    assignments = []
    for _ in range(20):
        out = clf.classify(frame, [red, blue])
        assignments.append(tuple(out))
    # All later frames produce the same labels.
    # (First few may be (-1, -1) during warmup.)
    final = assignments[-10:]
    assert all(a == final[0] for a in final)


def test_team_colors_bgr_returns_centers():
    clf = TeamClassifier(n_teams=2, warmup_frames=1, refit_every=1,
                          min_samples_to_classify=1)
    frame = _frame_with_two_jerseys()
    red = _player(1, (100, 100, 200, 600))
    blue = _player(2, (900, 100, 1000, 600))
    for _ in range(2):
        clf.classify(frame, [red, blue])
    colors = clf.team_colors_bgr
    assert len(colors) == 2
    # All should be valid BGR triples
    for b, g, r in colors:
        assert 0 <= b <= 255 and 0 <= g <= 255 and 0 <= r <= 255


def test_no_players_returns_empty():
    clf = TeamClassifier(n_teams=2, warmup_frames=1)
    assert clf.classify(_frame_with_two_jerseys(), []) == []


# ── Supervised (anchored) mode ──────────────────────────────────────────────


def test_resolve_team_profile_case_insensitive():
    """Team names are stripped + lowercased before lookup."""
    a = resolve_team_profile("Knicks")
    b = resolve_team_profile("  KNICKS  ")
    c = resolve_team_profile("knicks")
    assert a is not None
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_resolve_team_profile_unknown_returns_none():
    assert resolve_team_profile("not-a-team") is None
    assert resolve_team_profile("") is None


def test_all_team_profiles_have_consistent_shape():
    """Every profile is 7-D so the feature vectors line up."""
    for name, profile in TEAM_PROFILES.items():
        assert profile.shape == (7,), f"{name} has wrong shape: {profile.shape}"


def test_nearest_anchor_cosine_picks_obvious_match():
    """Each track's feature vector should snap to its true anchor."""
    anchors = np.stack(
        [TEAM_PROFILES["knicks"], TEAM_PROFILES["sixers"]], axis=0,
    )
    # Synthetic tracks: one looks Knicks-ish, one looks Sixers-ish
    tracks = np.array([
        [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.4],  # blue body + orange + dark → Knicks
        [0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0],  # red dominant → Sixers
    ], dtype=np.float32)
    labels = _nearest_anchor_cosine(tracks, anchors)
    assert labels[0] == 0  # Knicks
    assert labels[1] == 1  # Sixers


def test_nearest_anchor_cosine_handles_zero_vector():
    """A track that observed no saturated pixels gets a deterministic label."""
    anchors = np.stack(
        [TEAM_PROFILES["knicks"], TEAM_PROFILES["sixers"]], axis=0,
    )
    tracks = np.zeros((1, 7), dtype=np.float32)
    labels = _nearest_anchor_cosine(tracks, anchors)
    assert labels[0] in (0, 1)  # doesn't crash; picks one


def test_anchored_classifier_skips_kmeans():
    """In anchored mode, _kmeans is never set; assignments come from cosine."""
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=1,
        refit_every=1,
        min_samples_to_classify=1,
        team_anchors=[
            TEAM_PROFILES["sixers"],  # red → matches the left half of the frame
            TEAM_PROFILES["mavs"],    # blue → matches the right half
        ],
        team_names=["sixers", "mavs"],
    )
    frame = _frame_with_two_jerseys()  # left=red, right=blue
    red_player = _player(1, (100, 100, 200, 600))
    blue_player = _player(2, (900, 100, 1000, 600))

    for _ in range(2):
        out = clf.classify(frame, [red_player, blue_player])

    assert clf.is_calibrated
    assert clf._kmeans is None  # never fit
    # The red player should land on team 0 (sixers), blue on team 1 (mavs).
    assert out[0] == 0
    assert out[1] == 1


def test_anchored_classifier_works_with_single_track():
    """KMeans needs n_teams tracks to fit; anchored mode does not."""
    clf = TeamClassifier(
        n_teams=2,
        warmup_frames=1,
        refit_every=1,
        min_samples_to_classify=1,
        team_anchors=[TEAM_PROFILES["sixers"], TEAM_PROFILES["mavs"]],
        team_names=["sixers", "mavs"],
    )
    frame = _frame_with_two_jerseys()
    red_only = _player(1, (100, 100, 200, 600))

    for _ in range(2):
        out = clf.classify(frame, [red_only])

    # Should be assigned to team 0 (sixers/red) even with no second track.
    assert clf.is_calibrated
    assert out[0] == 0


def test_anchored_classifier_team_colors_bgr_uses_canonical():
    """When team_names are provided, team_colors_bgr returns canonical NBA BGRs."""
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


def test_anchored_classifier_rejects_mismatched_anchor_count():
    """team_anchors length must equal n_teams."""
    import pytest
    with pytest.raises(ValueError):
        TeamClassifier(
            n_teams=2,
            team_anchors=[TEAM_PROFILES["knicks"]],  # only 1
        )


def test_anchored_classifier_rejects_mismatched_names_count():
    import pytest
    with pytest.raises(ValueError):
        TeamClassifier(
            n_teams=2,
            team_anchors=[TEAM_PROFILES["knicks"], TEAM_PROFILES["sixers"]],
            team_names=["knicks"],  # only 1
        )
