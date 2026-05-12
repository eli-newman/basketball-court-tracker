"""Tests for PlayerIdentityRegistry — persistent identity across cuts."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from identity import PlayerIdentityRegistry


def test_resolve_returns_none_when_jersey_not_locked():
    reg = PlayerIdentityRegistry()
    assert reg.resolve(track_id=5, team_id=0, jersey_number=None,
                       jersey_locked=False, frame_idx=10) is None
    assert reg.resolve(track_id=5, team_id=0, jersey_number="11",
                       jersey_locked=False, frame_idx=10) is None


def test_resolve_returns_none_when_team_unknown():
    reg = PlayerIdentityRegistry()
    assert reg.resolve(track_id=5, team_id=-1, jersey_number="11",
                       jersey_locked=True, frame_idx=10) is None


def test_first_locked_jersey_registers_new_player():
    reg = PlayerIdentityRegistry()
    pid = reg.resolve(track_id=5, team_id=0, jersey_number="11",
                      jersey_locked=True, frame_idx=10)
    assert pid == 1
    p = reg.get(pid)
    assert p is not None
    assert p.team_id == 0
    assert p.jersey_number == "11"
    assert p.first_seen_frame == 10


def test_same_track_returns_cached_player_id():
    """Once a track is bound to a player_id, the cache returns it even
    if subsequent calls pass jersey_locked=False (e.g. OCR briefly
    drops below the lock threshold on a noisy frame)."""
    reg = PlayerIdentityRegistry()
    pid_first = reg.resolve(5, 0, "11", True, 10)
    pid_again = reg.resolve(5, 0, None, False, 11)
    assert pid_again == pid_first


def test_different_teams_same_number_get_different_player_ids():
    """Both teams can have a #11. They must register as separate players."""
    reg = PlayerIdentityRegistry()
    pid_a = reg.resolve(5, 0, "11", True, 10)
    pid_b = reg.resolve(7, 1, "11", True, 12)
    assert pid_a != pid_b
    # Both should be registered as separate PersistentPlayer entries.
    assert {p.team_id for p in reg.all_players()} == {0, 1}


def test_jersey_locked_after_cut_collapses_to_same_player():
    """The whole point: ByteTrack reissues track_id after a cut, but the
    same jersey number for the same team must collapse to the same
    persistent player_id."""
    reg = PlayerIdentityRegistry()
    pid_before = reg.resolve(5, 0, "11", True, 10)

    # Camera cut: caller drops the track→player bindings.
    reg.reset_track_bindings()

    # New track id, same player. OCR re-locks to "11".
    pid_after = reg.resolve(99, 0, "11", True, 200)
    assert pid_after == pid_before


def test_reset_track_bindings_preserves_registry():
    reg = PlayerIdentityRegistry()
    reg.resolve(5, 0, "11", True, 10)
    reg.resolve(6, 0, "25", True, 11)
    reg.resolve(7, 1, "21", True, 12)
    assert len(reg.all_players()) == 3

    reg.reset_track_bindings()
    # Registry stays — only the cache was cleared.
    assert len(reg.all_players()) == 3


def test_reset_all_clears_everything():
    reg = PlayerIdentityRegistry()
    reg.resolve(5, 0, "11", True, 10)
    reg.reset_all()
    assert reg.all_players() == []
    # And the next register starts player_id back at 1.
    pid = reg.resolve(5, 0, "11", True, 20)
    assert pid == 1


def test_player_ids_monotonic_in_registration_order():
    reg = PlayerIdentityRegistry()
    a = reg.resolve(1, 0, "11", True, 10)
    b = reg.resolve(2, 0, "25", True, 11)
    c = reg.resolve(3, 1, "21", True, 12)
    assert (a, b, c) == (1, 2, 3)


def test_two_tracks_same_jersey_within_segment_collapse():
    """Even without a cut, ByteTrack sometimes splits a player into a
    new track after a brief occlusion. If both tracks get the same
    locked jersey + team, they should resolve to the same player."""
    reg = PlayerIdentityRegistry()
    pid_a = reg.resolve(5, 0, "11", True, 10)
    pid_b = reg.resolve(42, 0, "11", True, 60)
    assert pid_a == pid_b
