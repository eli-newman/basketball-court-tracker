"""Persistent player identity from (team_id, jersey_number).

ByteTrack reissues `track_id`s on every camera cut, and even within a
continuous segment it occasionally splits a player into two tracks
after a brief occlusion. The jersey number is the same human across
all of those, so collapsing on (team_id, jersey_number) gives us a
stable identity that survives both.

Lifecycle
---------
- A track only enters the registry once its jersey number is **locked**
  by the `JerseyVoter` (3+ consistent OCR reads) AND its team is known.
  Provisional/unknown values are ignored — registering on noise would
  permanently bind a track to the wrong player.
- `resolve(track_id, ...)` is called every frame for every mapped player.
  On the first locked-jersey frame for a track, the (team, number) pair
  is looked up (or registered if new), and the resulting `player_id` is
  cached against the track_id. Subsequent frames return the cached id
  without re-walking the registry.
- `reset_track_bindings()` is called on camera cuts. The (team, number)
  → player_id registry survives, but the track_id → player_id cache is
  cleared because ByteTrack is about to reissue every id.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PersistentPlayer:
    """A player identity that's stable for the lifetime of one clip."""
    player_id: int           # 1-indexed, monotonic
    team_id: int
    jersey_number: str       # "0".."99"
    first_seen_frame: int    # frame_idx where this player was first registered


class PlayerIdentityRegistry:
    """Collapses (team_id, jersey_number) → stable player_id across cuts.

    The registry is append-only within a clip: once a (team, number) pair
    is observed it keeps the same player_id forever, even if that track
    disappears, the camera cuts, and the player reappears as a fresh
    track later.
    """

    def __init__(self):
        # (team_id, jersey_number) → PersistentPlayer
        self._players: Dict[Tuple[int, str], PersistentPlayer] = {}
        # track_id → player_id, cleared on cut (track ids are about to be reissued)
        self._track_to_player: Dict[int, int] = {}
        self._next_id: int = 1

    def resolve(
        self,
        track_id: int,
        team_id: int,
        jersey_number: Optional[str],
        jersey_locked: bool,
        frame_idx: int,
    ) -> Optional[int]:
        """Return the persistent player_id for this track, or None if unknown.

        - Already-cached track_id → return immediately. Cheap and stable.
        - Otherwise we need a *locked* jersey number AND a known team to
          either register a new player or look up an existing one. Loose
          (provisional / no-team) reads return None to avoid binding the
          track to a wrong identity.
        """
        cached = self._track_to_player.get(track_id)
        if cached is not None:
            return cached

        if not jersey_locked or jersey_number is None or team_id < 0:
            return None

        key = (team_id, jersey_number)
        existing = self._players.get(key)
        if existing is not None:
            self._track_to_player[track_id] = existing.player_id
            return existing.player_id

        # New player → register.
        new = PersistentPlayer(
            player_id=self._next_id,
            team_id=team_id,
            jersey_number=jersey_number,
            first_seen_frame=frame_idx,
        )
        self._players[key] = new
        self._track_to_player[track_id] = new.player_id
        self._next_id += 1
        return new.player_id

    def get(self, player_id: int) -> Optional[PersistentPlayer]:
        """Look up a player by their stable id."""
        for p in self._players.values():
            if p.player_id == player_id:
                return p
        return None

    def all_players(self) -> List[PersistentPlayer]:
        """Every player registered so far, in registration order."""
        return sorted(self._players.values(), key=lambda p: p.player_id)

    def reset_track_bindings(self):
        """Drop the track_id cache. Call this on every camera cut — the
        next frame's track ids will be unrelated to the previous frame's.
        The (team, number) → player_id registry is preserved so jerseys
        seen in earlier segments still collapse to the same player.
        """
        self._track_to_player.clear()

    def reset_all(self):
        """Forget every player. Use only when the whole clip restarts."""
        self._players.clear()
        self._track_to_player.clear()
        self._next_id = 1
