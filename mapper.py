"""Maps player pixel positions to court coordinates via homography.

Connects detection, tracking, and homography into court-space player positions
with temporal smoothing to reduce jitter.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config
from detector import BallDetection, PlayerDetection
from homography import CourtKeypoint, HomographyEngine


@dataclass
class MappedPlayer:
    """A player with both pixel and court coordinate positions."""
    track_id: int
    court_x: float       # feet, 0-94
    court_y: float       # feet, 0-50
    pixel_x: float       # bbox center x
    pixel_y: float       # bbox center y
    bbox: tuple           # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    team_id: int = -1     # -1 = unknown, 0 = team A, 1 = team B, 2 = referee
    jersey_number: Optional[str] = None  # "0".."99" when locked by JerseyVoter
    jersey_locked: bool = False          # True after the vote threshold
    has_ball: bool = False                # True when this player is the
                                          # confirmed possessor for the frame
    player_id: Optional[int] = None       # persistent identity from
                                          # PlayerIdentityRegistry; same number
                                          # collapses to the same player across
                                          # camera cuts. None until both
                                          # team_id and jersey_locked are set.


@dataclass
class MappedBall:
    """Ball with both pixel and court coordinate positions.

    Court coordinates are accurate when the ball is at floor level (loose
    on the court, in a player's hands at hip height). When airborne mid-
    shot/pass, the projected position is biased toward the camera — the
    homography assumes the projected point lies on the floor plane.
    """
    court_x: float
    court_y: float
    pixel_x: float
    pixel_y: float
    bbox: tuple
    confidence: float
    possessor_track_id: Optional[int] = None  # track_id of confirmed holder, or None


class CourtMapper:
    """Maps tracked players to court coordinates using per-frame homography."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = HomographyEngine(
            min_keypoints=config.min_keypoints,
            fallback_frames=config.homography_fallback_frames,
            ransac_threshold=config.ransac_threshold,
            max_reproj_error=config.max_reproj_error,
        )
        self._position_history: Dict[int, deque] = {}

    def map_frame(
        self,
        keypoints: List[CourtKeypoint],
        tracked_players: List[PlayerDetection],
    ) -> Tuple[bool, List[MappedPlayer]]:
        """Map all tracked players to court coordinates for one frame.

        Args:
            keypoints: Detected court keypoints.
            tracked_players: Tracked player detections with IDs.

        Returns:
            (homography_valid, mapped_players) tuple.
        """
        H = self.engine.compute(keypoints)
        if H is None:
            return False, []

        mapped = []
        for player in tracked_players:
            # Use bottom_center (feet position) for court mapping
            court_pos = self.engine.transform_point(
                H, player.bottom_center[0], player.bottom_center[1]
            )
            if court_pos is None:
                continue

            track_id = player.track_id

            # Apply temporal smoothing
            smoothed = self._smooth(track_id, court_pos)

            mapped.append(MappedPlayer(
                track_id=track_id,
                court_x=smoothed[0],
                court_y=smoothed[1],
                pixel_x=player.center[0],
                pixel_y=player.center[1],
                bbox=player.bbox,
                confidence=player.confidence,
                class_name=player.class_name,
            ))

        return True, mapped

    def _smooth(
        self, track_id: int, position: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Apply temporal smoothing (moving average) to a player's position."""
        if track_id not in self._position_history:
            self._position_history[track_id] = deque(
                maxlen=self.config.smoothing_window
            )

        history = self._position_history[track_id]
        history.append(position)

        if len(history) == 1:
            return position

        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def map_ball(
        self,
        ball: Optional[BallDetection],
    ) -> Optional[MappedBall]:
        """Map a ball detection to court coordinates using the latest H.

        Caller is expected to call this AFTER map_frame() so the engine has
        a valid (possibly cached) homography. Returns None when there's no
        ball OR no homography to project with.

        Uses the ball's bottom_center because that's least biased when the
        ball is on the floor or near a player's hands. See MappedBall
        docstring for the airborne-ball caveat.
        """
        if ball is None:
            return None
        H = self.engine.last_homography
        if H is None:
            return None
        court_pos = self.engine.transform_point(
            H, ball.bottom_center[0], ball.bottom_center[1],
        )
        if court_pos is None:
            return None
        return MappedBall(
            court_x=court_pos[0],
            court_y=court_pos[1],
            pixel_x=ball.center[0],
            pixel_y=ball.center[1],
            bbox=ball.bbox,
            confidence=ball.confidence,
        )

    def cleanup_stale_tracks(self, active_track_ids: set):
        """Remove position history for tracks that are no longer active."""
        stale = set(self._position_history.keys()) - active_track_ids
        for tid in stale:
            del self._position_history[tid]

    @property
    def homography_valid(self) -> bool:
        return self.engine.has_valid_homography
