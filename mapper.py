"""Maps player pixel positions to court coordinates via homography.

Connects detection, tracking, and homography into court-space player positions
with temporal smoothing to reduce jitter.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config
from detector import PlayerDetection
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

            # Extract track_id from the tracked detection
            # ByteTrack returns detections in order, but we need the ID
            # For now, use bbox hash as pseudo-ID (real ID comes from tracker)
            track_id = _bbox_hash(player.bbox)

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

    def cleanup_stale_tracks(self, active_track_ids: set):
        """Remove position history for tracks that are no longer active."""
        stale = set(self._position_history.keys()) - active_track_ids
        for tid in stale:
            del self._position_history[tid]

    @property
    def homography_valid(self) -> bool:
        return self.engine.has_valid_homography


def _bbox_hash(bbox: tuple) -> int:
    """Generate a simple hash from a bounding box for track matching."""
    return hash((round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1)))
