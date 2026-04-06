"""Player tracking with persistent IDs using ByteTrack via supervision."""

from typing import List, Optional

import numpy as np
import supervision as sv

from config import Config
from detector import PlayerDetection


class PlayerTracker:
    """ByteTrack wrapper for persistent player identity across frames."""

    def __init__(self, config: Config):
        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.player_confidence,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self._last_sv_detections: Optional[sv.Detections] = None

    def update(self, detections: List[PlayerDetection]) -> List[PlayerDetection]:
        """Assign persistent track IDs to detected players.

        Args:
            detections: Raw player detections from the current frame.

        Returns:
            Same detections with track_id set (via ordering match to sv.Detections).
        """
        if not detections:
            self._last_sv_detections = sv.Detections.empty()
            return []

        # Build supervision Detections
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
        )

        # Run ByteTrack
        tracked = self.tracker.update_with_detections(sv_dets)
        self._last_sv_detections = tracked

        # Build tracked detection list
        tracked_players = []
        for i in range(len(tracked)):
            bbox = tracked.xyxy[i]
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            # Find the original detection to get class_name
            class_name = "player"
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for d in detections:
                dx, dy = d.center
                if abs(dx - cx) < 5 and abs(dy - cy) < 5:
                    class_name = d.class_name
                    break

            tracked_players.append(PlayerDetection(
                bbox=(x1, y1, x2, y2),
                bottom_center=((x1 + x2) / 2, y2),
                center=(cx, cy),
                confidence=conf,
                class_name=class_name,
            ))

        return tracked_players

    @property
    def last_sv_detections(self) -> Optional[sv.Detections]:
        return self._last_sv_detections
