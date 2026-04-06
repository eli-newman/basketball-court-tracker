"""Visualization: 2D minimap, frame overlays, and composite output."""

from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

from config import Config
from court import court_to_minimap, COURT_LENGTH, COURT_WIDTH
from court_template import generate_court_image
from mapper import MappedPlayer


class MinimapRenderer:
    """Renders a 2D top-down court minimap with player dots and trails."""

    def __init__(self, config: Config):
        self.config = config
        self.court_base = generate_court_image(
            width=config.minimap_width,
            height=config.minimap_height,
            padding=config.minimap_padding,
        )
        self._trails: Dict[int, deque] = {}
        self.trail_length = config.trail_length

    def render(
        self,
        mapped_players: List[MappedPlayer],
        homography_valid: bool = True,
    ) -> np.ndarray:
        """Render the minimap with current player positions.

        Args:
            mapped_players: Players with court coordinates.
            homography_valid: Whether the current homography is valid.

        Returns:
            BGR image of the minimap.
        """
        img = self.court_base.copy()

        if not homography_valid:
            # Gray out the court and show "NO TRACKING" text
            overlay = np.full_like(img, (80, 80, 80))
            img = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
            cv2.putText(
                img, "NO TRACKING", (img.shape[1] // 2 - 80, img.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            return img

        for player in mapped_players:
            px, py = court_to_minimap(
                player.court_x, player.court_y,
                self.config.minimap_width, self.config.minimap_height,
                self.config.minimap_padding,
            )

            color = self._get_player_color(player)

            # Update trail
            if player.track_id not in self._trails:
                self._trails[player.track_id] = deque(maxlen=self.trail_length)
            self._trails[player.track_id].append((px, py))

            # Draw trail
            trail = list(self._trails[player.track_id])
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, trail[i - 1], trail[i], trail_color, 1)

            # Draw player dot
            cv2.circle(img, (px, py), 5, color, -1)
            cv2.circle(img, (px, py), 5, (255, 255, 255), 1)  # white outline

            # Draw track ID label
            label = str(player.track_id % 100)  # keep it short
            cv2.putText(
                img, label, (px + 7, py + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
            )

        return img

    def _get_player_color(self, player: MappedPlayer) -> tuple:
        if player.team_id == 0:
            return self.config.color_team_a
        elif player.team_id == 1:
            return self.config.color_team_b
        elif player.team_id == 2:
            return self.config.color_referee
        return self.config.color_unknown

    def cleanup_stale_trails(self, active_ids: set):
        """Remove trails for players no longer tracked."""
        stale = set(self._trails.keys()) - active_ids
        for tid in stale:
            del self._trails[tid]


class OverlayRenderer:
    """Draws bounding boxes and track IDs on the original video frame."""

    def __init__(self, config: Config):
        self.config = config
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            text_thickness=1,
            text_scale=0.5,
        )

    def render(
        self,
        frame: np.ndarray,
        sv_detections: Optional[sv.Detections],
        mapped_players: Optional[List[MappedPlayer]] = None,
        keypoints=None,
        debug: bool = False,
    ) -> np.ndarray:
        """Draw detection overlays on the original frame."""
        annotated = frame.copy()

        if sv_detections is not None and len(sv_detections) > 0:
            # Build labels
            labels = []
            for i in range(len(sv_detections)):
                tid = int(sv_detections.tracker_id[i]) if sv_detections.tracker_id is not None else -1
                conf = float(sv_detections.confidence[i]) if sv_detections.confidence is not None else 0
                labels.append(f"#{tid} {conf:.1%}")

            annotated = self.box_annotator.annotate(annotated, sv_detections)
            annotated = self.label_annotator.annotate(annotated, sv_detections, labels)

        # Debug: draw detected court keypoints
        if debug and keypoints:
            for kp in keypoints:
                px, py = int(kp.pixel_x), int(kp.pixel_y)
                cv2.drawMarker(
                    annotated, (px, py), (0, 255, 0),
                    cv2.MARKER_CROSS, 15, 2,
                )
                cv2.putText(
                    annotated, kp.name, (px + 10, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                )

        return annotated


class CompositeRenderer:
    """Combines annotated frame + minimap into a side-by-side composite."""

    def __init__(self, config: Config, video_width: int, video_height: int):
        self.config = config
        self.video_width = video_width
        self.video_height = video_height
        self.minimap = MinimapRenderer(config)
        self.overlay = OverlayRenderer(config)

    def render(
        self,
        frame: np.ndarray,
        sv_detections: Optional[sv.Detections],
        mapped_players: List[MappedPlayer],
        homography_valid: bool,
        keypoints=None,
    ) -> np.ndarray:
        """Create composite frame: annotated video on left, minimap on right."""
        # Annotated original frame
        annotated = self.overlay.render(
            frame, sv_detections, mapped_players,
            keypoints=keypoints, debug=self.config.debug,
        )

        # Minimap
        minimap = self.minimap.render(mapped_players, homography_valid)

        # Scale minimap to match video height
        scale = self.video_height / minimap.shape[0]
        minimap_scaled = cv2.resize(
            minimap,
            (int(minimap.shape[1] * scale), self.video_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Compose side by side
        composite = np.hstack([annotated, minimap_scaled])

        # Add frame info bar at bottom of minimap area
        info_y = self.video_height - 20
        info_x = self.video_width + 10
        status = "TRACKING" if homography_valid else "NO HOMOGRAPHY"
        players_count = len(mapped_players)
        cv2.putText(
            composite, f"{status} | {players_count} players",
            (info_x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

        return composite

    @property
    def output_width(self) -> int:
        scale = self.video_height / self.config.minimap_height
        return self.video_width + int(self.config.minimap_width * scale)

    @property
    def output_height(self) -> int:
        return self.video_height
