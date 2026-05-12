"""Visualization: 2D minimap, frame overlays, and composite output."""

from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

from config import Config
from court import COURT_LENGTH, COURT_WIDTH, court_to_minimap, court_to_minimap_half, is_on_half
from court_template import generate_court_image, generate_half_court_image
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

            # Draw jersey number if known, else short track ID
            if player.jersey_number is not None:
                label = f"#{player.jersey_number}"
                # Bold if locked, normal if provisional
                thickness = 2 if player.jersey_locked else 1
            else:
                label = str(player.track_id % 100)
                thickness = 1
            cv2.putText(
                img, label, (px + 7, py + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), thickness,
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


class HalfCourtMinimapRenderer:
    """Renders just the active half of the court at 2x scale.

    Players on the inactive half are skipped. Trails are scoped per side so
    they don't visually leak across a side flip.
    """

    def __init__(self, config: Config):
        self.config = config
        self.width = config.half_minimap_width
        self.height = config.half_minimap_height
        self.padding = config.minimap_padding
        # Pre-render both half templates once
        self._templates = {
            "left": generate_half_court_image("left", self.width, self.height, self.padding),
            "right": generate_half_court_image("right", self.width, self.height, self.padding),
        }
        # Trails per (side, track_id) so a side flip starts fresh
        self._trails: Dict[tuple, deque] = {}
        self.trail_length = config.trail_length

    def render(
        self,
        side: Optional[str],
        mapped_players: List[MappedPlayer],
        homography_valid: bool = True,
    ) -> np.ndarray:
        """Render the half-court minimap for the given side.

        Returns a placeholder ("AWAITING SIGNAL") if `side` is None.
        """
        if side is None:
            img = np.full((self.height, self.width, 3), (60, 60, 60), dtype=np.uint8)
            cv2.putText(
                img, "AWAITING SIGNAL", (self.width // 2 - 110, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2,
            )
            return img

        img = self._templates[side].copy()

        if not homography_valid:
            overlay = np.full_like(img, (80, 80, 80))
            img = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
            cv2.putText(
                img, "NO TRACKING", (self.width // 2 - 80, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            return img

        for player in mapped_players:
            if not is_on_half(player.court_x, side):
                continue
            px, py = court_to_minimap_half(
                player.court_x, player.court_y, side,
                self.width, self.height, self.padding,
            )
            color = MinimapRenderer._get_player_color(self, player)  # reuse logic

            key = (side, player.track_id)
            if key not in self._trails:
                self._trails[key] = deque(maxlen=self.trail_length)
            self._trails[key].append((px, py))

            trail = list(self._trails[key])
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, trail[i - 1], trail[i], trail_color, 1)

            cv2.circle(img, (px, py), 7, color, -1)
            cv2.circle(img, (px, py), 7, (255, 255, 255), 1)
            if player.jersey_number is not None:
                label = f"#{player.jersey_number}"
                thickness = 2 if player.jersey_locked else 1
            else:
                label = str(player.track_id % 100)
                thickness = 1
            cv2.putText(
                img, label, (px + 9, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness,
            )

        return img


class OverlayRenderer:
    """Draws bounding boxes and track IDs on the original video frame.

    Box color reflects TEAM ASSIGNMENT (team 0 / team 1 / unknown), not the
    track ID. This is the at-a-glance signal: same-color boxes = system
    thinks those players are on the same team. Use this to debug team
    classification visually.
    """

    # BGR colors for team_id 0, 1, 2, -1
    _TEAM_BGR = {
        0: (200, 100, 50),    # team 0 → blue-ish
        1: (40, 40, 220),     # team 1 → red-ish
        2: (0, 220, 220),     # team 2 (refs) → yellow
        -1: (200, 200, 200),  # unknown → light gray
    }

    def __init__(self, config: Config):
        self.config = config

    def render(
        self,
        frame: np.ndarray,
        sv_detections: Optional[sv.Detections],
        mapped_players: Optional[List[MappedPlayer]] = None,
        keypoints=None,
        debug: bool = False,
    ) -> np.ndarray:
        """Draw team-colored boxes + track ID labels."""
        annotated = frame.copy()

        if mapped_players:
            for p in mapped_players:
                x1, y1, x2, y2 = [int(v) for v in p.bbox]
                color = self._TEAM_BGR.get(p.team_id, self._TEAM_BGR[-1])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"#{p.track_id} T{p.team_id}"
                if p.jersey_number is not None:
                    label = f"#{p.jersey_number} T{p.team_id}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
                )
                cv2.rectangle(
                    annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                    color, -1,
                )
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

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
    """Combines annotated frame + minimap(s) into a side-by-side composite.

    Layout depends on `Config.view`:
      "full" — annotated broadcast | full-court minimap (legacy)
      "half" — annotated broadcast | half-court minimap (active half)
      "both" — annotated broadcast | full-court (top) + half-court (bottom)
    """

    def __init__(self, config: Config, video_width: int, video_height: int):
        self.config = config
        self.video_width = video_width
        self.video_height = video_height
        self.full = MinimapRenderer(config)
        self.half = HalfCourtMinimapRenderer(config)
        self.overlay = OverlayRenderer(config)
        # Calibrate output dims by doing a dry render. The panel width depends
        # on the view's aspect ratio, which is hard to predict cleanly for the
        # "both" stacked case — easier to measure once.
        self._output_width: int = video_width  # set below
        self._output_height: int = video_height
        self._calibrate()

    def _calibrate(self):
        dummy = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        out = self.render(dummy, None, [], homography_valid=False, active_half=None)
        self._output_height, self._output_width = out.shape[:2]

    def render(
        self,
        frame: np.ndarray,
        sv_detections: Optional[sv.Detections],
        mapped_players: List[MappedPlayer],
        homography_valid: bool,
        active_half: Optional[str] = None,
        keypoints=None,
    ) -> np.ndarray:
        annotated = self.overlay.render(
            frame, sv_detections, mapped_players,
            keypoints=keypoints, debug=self.config.debug,
        )

        right_panel = self._render_right_panel(
            mapped_players, homography_valid, active_half,
        )

        composite = np.hstack([annotated, right_panel])

        # Pad to even dimensions for H.264
        h, w = composite.shape[:2]
        pad_w = (w % 2)
        pad_h = (h % 2)
        if pad_w or pad_h:
            composite = cv2.copyMakeBorder(
                composite, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # Status bar in bottom-right corner of the right panel
        status = "TRACKING" if homography_valid else "NO HOMOGRAPHY"
        half_str = active_half.upper() if active_half else "—"
        text = f"{status} | {len(mapped_players)} players | half: {half_str}"
        cv2.putText(
            composite, text,
            (self.video_width + 10, self.video_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )
        return composite

    def _render_right_panel(
        self,
        mapped_players: List[MappedPlayer],
        homography_valid: bool,
        active_half: Optional[str],
    ) -> np.ndarray:
        """Returns the right-side panel scaled to height=video_height.

        Width is whatever the panel's natural aspect ratio yields. Output dims
        are calibrated once in __init__ so the video sink knows the size up
        front (so we don't need to force-fit each panel into a fixed box,
        which was distorting the half-court template).
        """
        view = self.config.view
        if view == "full":
            mini = self.full.render(mapped_players, homography_valid)
        elif view == "half":
            mini = self.half.render(active_half, mapped_players, homography_valid)
        else:
            # "both" — full on top, half on bottom, stacked at common width
            target_w = self.config.minimap_width
            full_mini = _resize_to_width(
                self.full.render(mapped_players, homography_valid), target_w,
            )
            half_mini = _resize_to_width(
                self.half.render(active_half, mapped_players, homography_valid),
                target_w,
            )
            mini = np.vstack([full_mini, half_mini])
        return _resize_to_height(mini, self.video_height)

    @property
    def output_width(self) -> int:
        return self._output_width

    @property
    def output_height(self) -> int:
        return self._output_height


def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_LINEAR)


def _resize_to_width(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] == w:
        return img
    scale = w / img.shape[1]
    return cv2.resize(img, (w, int(img.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
