"""Visualization: 2D minimap, frame overlays, and composite output."""

from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

from config import Config
from court import COURT_LENGTH, COURT_WIDTH, court_to_minimap, court_to_minimap_half, is_on_half
from court_template import generate_court_image, generate_half_court_image
from detector import BallDetection
from events import ShotEvent
from mapper import MappedBall, MappedPlayer
from scoreboard import Scoreboard

# Visualization constants for the ball.
_BALL_BGR = (40, 140, 255)        # broadcast-orange
_BALL_OUTLINE_BGR = (0, 0, 0)     # black ring for contrast on any background

# Shot-marker colors on the minimap (BGR).
_MADE_BGR = (60, 200, 60)         # green
_MISSED_BGR = (60, 60, 220)       # red

# Scoreboard panel layout (top-right of broadcast frame).
_SCOREBOARD_PAD = 12
_SCOREBOARD_BG = (30, 30, 30)
_SCOREBOARD_TEXT = (240, 240, 240)


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
        mapped_ball: Optional[MappedBall] = None,
        shot_events: Optional[List[ShotEvent]] = None,
    ) -> np.ndarray:
        """Render the minimap with current player positions.

        Args:
            mapped_players: Players with court coordinates.
            homography_valid: Whether the current homography is valid.
            mapped_ball: Optional ball position in court coordinates.
            shot_events: Optional list of past shot events. Made shots
                render as green dots, missed as red. Draws all events to
                this frame so you can see the accumulated shot chart.

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

            # Draw player dot; possessor gets a halo ring to stand out.
            if player.has_ball:
                cv2.circle(img, (px, py), 9, _BALL_BGR, 2)
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

        # Draw shot markers — every made/missed shot to date. Drawn BEFORE
        # the ball so the live ball dot stays on top.
        if shot_events:
            for ev in shot_events:
                if ev.court_x is None or ev.court_y is None:
                    continue
                sx, sy = court_to_minimap(
                    ev.court_x, ev.court_y,
                    self.config.minimap_width, self.config.minimap_height,
                    self.config.minimap_padding,
                )
                color = _MADE_BGR if ev.made else _MISSED_BGR
                if ev.made:
                    cv2.circle(img, (sx, sy), 5, color, -1)
                    cv2.circle(img, (sx, sy), 5, (255, 255, 255), 1)
                else:
                    # X for misses — easy to distinguish at a glance.
                    cv2.line(img, (sx - 4, sy - 4), (sx + 4, sy + 4), color, 2)
                    cv2.line(img, (sx - 4, sy + 4), (sx + 4, sy - 4), color, 2)

        # Draw ball as an orange dot on the court. Drawn last so it sits on
        # top of player dots when they overlap (e.g. ball in possession).
        if mapped_ball is not None:
            bx, by = court_to_minimap(
                mapped_ball.court_x, mapped_ball.court_y,
                self.config.minimap_width, self.config.minimap_height,
                self.config.minimap_padding,
            )
            cv2.circle(img, (bx, by), 4, _BALL_OUTLINE_BGR, -1)
            cv2.circle(img, (bx, by), 3, _BALL_BGR, -1)

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
        mapped_ball: Optional[MappedBall] = None,
        shot_events: Optional[List[ShotEvent]] = None,
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

            if player.has_ball:
                cv2.circle(img, (px, py), 12, _BALL_BGR, 2)
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

        # Draw shot markers (only those on the active half).
        if shot_events:
            for ev in shot_events:
                if (
                    ev.court_x is None
                    or ev.court_y is None
                    or not is_on_half(ev.court_x, side)
                ):
                    continue
                sx, sy = court_to_minimap_half(
                    ev.court_x, ev.court_y, side,
                    self.width, self.height, self.padding,
                )
                color = _MADE_BGR if ev.made else _MISSED_BGR
                if ev.made:
                    cv2.circle(img, (sx, sy), 7, color, -1)
                    cv2.circle(img, (sx, sy), 7, (255, 255, 255), 1)
                else:
                    cv2.line(img, (sx - 6, sy - 6), (sx + 6, sy + 6), color, 2)
                    cv2.line(img, (sx - 6, sy + 6), (sx + 6, sy - 6), color, 2)

        # Draw the ball — only if it falls on the active half (otherwise
        # we'd be plotting at clamped boundary coords and confusing the eye).
        if mapped_ball is not None and is_on_half(mapped_ball.court_x, side):
            bx, by = court_to_minimap_half(
                mapped_ball.court_x, mapped_ball.court_y, side,
                self.width, self.height, self.padding,
            )
            cv2.circle(img, (bx, by), 5, _BALL_OUTLINE_BGR, -1)
            cv2.circle(img, (bx, by), 4, _BALL_BGR, -1)

        return img


class OverlayRenderer:
    """Draws bounding boxes and track IDs on the original video frame.

    Box color reflects TEAM ASSIGNMENT (team 0 / team 1 / unknown), not the
    track ID. This is the at-a-glance signal: same-color boxes = system
    thinks those players are on the same team. Use this to debug team
    classification visually.
    """

    # BGR colors for team_id 0, 1, 2, -1. Picked to be MAXIMALLY distinct
    # so you can read team assignments at a glance — pure saturated blue
    # vs pure saturated red vs bright magenta for unknown. Previously
    # team-0 (200,100,50) and unknown (200,200,200) shared a high blue
    # channel and looked similar on screen, which made T-1 boxes around
    # unclassified players read as "team 0" to the eye.
    _TEAM_BGR = {
        0: (255, 0, 0),        # team 0 → pure blue
        1: (0, 0, 255),        # team 1 → pure red
        2: (0, 220, 220),      # team 2 (refs) → yellow
        -1: (255, 0, 255),     # unknown → bright magenta (impossible to miss)
    }

    def __init__(self, config: Config):
        self.config = config

    # How many frames a "SHOT MADE" / "SHOT MISSED" banner stays visible.
    # At 30 fps, 45 ≈ 1.5s — long enough to read without lingering forever.
    _BANNER_PERSIST_FRAMES = 45

    def render(
        self,
        frame: np.ndarray,
        sv_detections: Optional[sv.Detections],
        mapped_players: Optional[List[MappedPlayer]] = None,
        keypoints=None,
        debug: bool = False,
        ball: Optional[BallDetection] = None,
        shot_events: Optional[List[ShotEvent]] = None,
        current_frame: int = 0,
        scoreboard: Optional[Scoreboard] = None,
        team_labels: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """Draw team-colored boxes + track ID labels + ball + possession ring.

        If a shot event ended within the last `_BANNER_PERSIST_FRAMES`
        frames, also render a green/red banner along the top of the frame.
        If a scoreboard is supplied, render a small score panel in the
        top-right corner.
        """
        annotated = frame.copy()

        if mapped_players:
            for p in mapped_players:
                x1, y1, x2, y2 = [int(v) for v in p.bbox]
                color = self._TEAM_BGR.get(p.team_id, self._TEAM_BGR[-1])
                # Possessor gets a thicker box so it pops on the broadcast feed.
                box_thickness = 4 if p.has_ball else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

                # Mark the possessor explicitly. The 🏀 emoji doesn't render
                # reliably in cv2 fonts so use a plain "BALL" tag.
                if p.has_ball:
                    cv2.rectangle(
                        annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                        _BALL_BGR, 2,
                    )

                label = f"#{p.track_id} T{p.team_id}"
                if p.jersey_number is not None:
                    label = f"#{p.jersey_number} T{p.team_id}"
                if p.has_ball:
                    label = f"BALL {label}"
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

        # Draw the ball as a filled orange disc with a black outline.
        # Drawn AFTER players so it always sits on top of overlapping boxes.
        if ball is not None:
            bx, by = int(ball.center[0]), int(ball.center[1])
            # Radius from bbox; ball bboxes are typically tight, so half the
            # shorter side ≈ ball radius. Clamp to keep it visible in case
            # the detector returns a 1-px ball.
            bw = max(1, int(ball.bbox[2] - ball.bbox[0]))
            bh = max(1, int(ball.bbox[3] - ball.bbox[1]))
            radius = max(6, min(bw, bh) // 2)
            cv2.circle(annotated, (bx, by), radius + 1, _BALL_OUTLINE_BGR, 2)
            cv2.circle(annotated, (bx, by), radius, _BALL_BGR, -1)

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

        # NOTE: a scoreboard panel on the broadcast frame was tried and
        # removed — broadcast video already burns in the real game
        # scoreboard, so a system-drawn one was redundant overlay clutter.
        # The Scoreboard class is still used: its state is saved to
        # output_scoreboard.json for downstream analysis, and the SHOT
        # MADE banner below pulls point values from it.

        # Recent-shot banner — flash MADE/MISSED across the top of the
        # frame for a short window after the event resolves. Picks the
        # most recent event still in-window so simultaneous events
        # (basket + putback miss) don't fight for the banner.
        if shot_events:
            recent = [
                e for e in shot_events
                if 0 <= (current_frame - e.frame_end) <= self._BANNER_PERSIST_FRAMES
            ]
            if recent:
                ev = recent[-1]
                color = _MADE_BGR if ev.made else _MISSED_BGR
                shooter = (
                    f"#{ev.shooter_track_id}"
                    if ev.shooter_track_id >= 0
                    else "?"
                )
                # Look up the point value via the scoreboard's history if
                # available — that's the source of truth (it does the
                # 2-vs-3 geometry check).
                points: Optional[int] = None
                if scoreboard is not None:
                    for s in reversed(scoreboard.history):
                        if s.event is ev:
                            points = s.points
                            break
                if ev.made:
                    text = (
                        f"SHOT MADE +{points}" if points else "SHOT MADE"
                    )
                else:
                    text = "SHOT MISSED"
                full = f"{text}  {shooter}  ({ev.shot_type})"
                (tw, th), _ = cv2.getTextSize(
                    full, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2,
                )
                pad = 12
                cv2.rectangle(
                    annotated, (0, 0), (tw + pad * 2, th + pad * 2),
                    color, -1,
                )
                cv2.putText(
                    annotated, full, (pad, th + pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )

        return annotated

    def _draw_scoreboard(
        self,
        frame: np.ndarray,
        scoreboard: Scoreboard,
        team_labels: Optional[Dict[int, str]],
    ):
        """Render a compact scoreboard panel in the top-right corner.

        Layout: dark panel with team-colored team labels + scores side
        by side. Lives in the broadcast frame (NOT the right-panel
        minimap area) so it stays visible regardless of view mode.

        Mutates `frame` in place.
        """
        state = scoreboard.state
        if not state.score_by_team:
            return

        teams = sorted(state.score_by_team)

        # Build the display strings + colors per team.
        items = []
        for tid in teams:
            label = (
                (team_labels or {}).get(tid)
                or f"T{tid}"
            )
            score = state.score_by_team.get(tid, 0)
            color = self._TEAM_BGR.get(tid, self._TEAM_BGR[-1])
            items.append((label[:5].upper(), score, color))

        # Layout math. Each item is "LABEL  SCORE" with the LABEL in team
        # color and SCORE in plain text. Total panel width: sum of item
        # widths + separator gaps + padding.
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = 0.7
        score_scale = 0.9
        label_thick = 2
        score_thick = 2
        gap = 18  # horizontal gap between teams

        per_item_widths = []
        max_h = 0
        for label, score, _ in items:
            (lw, lh), _ = cv2.getTextSize(label, font, label_scale, label_thick)
            (sw, sh), _ = cv2.getTextSize(str(score), font, score_scale, score_thick)
            item_w = lw + 8 + sw
            per_item_widths.append((lw, sw, item_w, max(lh, sh)))
            max_h = max(max_h, lh, sh)

        panel_w = sum(w for _, _, w, _ in per_item_widths) + gap * (len(items) - 1) + _SCOREBOARD_PAD * 2
        panel_h = max_h + _SCOREBOARD_PAD * 2

        # Anchor top-right with a small inset.
        fh, fw = frame.shape[:2]
        x0 = fw - panel_w - 10
        y0 = 10
        if x0 < 0:
            x0 = 0
        x1 = x0 + panel_w
        y1 = y0 + panel_h

        # Translucent dark background — blend a filled rect so the score
        # is readable against any broadcast color underneath.
        bg = frame[y0:y1, x0:x1].copy()
        cv2.rectangle(bg, (0, 0), (panel_w, panel_h), _SCOREBOARD_BG, -1)
        cv2.addWeighted(bg, 0.75, frame[y0:y1, x0:x1], 0.25, 0, frame[y0:y1, x0:x1])

        # Draw each team.
        cursor = x0 + _SCOREBOARD_PAD
        text_y = y0 + _SCOREBOARD_PAD + max_h - 2
        for (label, score, color), (lw, sw, _item_w, _h) in zip(items, per_item_widths):
            cv2.putText(frame, label, (cursor, text_y), font, label_scale, color, label_thick)
            cursor += lw + 8
            cv2.putText(frame, str(score), (cursor, text_y), font, score_scale, _SCOREBOARD_TEXT, score_thick)
            cursor += sw + gap


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
        ball: Optional[BallDetection] = None,
        mapped_ball: Optional[MappedBall] = None,
        shot_events: Optional[List[ShotEvent]] = None,
        current_frame: int = 0,
        scoreboard: Optional[Scoreboard] = None,
        team_labels: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        annotated = self.overlay.render(
            frame, sv_detections, mapped_players,
            keypoints=keypoints, debug=self.config.debug,
            ball=ball,
            shot_events=shot_events,
            current_frame=current_frame,
            scoreboard=scoreboard,
            team_labels=team_labels,
        )

        right_panel = self._render_right_panel(
            mapped_players, homography_valid, active_half, mapped_ball,
            shot_events,
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
        possessor = next(
            (p for p in mapped_players if p.has_ball), None,
        )
        if possessor is not None:
            label = (
                f"#{possessor.jersey_number}"
                if possessor.jersey_number is not None
                else f"#{possessor.track_id}"
            )
            poss_str = f"BALL: {label} T{possessor.team_id}"
        else:
            poss_str = "BALL: loose"
        text = f"{status} | {len(mapped_players)} pl | half: {half_str} | {poss_str}"
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
        mapped_ball: Optional[MappedBall] = None,
        shot_events: Optional[List[ShotEvent]] = None,
    ) -> np.ndarray:
        """Returns the right-side panel scaled to height=video_height.

        Width is whatever the panel's natural aspect ratio yields. Output dims
        are calibrated once in __init__ so the video sink knows the size up
        front (so we don't need to force-fit each panel into a fixed box,
        which was distorting the half-court template).
        """
        view = self.config.view
        if view == "full":
            mini = self.full.render(
                mapped_players, homography_valid, mapped_ball, shot_events,
            )
        elif view == "half":
            mini = self.half.render(
                active_half, mapped_players, homography_valid, mapped_ball,
                shot_events,
            )
        else:
            # "both" — full on top, half on bottom, stacked at common width
            target_w = self.config.minimap_width
            full_mini = _resize_to_width(
                self.full.render(
                    mapped_players, homography_valid, mapped_ball, shot_events,
                ),
                target_w,
            )
            half_mini = _resize_to_width(
                self.half.render(
                    active_half, mapped_players, homography_valid, mapped_ball,
                    shot_events,
                ),
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
