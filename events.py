"""Shot / scoring event detection from action classes the model already returns.

The Roboflow basketball-player-detection-3 model emits per-frame action
predictions (`player-jump-shot`, `player-layup-dunk`, `player-shot-block`,
`ball-in-basket`, `rim`) alongside players and balls. This module turns
those raw per-frame observations into a structured sequence of
**ShotEvent**s: who shot, what kind of shot, where on the court, and
whether it went in.

Pipeline
--------
1. Per frame, the detector returns `actions: List[ActionObservation]`.
   Each action is a bbox + class label.
2. We associate every action bbox to the tracked player whose bbox has the
   highest IoU with it. Below a min-IoU floor we drop the association —
   the action is then in-frame but unassigned (used only for things like
   ball-in-basket where the player attribution is implicit from
   possession).
3. Per-track action history is maintained as a rolling window. A shot is
   *confirmed* when ≥ `shot_confirm_at` of the last `shot_window` frames
   for a track show a shot action. This drops single-frame action
   flickers from the model.
4. When the run of shot-action frames ends, we emit a **ShotEvent**:
       made: any `ball-in-basket` observation within
             `made_window_frames` of the shot's last frame.
       missed: no such observation.
5. The shooter's court coordinates come from their last MappedPlayer
   position during the shot.

What this module does **not** do
--------------------------------
- It doesn't fire the same shot twice in a single continuous run.
- It doesn't depend on the possession tracker — but the pipeline can
  prefer the *possessor's* track_id when both signals agree, which is
  more robust on a contested layup than IoU alone. That cross-check
  lives in the pipeline.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from detector import ActionObservation, BallDetection, PlayerDetection
from mapper import MappedPlayer


# ── Action class taxonomy ──────────────────────────────────────────────────


# Action classes that count as "this player is shooting."
SHOT_ACTION_CLASSES = {"player-jump-shot", "player-layup-dunk"}
# Action classes that count as "a shot just went in," regardless of which
# track the bbox is closest to (the model doesn't assign basket to a player).
SCORE_ACTION_CLASSES = {"ball-in-basket"}
# Defensive plays we tag but don't currently emit events for.
DEFENSIVE_ACTION_CLASSES = {"player-shot-block"}


@dataclass
class ShotEvent:
    """A confirmed shot attempt with outcome."""
    shooter_track_id: int
    shot_type: str           # "jump-shot" | "layup-dunk"
    frame_start: int         # first frame the shot action was observed
    frame_end: int           # last frame the shot action was observed
    made: bool               # True iff ball-in-basket fired within window
    court_x: Optional[float] = None   # shooter's court position at shot time
    court_y: Optional[float] = None
    team_id: int = -1                  # shooter's team_id at shot time
    block_track_id: Optional[int] = None  # if player-shot-block was tagged
                                           # against the same shot, the
                                           # blocker's track_id


@dataclass
class _ShotInProgress:
    """Internal state for a shot that's currently being observed."""
    shooter_track_id: int
    shot_type: str
    frame_start: int
    frame_last: int          # last frame we saw the shot action
    seen_frames: int = 0     # how many frames in the rolling window have
                             # shown the action; promoted via confirm
    confirmed: bool = False
    block_track_id: Optional[int] = None
    last_court_x: Optional[float] = None
    last_court_y: Optional[float] = None
    last_team_id: int = -1


class EventDetector:
    """Per-frame action observations → confirmed ShotEvents."""

    def __init__(
        self,
        shot_window: int = 6,
        shot_confirm_at: int = 3,
        made_window_frames: int = 18,
        end_after_idle_frames: int = 6,
        min_action_iou: float = 0.10,
    ):
        """
        Args:
            shot_window: Size of the per-track rolling buffer of action
                labels. Must be ≥ shot_confirm_at.
            shot_confirm_at: How many of the last `shot_window` frames must
                show a shot action before the shot is confirmed. Drops
                single-frame model flickers.
            made_window_frames: After a shot ends, we look for a
                `ball-in-basket` observation within this many frames to
                call it made. At 30 fps, 18 ≈ 0.6s — about the time the
                ball needs to travel through the rim after the release.
            end_after_idle_frames: A shot-in-progress is considered ended
                after this many consecutive frames with no shot action.
                Closes out the event so we can decide made/missed.
            min_action_iou: Minimum IoU between an action bbox and a
                player bbox for the action to be attributed to that
                track. Below this we treat the action as unattributed.
        """
        if shot_window < shot_confirm_at:
            raise ValueError(
                f"shot_window ({shot_window}) must be ≥ "
                f"shot_confirm_at ({shot_confirm_at})"
            )
        self.shot_window = shot_window
        self.shot_confirm_at = shot_confirm_at
        self.made_window_frames = made_window_frames
        self.end_after_idle_frames = end_after_idle_frames
        self.min_action_iou = min_action_iou

        # Per-track rolling action history. Each entry is the action class
        # name observed for that track this frame, or "" if no action was
        # observed for the track this frame.
        self._track_history: Dict[int, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.shot_window)
        )

        # Shots currently in progress, keyed by track_id.
        self._open_shots: Dict[int, _ShotInProgress] = {}

        # Shots that have ended and are waiting for a possible made-window
        # `ball-in-basket` observation before being emitted.
        # List of (shot, frames_remaining_in_window).
        self._pending: List[Tuple[_ShotInProgress, int]] = []

        # Emitted events (in chronological order).
        self._events: List[ShotEvent] = []

        # Per-track idle counter for closing out shots.
        self._idle_counters: Dict[int, int] = defaultdict(int)

    def update(
        self,
        frame_idx: int,
        actions: List[ActionObservation],
        tracked_players: List[PlayerDetection],
        mapped_players: Optional[List[MappedPlayer]] = None,
    ) -> List[ShotEvent]:
        """Advance one frame; return any shots that resolved this frame.

        Args:
            frame_idx: Absolute frame number (for event timestamps).
            actions: Per-frame action observations from the detector.
            tracked_players: Players with assigned track_ids.
            mapped_players: Optional — if supplied, we'll record the
                shooter's court coordinates and team_id at shot time.

        Returns:
            List of ShotEvent that were *just emitted* this frame. The full
            history is on `self.events`.
        """
        # Map each action obs → most-overlapping tracked player (or None).
        attributed = self._attribute_actions(actions, tracked_players)

        # Per track this frame: did we observe a shot action?
        active_tracks: set = set()
        seen_score_this_frame = any(
            a.class_name in SCORE_ACTION_CLASSES for a in actions
        )
        block_track_this_frame: Optional[int] = None

        for action, track_id in attributed:
            if action.class_name in SHOT_ACTION_CLASSES:
                if track_id is None:
                    continue
                active_tracks.add(track_id)
                self._track_history[track_id].append(action.class_name)
                self._open_or_extend_shot(
                    track_id, action.class_name, frame_idx,
                    mapped_players,
                )
            elif action.class_name in DEFENSIVE_ACTION_CLASSES:
                if track_id is not None:
                    block_track_this_frame = track_id

        # Stamp the latest blocker onto any open shot (it'll get included
        # in the eventual event).
        if block_track_this_frame is not None:
            for shot in self._open_shots.values():
                shot.block_track_id = block_track_this_frame

        # Push an empty history slot for tracks that didn't show a shot
        # action — this is what lets the rolling window drain.
        for tid in list(self._open_shots.keys()):
            if tid not in active_tracks:
                self._track_history[tid].append("")
                self._idle_counters[tid] += 1
            else:
                self._idle_counters[tid] = 0

        # Close out shots whose idle counter passed the threshold.
        ended_now = self._end_idle_shots()

        # Decrement pending-shot windows and resolve any due ones.
        self._tick_pending(seen_score_this_frame)

        return ended_now

    @property
    def events(self) -> List[ShotEvent]:
        """All emitted shot events so far, chronological."""
        return self._events

    # ── Internals ───────────────────────────────────────────────────────────

    def _attribute_actions(
        self,
        actions: List[ActionObservation],
        tracked_players: List[PlayerDetection],
    ) -> List[Tuple[ActionObservation, Optional[int]]]:
        """For each action, find the tracked player with highest IoU."""
        out: List[Tuple[ActionObservation, Optional[int]]] = []
        for action in actions:
            best_tid: Optional[int] = None
            best_iou = self.min_action_iou
            for p in tracked_players:
                if p.track_id < 0:
                    continue
                iou = _bbox_iou(action.bbox, p.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = p.track_id
            out.append((action, best_tid))
        return out

    def _open_or_extend_shot(
        self,
        track_id: int,
        shot_class: str,
        frame_idx: int,
        mapped_players: Optional[List[MappedPlayer]],
    ):
        """Create or extend an in-progress shot for this track."""
        shot = self._open_shots.get(track_id)
        shot_type = (
            "jump-shot" if shot_class == "player-jump-shot" else "layup-dunk"
        )
        if shot is None:
            shot = _ShotInProgress(
                shooter_track_id=track_id,
                shot_type=shot_type,
                frame_start=frame_idx,
                frame_last=frame_idx,
                seen_frames=1,
            )
            self._open_shots[track_id] = shot
        else:
            shot.frame_last = frame_idx
            shot.seen_frames += 1
            # If we previously thought it was a jumper and now see a layup
            # (or vice versa) — prefer the most recent label; it's usually
            # right closer to the rim.
            shot.shot_type = shot_type

        # Confirm if we've seen enough shot-action frames in the window.
        hist = self._track_history[track_id]
        recent_shot_count = sum(
            1 for c in hist if c in SHOT_ACTION_CLASSES
        )
        if recent_shot_count >= self.shot_confirm_at:
            shot.confirmed = True

        # Record location for the eventual event.
        if mapped_players is not None:
            for mp in mapped_players:
                if mp.track_id == track_id:
                    shot.last_court_x = mp.court_x
                    shot.last_court_y = mp.court_y
                    shot.last_team_id = mp.team_id
                    break

    def _end_idle_shots(self) -> List[ShotEvent]:
        """Move shots whose idle counter hit the threshold to pending.

        Unconfirmed shots are discarded — they were just model noise.
        Confirmed shots wait `made_window_frames` for a `ball-in-basket`
        observation; whichever way it resolves we then emit the event.
        """
        emitted: List[ShotEvent] = []
        to_remove: List[int] = []
        for tid, shot in self._open_shots.items():
            if self._idle_counters[tid] < self.end_after_idle_frames:
                continue
            to_remove.append(tid)
            if not shot.confirmed:
                continue
            self._pending.append((shot, self.made_window_frames))

        for tid in to_remove:
            del self._open_shots[tid]
            self._idle_counters[tid] = 0

        return emitted

    def _tick_pending(self, scored_this_frame: bool):
        """Decrement pending shots' made-window and emit those that resolve.

        If a score (`ball-in-basket`) fires this frame, *every* pending
        shot is marked made immediately and emitted. That's a coarse
        association but holds up well: it's rare to have two shots in
        flight at the same time, and the model only fires basket-in
        when the ball actually drops through.
        """
        still_pending: List[Tuple[_ShotInProgress, int]] = []
        for shot, frames_left in self._pending:
            if scored_this_frame:
                self._emit(shot, made=True)
                continue
            if frames_left <= 1:
                self._emit(shot, made=False)
                continue
            still_pending.append((shot, frames_left - 1))
        self._pending = still_pending

    def _emit(self, shot: _ShotInProgress, made: bool):
        self._events.append(ShotEvent(
            shooter_track_id=shot.shooter_track_id,
            shot_type=shot.shot_type,
            frame_start=shot.frame_start,
            frame_end=shot.frame_last,
            made=made,
            court_x=shot.last_court_x,
            court_y=shot.last_court_y,
            team_id=shot.last_team_id,
            block_track_id=shot.block_track_id,
        ))


def _bbox_iou(a: tuple, b: tuple) -> float:
    """Standard bbox IoU. Returns 0 for non-overlapping or degenerate boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)
