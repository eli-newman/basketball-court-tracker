"""Pipeline orchestrator: ties all modules together for end-to-end processing."""

import json
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import supervision as sv

from config import Config
from cut_detector import CameraCutDetector
from detector import PlayerDetector, CourtKeypointDetector
from tracker import PlayerTracker
from mapper import CourtMapper, MappedBall, MappedPlayer
from possession import PossessionTracker
from events import EventDetector, ShotEvent
from geo_shot import GeometricShotDetector
from identity import PlayerIdentityRegistry
from player_stats import PlayerStatsAggregator
from scoreboard import Scoreboard
from team_classifier import TeamClassifier, resolve_team_profile
from jersey import (
    JerseyNumberRecognizer, JerseyRead, JerseyVoter,
    crop_chest, parse_jersey_response,
)
from view_selector import ActiveHalfSelector
from visualizer import CompositeRenderer


class Pipeline:
    """End-to-end basketball court tracking pipeline."""

    def __init__(self, config: Config):
        self.config = config

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize modules
        self.player_detector = PlayerDetector(config)
        self.court_detector = CourtKeypointDetector(config)
        self.tracker = PlayerTracker(config)
        self.mapper = CourtMapper(config)
        self.team_classifier = self._build_team_classifier(config)
        self.possession = PossessionTracker(
            max_distance_px=config.possession_max_distance_px,
            confirm_at=config.possession_confirm_at,
            release_after_missing=config.possession_release_after_missing,
        )
        self.events = EventDetector(
            shot_window=config.shot_window,
            shot_confirm_at=config.shot_confirm_at,
            made_window_frames=config.made_window_frames,
        )
        # Geometric fallback: emits made-shot events when the ball passes
        # through a rim bbox from above. Catches shots on model versions
        # that don't emit the action classes EventDetector relies on.
        self.geo_shot = GeometricShotDetector()
        # Separate list for geometric shots so we can show them in the
        # banner / scoreboard while keeping the action-class events list
        # clean for inspection.
        self._geo_events: List[ShotEvent] = []
        self.scoreboard = Scoreboard(n_teams=config.n_teams)
        self.cut_detector = CameraCutDetector(
            cut_threshold=config.cut_threshold,
        )
        self._cuts_seen: int = 0
        self.half_selector = ActiveHalfSelector(
            history_frames=config.half_hysteresis_frames,
        )

        # Jersey OCR (opt-in; expensive enough to be off by default)
        self.jersey_recognizer: Optional[JerseyNumberRecognizer] = None
        self.jersey_voter: Optional[JerseyVoter] = None
        self._jersey_sample_offset: dict = {}  # track_id -> next sample frame
        if config.enable_jersey_ocr:
            self.jersey_recognizer = JerseyNumberRecognizer(config)
            self.jersey_voter = JerseyVoter(
                confirm_at=config.jersey_confirm_at,
                min_confidence=config.jersey_min_confidence,
            )

        # Persistent identity: collapses (team_id, jersey_number) → stable
        # player_id across camera cuts. Always present (cheap), but only
        # populates when jersey OCR is enabled and locks a number.
        self.identity = PlayerIdentityRegistry()

        # Per-player stats (PTS / FGM / FGA) keyed by persistent player_id.
        # Driven by the ShotEvent stream same as Scoreboard; renders a
        # "top scorers" panel when we have identified shooters.
        self.player_stats = PlayerStatsAggregator()

        # Video info (set in run())
        self.video_info = None
        self.renderer = None

        # Two-worker pool: player + keypoint detection run concurrently per frame
        # Worker pool sized for parallel detection + OCR. Player + court
        # detection use 2 of these slots per frame; the rest are
        # available for jersey-OCR HTTP calls (one per unlocked track).
        # 12 workers means even 10 unlocked tracks finish their OCR calls
        # in a single round-trip instead of 5 batches of 2.
        self._detector_pool = ThreadPoolExecutor(max_workers=12)

    @staticmethod
    def _build_team_classifier(config: Config) -> TeamClassifier:
        """Construct the team classifier, switching to anchored mode when both
        --team-a and --team-b resolve to known canonical color profiles.

        Falls back to KMeans if either name is missing or unrecognised — we
        warn but don't crash so the pipeline still runs on a clip where the
        user didn't specify a matchup.
        """
        if not config.team_a or not config.team_b:
            return TeamClassifier(n_teams=config.n_teams)

        anchor_a = resolve_team_profile(config.team_a)
        anchor_b = resolve_team_profile(config.team_b)
        if anchor_a is None or anchor_b is None:
            missing = []
            if anchor_a is None:
                missing.append(config.team_a)
            if anchor_b is None:
                missing.append(config.team_b)
            print(
                f"[team-classifier] Unknown team name(s): {missing}. "
                "Falling back to unsupervised KMeans. See TEAM_PROFILES in "
                "team_classifier.py for the list of recognised names."
            )
            return TeamClassifier(n_teams=config.n_teams)

        print(
            f"[team-classifier] Anchored mode: "
            f"team 0 = {config.team_a}, team 1 = {config.team_b}"
        )
        return TeamClassifier(
            n_teams=2,
            team_anchors=[anchor_a, anchor_b],
            team_names=[config.team_a, config.team_b],
            debug_crop_dir=config.team_classifier_debug_crops,
        )

    def run(self):
        """Process the full video and generate outputs."""
        # Open video
        self.video_info = sv.VideoInfo.from_video_path(self.config.video_path)
        print(f"Input: {self.config.video_path}")
        print(f"Resolution: {self.video_info.width}x{self.video_info.height}")
        print(f"FPS: {self.video_info.fps}")
        print(f"Total frames: {self.video_info.total_frames}")
        print()

        # Initialize renderer
        self.renderer = CompositeRenderer(
            self.config, self.video_info.width, self.video_info.height
        )

        # Output paths
        video_out = os.path.join(self.config.output_dir, "output_composite.mp4")
        json_out = os.path.join(self.config.output_dir, "output_coordinates.json")
        csv_out = os.path.join(self.config.output_dir, "output_coordinates.csv")

        # Set up video writer
        output_info = sv.VideoInfo(
            width=self.renderer.output_width,
            height=self.renderer.output_height,
            fps=self.video_info.fps,
        )

        all_frame_data = []
        frame_count = 0
        processed = 0
        valid_homography_count = 0
        start_time = time.time()

        frame_gen = sv.get_video_frames_generator(
            self.config.video_path,
            stride=self.config.frame_skip,
        )

        with sv.VideoSink(video_out, output_info) as sink:
            for frame in frame_gen:
                frame_count += 1

                if self.config.max_frames > 0 and processed >= self.config.max_frames:
                    break

                # 0. Camera cut detection FIRST. If this frame is the first
                # of a new shot, wipe per-shot state (tracker, possession,
                # in-progress shots, ball history, minimap trails,
                # homography cache) so the new shot starts fresh and we
                # don't try to match new players to stale track IDs.
                if self.cut_detector.update(frame):
                    self._cuts_seen += 1
                    self.tracker.reset()
                    self.possession.reset()
                    self.events.reset_shot_progress()
                    self.geo_shot.reset()
                    self.mapper.engine.reset()
                    # Team classifier: drop per-track sample buffers so
                    # the next "track 4" doesn't inherit the previous
                    # "track 4"'s color samples. Keeps fitted anchors.
                    self.team_classifier.reset_track_state()
                    # Identity: drop track→player bindings (ByteTrack will
                    # reissue ids), but keep the (team, number)→player
                    # registry so jerseys from prior segments still
                    # collapse to the same player when they reappear.
                    self.identity.reset_track_bindings()
                    self.renderer.reset_for_cut()
                    print(
                        f"[cut] frame {frame_count}: camera cut detected "
                        f"(distance={self.cut_detector.last_distance:.2f})"
                    )

                # 1+2. Run court keypoint and player detection in parallel
                # (both are I/O-bound HTTP calls to Roboflow ~1s each). The
                # player call also returns ball + action observations from
                # the same response — free, since it's the same model.
                keypoints_fut = self._detector_pool.submit(self.court_detector.detect, frame)
                players_fut = self._detector_pool.submit(self.player_detector.detect_all, frame)
                keypoints = keypoints_fut.result()
                raw_players, ball, actions = players_fut.result()

                # 3. Track players (assign persistent IDs first — the team
                #    classifier aggregates per track_id, so it needs them).
                tracked_players = self.tracker.update(raw_players)

                # 4. Classify teams by jersey color (track-aware, occlusion-
                #    masked). Returns team_id per tracked player in order.
                team_ids = self.team_classifier.classify(frame, tracked_players)
                for tp, tid in zip(tracked_players, team_ids):
                    tp.class_name = tp.class_name  # noop — keep for clarity
                team_by_track = {
                    tp.track_id: t for tp, t in zip(tracked_players, team_ids)
                }

                # 5. Map to court coordinates
                h_valid, mapped_players = self.mapper.map_frame(
                    keypoints, tracked_players
                )

                # 6. Stamp team_id onto mapped players (looked up by track_id —
                #    no more brittle bbox-proximity matching).
                for mp in mapped_players:
                    mp.team_id = team_by_track.get(mp.track_id, -1)

                if h_valid:
                    valid_homography_count += 1

                # 6.2 Possession: which tracked player has the ball this frame.
                #     Run against the tracked players (with track_ids), not
                #     mapped players, because the ball signal is in pixel space.
                possession = self.possession.update(ball, tracked_players)
                if possession.possessor_track_id is not None:
                    for mp in mapped_players:
                        mp.has_ball = (mp.track_id == possession.possessor_track_id)

                # 6.3 Map the ball to court coords (uses cached H, so fine
                #     even when the current frame had no fresh keypoints).
                mapped_ball = self.mapper.map_ball(ball)
                if mapped_ball is not None:
                    mapped_ball.possessor_track_id = possession.possessor_track_id

                # 6.35 Event detection: shot attempts + makes from action
                #      classes the model returned this frame.
                self.events.update(
                    frame_idx=frame_count,
                    actions=actions,
                    tracked_players=tracked_players,
                    mapped_players=mapped_players,
                )

                # 6.36 Geometric shot detection: catches makes that the
                #      action-class detector misses (e.g. on models that
                #      don't emit player-jump-shot / ball-in-basket).
                new_geo = self.geo_shot.update(
                    frame_idx=frame_count,
                    ball=ball,
                    actions=actions,
                    mapped_players=mapped_players,
                    possessor_track_id=possession.possessor_track_id,
                )
                if new_geo:
                    self._geo_events.extend(new_geo)

                # 6.37 Scoreboard: combined event list from BOTH detectors
                #      (action-class + geometric). Scoreboard internally
                #      dedupes by object identity, so we can rebuild the
                #      list each frame without double-counting.
                combined_events = self.events.events + self._geo_events
                self.scoreboard.update(combined_events)
                self.player_stats.update(combined_events)

                # 6.4 Jersey OCR on unlocked tracks (every Nth frame per track)
                if self.jersey_recognizer is not None:
                    self._update_jersey_numbers(frame, mapped_players, frame_count)

                # 6.45 Resolve persistent player identity for every mapped
                #      player. No-op until both team and jersey are known;
                #      from then on, the same human gets the same player_id
                #      across cuts so stats can be aggregated per player.
                for mp in mapped_players:
                    mp.player_id = self.identity.resolve(
                        track_id=mp.track_id,
                        team_id=mp.team_id,
                        jersey_number=mp.jersey_number,
                        jersey_locked=mp.jersey_locked,
                        frame_idx=frame_count,
                    )

                # 6.5 Pick active half from court keypoints (with hysteresis)
                active_half = self.half_selector.update(keypoints)

                # 7. Render composite frame
                composite = self.renderer.render(
                    frame,
                    self.tracker.last_sv_detections,
                    mapped_players,
                    h_valid,
                    active_half=active_half,
                    keypoints=keypoints if self.config.debug else None,
                    ball=ball,
                    mapped_ball=mapped_ball,
                    shot_events=combined_events,
                    current_frame=frame_count,
                    scoreboard=self.scoreboard,
                    team_labels=self._team_labels_map(),
                    player_stats=self.player_stats,
                    identity=self.identity,
                )

                # 8. Write frame
                sink.write_frame(composite)

                # 9. Collect coordinate data
                frame_data = self._build_frame_record(
                    frame_idx=frame_count,
                    timestamp_ms=frame_count * (1000.0 / self.video_info.fps),
                    mapped_players=mapped_players,
                    homography_valid=h_valid,
                    keypoints_detected=len(keypoints),
                    active_half=active_half,
                    mapped_ball=mapped_ball,
                    possessor_track_id=possession.possessor_track_id,
                )
                all_frame_data.append(frame_data)

                processed += 1
                if processed % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = processed / elapsed if elapsed > 0 else 0
                    print(
                        f"  Frame {processed}/{self.config.max_frames or 'all'} "
                        f"| {fps:.1f} fps | keypoints: {len(keypoints)} "
                        f"| players: {len(mapped_players)} "
                        f"| H valid: {h_valid}"
                    )

        elapsed = time.time() - start_time

        # Save coordinate data
        self._save_json(all_frame_data, json_out)
        self._save_csv(all_frame_data, csv_out)

        # Save shot/event log + scoreboard (combined: action-class + geo)
        combined_events_final = self.events.events + self._geo_events
        events_out = os.path.join(self.config.output_dir, "output_events.json")
        self._save_events(combined_events_final, events_out)
        scoreboard_out = os.path.join(self.config.output_dir, "output_scoreboard.json")
        self._save_scoreboard(scoreboard_out)
        # Per-player stats (only meaningful when jersey OCR identified some
        # shooters; otherwise the file is `{"players": []}`).
        player_stats_out = os.path.join(
            self.config.output_dir, "output_player_stats.json",
        )
        self._save_player_stats(player_stats_out)

        # Print summary
        n_events = len(combined_events_final)
        made = sum(1 for e in combined_events_final if e.made)
        n_action = len(self.events.events)
        n_geo = len(self._geo_events)
        score_state = self.scoreboard.state
        score_summary = " - ".join(
            f"{self._team_label(t)} {score_state.score_by_team.get(t, 0)}"
            for t in sorted(score_state.score_by_team)
        )
        print()
        print("=" * 60)
        print(f"Done! Processed {processed} frames in {elapsed:.1f}s ({processed / elapsed:.1f} fps)")
        print(f"Homography valid: {valid_homography_count}/{processed} frames ({valid_homography_count / max(processed, 1) * 100:.0f}%)")
        print(f"Camera cuts detected: {self._cuts_seen}")
        print(f"Shot events: {n_events} ({made} made, {n_events - made} missed) "
              f"[{n_action} from action classes, {n_geo} geometric]")
        print(f"Final score: {score_summary}")
        print(f"Output video:      {video_out}")
        print(f"Output JSON:       {json_out}")
        print(f"Output CSV:        {csv_out}")
        print(f"Output events:     {events_out}")
        print(f"Output scoreboard: {scoreboard_out}")
        print(f"Output player-stats: {player_stats_out}")
        # Per-player scoring summary (when jersey OCR populated identities).
        top = self.player_stats.top_scorers(n=5)
        if top:
            print("Top scorers:")
            for s in top:
                p = self.identity.get(s.player_id)
                label = f"#{p.jersey_number}" if p else f"P{s.player_id}"
                team = self._team_label(s.team_id) if s.team_id >= 0 else "??"
                print(
                    f"  {team:>5}  {label:>5}  "
                    f"{s.points:>3} PTS  {s.fgm}/{s.fga}  "
                    f"({s.fg_pct * 100:.0f}%)"
                )
        print("=" * 60)
        # Detector class histogram — surfaces "no action classes ever fired"
        # bugs that otherwise look like "shot detection is broken."
        self.player_detector.print_class_histogram()

    def _team_labels_map(self) -> dict:
        """Map of team_id → display label for each known team.

        Empty when no team names were configured (KMeans mode); the
        visualizer falls back to "T0"/"T1" in that case.
        """
        return {
            tid: self._team_label(tid) for tid in range(self.config.n_teams)
        }

    def _team_label(self, team_id: int) -> str:
        """Display name for a team in the summary line.

        Uses the canonical team name from the classifier when available
        (e.g. "knicks"), falls back to "T<n>" otherwise.
        """
        names = getattr(self.team_classifier, "_team_names", None)
        if names is not None and 0 <= team_id < len(names):
            return names[team_id].upper()[:5]
        return f"T{team_id}"

    def _save_scoreboard(self, path: str):
        """Persist final score + per-shot breakdown to JSON."""
        history = self.scoreboard.history
        payload = {
            "score_by_team": self.scoreboard.state.score_by_team,
            "team_labels": {
                i: self._team_label(i) for i in self.scoreboard.state.score_by_team
            },
            "unattributed_points": self.scoreboard.state.unattributed_points,
            "shots": [
                {
                    "shooter_track_id": s.event.shooter_track_id,
                    "team_id": s.event.team_id,
                    "shot_type": s.event.shot_type,
                    "made": s.event.made,
                    "points": s.points,
                    "rim_side": s.rim_side,
                    "frame_end": s.event.frame_end,
                    "court_x": (
                        round(s.event.court_x, 2) if s.event.court_x is not None else None
                    ),
                    "court_y": (
                        round(s.event.court_y, 2) if s.event.court_y is not None else None
                    ),
                }
                for s in history
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _save_player_stats(self, path: str):
        """Persist per-player counting stats to JSON.

        Empty `players` list when no jersey-locked shooter was ever
        attributed to a shot (i.e. OCR off, or off-screen at shot time).
        """
        rows = self.player_stats.all()
        payload = {
            "players": [
                {
                    "player_id": s.player_id,
                    "team_id": s.team_id,
                    "team": (
                        self._team_label(s.team_id) if s.team_id >= 0 else None
                    ),
                    "jersey_number": (
                        self.identity.get(s.player_id).jersey_number
                        if self.identity.get(s.player_id) is not None
                        else None
                    ),
                    "points": s.points,
                    "fga": s.fga,
                    "fgm": s.fgm,
                    "fg_pct": round(s.fg_pct, 3),
                    "fg2a": s.fg2a,
                    "fg2m": s.fg2m,
                    "fg3a": s.fg3a,
                    "fg3m": s.fg3m,
                    "fg3_pct": round(s.fg3_pct, 3),
                }
                for s in rows
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _save_events(self, events: List[ShotEvent], path: str):
        """Persist the shot-event log as JSON. One entry per attempt."""
        payload = [
            {
                "shooter_track_id": e.shooter_track_id,
                "shooter_player_id": e.shooter_player_id,
                "team_id": e.team_id,
                "shot_type": e.shot_type,
                "made": e.made,
                "frame_start": e.frame_start,
                "frame_end": e.frame_end,
                "court_x": (
                    round(e.court_x, 2) if e.court_x is not None else None
                ),
                "court_y": (
                    round(e.court_y, 2) if e.court_y is not None else None
                ),
                "block_track_id": e.block_track_id,
            }
            for e in events
        ]
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _update_jersey_numbers(
        self,
        frame: np.ndarray,
        mapped_players: List[MappedPlayer],
        frame_count: int,
    ):
        """OCR jersey numbers on tracks not yet locked, then stamp results."""
        recognizer = self.jersey_recognizer
        voter = self.jersey_voter
        if recognizer is None or voter is None:
            return

        sample_every = max(1, self.config.jersey_sample_every)

        # Sample unlocked tracks; reuse already-locked numbers without re-OCR
        chest_crops = []
        sampled_track_ids = []
        for p in mapped_players:
            existing = voter.current(p.track_id)
            if existing and existing.locked:
                continue  # already locked, skip OCR
            next_sample = self._jersey_sample_offset.get(p.track_id, 0)
            if frame_count < next_sample:
                continue
            crop = crop_chest(frame, p.bbox)
            if crop.size == 0:
                continue
            chest_crops.append(crop)
            sampled_track_ids.append(p.track_id)
            self._jersey_sample_offset[p.track_id] = frame_count + sample_every

        # OCR each sampled crop in parallel; submit votes
        if chest_crops:
            futures = [
                self._detector_pool.submit(recognizer.read, c) for c in chest_crops
            ]
            for tid, fut in zip(sampled_track_ids, futures):
                resp = fut.result()
                read = parse_jersey_response(resp)
                if read is None:
                    continue
                read.track_id = tid
                voter.submit(read)

        # Stamp every mapped player with whatever the voter currently has
        for p in mapped_players:
            a = voter.current(p.track_id)
            if a is not None:
                p.jersey_number = a.number
                p.jersey_locked = a.locked

    def _build_frame_record(
        self,
        frame_idx: int,
        timestamp_ms: float,
        mapped_players: List[MappedPlayer],
        homography_valid: bool,
        keypoints_detected: int,
        active_half: Optional[str] = None,
        mapped_ball: Optional[MappedBall] = None,
        possessor_track_id: Optional[int] = None,
    ) -> dict:
        ball_record: Optional[dict] = None
        if mapped_ball is not None:
            ball_record = {
                "court_x": round(mapped_ball.court_x, 2),
                "court_y": round(mapped_ball.court_y, 2),
                "pixel_x": round(mapped_ball.pixel_x, 1),
                "pixel_y": round(mapped_ball.pixel_y, 1),
                "confidence": round(mapped_ball.confidence, 3),
                "possessor_track_id": possessor_track_id,
            }
        return {
            "frame": frame_idx,
            "timestamp_ms": round(timestamp_ms, 1),
            "homography_valid": homography_valid,
            "keypoints_detected": keypoints_detected,
            "active_half": active_half,
            "possessor_track_id": possessor_track_id,
            "ball": ball_record,
            "players": [
                {
                    "track_id": p.track_id,
                    "court_x": round(p.court_x, 2),
                    "court_y": round(p.court_y, 2),
                    "pixel_x": round(p.pixel_x, 1),
                    "pixel_y": round(p.pixel_y, 1),
                    "confidence": round(p.confidence, 3),
                    "class_name": p.class_name,
                    "team_id": p.team_id,
                    "jersey_number": p.jersey_number,
                    "jersey_locked": p.jersey_locked,
                    "player_id": p.player_id,
                    "has_ball": p.has_ball,
                }
                for p in mapped_players
            ],
        }

    def _save_json(self, data: list, path: str):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, data: list, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "timestamp_ms", "homography_valid",
                "track_id", "court_x", "court_y",
                "pixel_x", "pixel_y", "confidence", "class_name", "team_id",
                "has_ball",
            ])
            for frame_data in data:
                for p in frame_data["players"]:
                    writer.writerow([
                        frame_data["frame"],
                        frame_data["timestamp_ms"],
                        frame_data["homography_valid"],
                        p["track_id"],
                        p["court_x"],
                        p["court_y"],
                        p["pixel_x"],
                        p["pixel_y"],
                        p["confidence"],
                        p["class_name"],
                        p["team_id"],
                        p.get("has_ball", False),
                    ])
