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
from detector import PlayerDetector, CourtKeypointDetector
from tracker import PlayerTracker
from mapper import CourtMapper, MappedPlayer
from team_classifier import TeamClassifier
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
        self.team_classifier = TeamClassifier(n_teams=config.n_teams)
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

        # Video info (set in run())
        self.video_info = None
        self.renderer = None

        # Two-worker pool: player + keypoint detection run concurrently per frame
        self._detector_pool = ThreadPoolExecutor(max_workers=2)

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

                # 1+2. Run court keypoint and player detection in parallel
                # (both are I/O-bound HTTP calls to Roboflow ~1s each).
                keypoints_fut = self._detector_pool.submit(self.court_detector.detect, frame)
                players_fut = self._detector_pool.submit(self.player_detector.detect, frame)
                keypoints = keypoints_fut.result()
                raw_players = players_fut.result()

                # 3. Classify teams by jersey color
                team_ids = self.team_classifier.classify(frame, raw_players)

                # 4. Track players (assign persistent IDs)
                tracked_players = self.tracker.update(raw_players)

                # 5. Map to court coordinates
                h_valid, mapped_players = self.mapper.map_frame(
                    keypoints, tracked_players
                )

                # 6. Assign team IDs to mapped players
                #    Match by bbox proximity since tracker may reorder
                if team_ids and team_ids[0] != -1:
                    self._assign_team_ids(mapped_players, raw_players, team_ids)

                if h_valid:
                    valid_homography_count += 1

                # 6.4 Jersey OCR on unlocked tracks (every Nth frame per track)
                if self.jersey_recognizer is not None:
                    self._update_jersey_numbers(frame, mapped_players, frame_count)

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

        # Print summary
        print()
        print("=" * 60)
        print(f"Done! Processed {processed} frames in {elapsed:.1f}s ({processed / elapsed:.1f} fps)")
        print(f"Homography valid: {valid_homography_count}/{processed} frames ({valid_homography_count / max(processed, 1) * 100:.0f}%)")
        print(f"Output video: {video_out}")
        print(f"Output JSON:  {json_out}")
        print(f"Output CSV:   {csv_out}")
        print("=" * 60)

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

    @staticmethod
    def _assign_team_ids(
        mapped_players: List[MappedPlayer],
        raw_players,
        team_ids: List[int],
    ):
        """Match team_ids from raw detections to mapped players by bbox proximity."""
        for mp in mapped_players:
            best_dist = float("inf")
            best_team = -1
            for raw, tid in zip(raw_players, team_ids):
                dx = mp.pixel_x - raw.center[0]
                dy = mp.pixel_y - raw.center[1]
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_team = tid
            mp.team_id = best_team

    def _build_frame_record(
        self,
        frame_idx: int,
        timestamp_ms: float,
        mapped_players: List[MappedPlayer],
        homography_valid: bool,
        keypoints_detected: int,
        active_half: Optional[str] = None,
    ) -> dict:
        return {
            "frame": frame_idx,
            "timestamp_ms": round(timestamp_ms, 1),
            "homography_valid": homography_valid,
            "keypoints_detected": keypoints_detected,
            "active_half": active_half,
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
                    ])
