"""Pipeline orchestrator: ties all modules together for end-to-end processing."""

import json
import csv
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import supervision as sv

from config import Config
from detector import PlayerDetector, CourtKeypointDetector
from tracker import PlayerTracker
from mapper import CourtMapper, MappedPlayer
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

        # Video info (set in run())
        self.video_info = None
        self.renderer = None

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

                # 1. Detect court keypoints
                keypoints = self.court_detector.detect(frame)

                # 2. Detect players
                raw_players = self.player_detector.detect(frame)

                # 3. Track players (assign persistent IDs)
                tracked_players = self.tracker.update(raw_players)

                # 4. Map to court coordinates
                h_valid, mapped_players = self.mapper.map_frame(
                    keypoints, tracked_players
                )

                if h_valid:
                    valid_homography_count += 1

                # 5. Render composite frame
                composite = self.renderer.render(
                    frame,
                    self.tracker.last_sv_detections,
                    mapped_players,
                    h_valid,
                    keypoints=keypoints if self.config.debug else None,
                )

                # 6. Write frame
                sink.write_frame(composite)

                # 7. Collect coordinate data
                frame_data = self._build_frame_record(
                    frame_idx=frame_count,
                    timestamp_ms=frame_count * (1000.0 / self.video_info.fps),
                    mapped_players=mapped_players,
                    homography_valid=h_valid,
                    keypoints_detected=len(keypoints),
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

    def _build_frame_record(
        self,
        frame_idx: int,
        timestamp_ms: float,
        mapped_players: List[MappedPlayer],
        homography_valid: bool,
        keypoints_detected: int,
    ) -> dict:
        return {
            "frame": frame_idx,
            "timestamp_ms": round(timestamp_ms, 1),
            "homography_valid": homography_valid,
            "keypoints_detected": keypoints_detected,
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
