"""Configuration for the basketball court tracker."""

from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class Config:
    # Paths
    video_path: str = ""
    output_dir: str = "./output"

    # Roboflow API
    roboflow_api_key: str = ""

    # Player detection
    player_model_id: str = "basketball-player-detection-3-ycjdo/6"
    player_confidence: float = 0.4

    # Court keypoint detection
    court_model_id: str = "basketball-court-detection-2/13"
    court_confidence: float = 0.3

    # Homography
    min_keypoints: int = 4
    homography_fallback_frames: int = 30
    ransac_threshold: float = 5.0
    max_reproj_error: float = 10.0

    # Coordinate smoothing
    smoothing_window: int = 5

    # Processing
    frame_skip: int = 1
    max_frames: int = 0  # 0 = all frames

    # Visualization
    minimap_width: int = 470
    minimap_height: int = 250
    minimap_padding: int = 10
    trail_length: int = 15

    # Colors (BGR)
    color_team_a: tuple = (255, 100, 50)    # blue-ish
    color_team_b: tuple = (50, 50, 255)     # red-ish
    color_unknown: tuple = (200, 200, 200)  # gray
    color_referee: tuple = (0, 200, 200)    # yellow

    # Debug
    debug: bool = False  # draw keypoints + reprojection on frame

    @classmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(description="Basketball Court Tracker")
        parser.add_argument("video", help="Path to input video")
        parser.add_argument("--output", "-o", default="./output", help="Output directory")
        parser.add_argument("--roboflow-key", required=True, help="Roboflow API key")
        parser.add_argument("--player-model", default="basketball-player-detection-3-ycjdo/6")
        parser.add_argument("--court-model", default="basketball-court-detection-2/13")
        parser.add_argument("--player-confidence", type=float, default=0.4)
        parser.add_argument("--court-confidence", type=float, default=0.3)
        parser.add_argument("--min-keypoints", type=int, default=4)
        parser.add_argument("--frame-skip", type=int, default=1)
        parser.add_argument("--max-frames", type=int, default=0)
        parser.add_argument("--smoothing", type=int, default=5, help="Temporal smoothing window")
        parser.add_argument("--debug", action="store_true", help="Draw debug overlays")

        args = parser.parse_args()
        return cls(
            video_path=args.video,
            output_dir=args.output,
            roboflow_api_key=args.roboflow_key,
            player_model_id=args.player_model,
            court_model_id=args.court_model,
            player_confidence=args.player_confidence,
            court_confidence=args.court_confidence,
            min_keypoints=args.min_keypoints,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            smoothing_window=args.smoothing,
            debug=args.debug,
        )

    @classmethod
    def for_colab(
        cls,
        video_path: str,
        roboflow_api_key: str,
        output_dir: str = "/content/drive/MyDrive/basketball_output",
        **kwargs,
    ) -> "Config":
        return cls(
            video_path=video_path,
            output_dir=output_dir,
            roboflow_api_key=roboflow_api_key,
            **kwargs,
        )
