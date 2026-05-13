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
    # "hosted" = POST to `inference_host` (default detect.roboflow.com).
    #            Same protocol works against any Roboflow inference-server
    #            container (e.g. friend's DGX Spark exposed over Tailscale).
    # "local"  = run model in-process via the `inference` package (needs
    #            `inference` or `inference-gpu`; ~30 fps on a Colab T4 GPU).
    inference_backend: str = "hosted"
    # HTTP base URL for the "hosted" backend. Override to point at a self-
    # hosted inference server, e.g. http://spark.tailnet:9001
    inference_host: str = "https://detect.roboflow.com"

    # Player detection
    player_model_id: str = "basketball-player-detection-3-ycjdo/6"
    player_confidence: float = 0.4
    # NMS only drops near-duplicate boxes (same player detected twice).
    # 0.85 keeps stacked-player pairs (post-ups, screens, tight defense) where
    # bboxes legitimately overlap 0.5-0.8. 0.5 was killing real defenders.
    nms_iou_threshold: float = 0.85
    # Sanity filters applied at detection time (before tracking) so phantom
    # detections (people in the crowd, coaches on the sideline) don't pollute
    # the tracker's ID assignment.
    min_player_bbox_height: int = 30   # pixels — drops tiny far-distance/crowd boxes
    min_player_aspect_ratio: float = 1.0   # height/width — drops wide horizontal blobs

    # Court keypoint detection
    court_model_id: str = "basketball-court-detection-2/13"
    court_confidence: float = 0.3

    # Jersey number OCR (opt-in — adds ~1 API call per tracked player per
    # `jersey_sample_every` frames, so it costs real wall time).
    enable_jersey_ocr: bool = False
    # Workspace prefix omitted on purpose: the local `inference` package's
    # parser (`inference.core.utils.roboflow.get_model_id_chunks`) only
    # accepts `<dataset>/<version>` and rejects the workspace-prefixed
    # `roboflow-jvuqo/basketball-jersey-numbers-ocr/7` form, even though
    # the hosted HTTP API accepts both. Short form works on both backends.
    jersey_model_id: str = "basketball-jersey-numbers-ocr/7"
    jersey_confidence: float = 0.4
    jersey_sample_every: int = 5      # call OCR every Nth frame per track
    jersey_confirm_at: int = 3        # votes needed to lock a number
    jersey_min_confidence: float = 0.5  # discard reads below this

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
    # Half-court minimap — square-ish since one half is 47x50 ft.
    half_minimap_width: int = 470
    half_minimap_height: int = 500

    # Composite layout — which minimap(s) to render alongside the broadcast.
    # "full"  = full-court only (legacy)
    # "half"  = active half only, 2x scale, auto-flipped from keypoints
    # "both"  = full (top) + active half (bottom), stacked
    view: str = "both"

    # Active-half hysteresis — number of recent frames to consider before
    # switching the rendered half. Higher = more stable but slower to react.
    half_hysteresis_frames: int = 15

    # Ball possession (which tracked player has the ball this frame)
    # Pixel distance from ball center to a player's bbox — beyond this we
    # consider the ball "loose" (mid-air, mid-pass, on the floor). 120px is
    # roughly arm's reach in a 1080p broadcast frame; broadcasts that zoom
    # tighter may need a larger value.
    possession_max_distance_px: float = 120.0
    # Frames a candidate must be nearest before we commit possession.
    # Higher = more stable but slower to react to steals/rebounds.
    possession_confirm_at: int = 3
    # How many consecutive frames the ball can be missing / loose before
    # we drop the current possessor. Smooths over single-frame detection
    # gaps without holding stale possession across plays.
    possession_release_after_missing: int = 8

    # Shot/event detection (uses action classes the model already returns).
    # shot_window: rolling per-track buffer of action labels (frames).
    # shot_confirm_at: how many of the last shot_window frames must show a
    #   shot action for the shot to be confirmed. Drops single-frame noise.
    # made_window_frames: after a shot's last frame, look this many frames
    #   ahead for a ball-in-basket observation to call it made. 18 ≈ 0.6s
    #   at 30 fps — about how long the ball takes to drop through.
    shot_window: int = 6
    shot_confirm_at: int = 3
    made_window_frames: int = 18

    # Camera cut detection: histogram Bhattacharyya distance threshold
    # above which a hard cut is declared (then tracker, possession, and
    # in-progress shots all reset). Empirically 0.45 sits cleanly between
    # "lots of camera motion" (~0.25) and "different scene" (~0.55+).
    cut_threshold: float = 0.45

    # Team classification
    n_teams: int = 2  # 2 = just teams, 3 = teams + referees
    # When set, every chest crop the classifier samples is saved as a PNG
    # with V/S/accent values in the filename. Lets you visually verify the
    # ROI lands on the jersey body — debug tool, not for production runs.
    team_classifier_debug_crops: Optional[str] = None
    # When both team names are set, the classifier uses canonical NBA color
    # signatures (TEAM_PROFILES in team_classifier.py) and skips KMeans.
    # Recommended for any clip where you know the matchup — KMeans is brittle
    # on warm-on-warm matchups (Knicks orange vs Sixers red are 10° apart in
    # hue and trip up an unsupervised cluster). Names are case-insensitive.
    # Example: --team-a knicks --team-b sixers
    team_a: Optional[str] = None
    team_b: Optional[str] = None

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
        parser.add_argument("--n-teams", type=int, default=2, help="Number of teams (2 or 3 for refs)")
        parser.add_argument("--debug", action="store_true", help="Draw debug overlays")
        parser.add_argument(
            "--backend", choices=["hosted", "local"], default="hosted",
            help="Inference backend: 'hosted' (HTTP, default) or 'local' "
                 "(in-process via the inference package).",
        )
        parser.add_argument(
            "--inference-host", default="https://detect.roboflow.com",
            help="Base URL for the 'hosted' backend. Override to point at a "
                 "self-hosted Roboflow inference-server (e.g. a friend's "
                 "DGX Spark on Tailscale): http://spark.tail-xxx.ts.net:9001",
        )
        parser.add_argument(
            "--view", choices=["full", "half", "both"], default="both",
            help="Minimap layout: 'full' (full court only), 'half' (active "
                 "half only, auto-flipped), or 'both' (stacked). Default: both.",
        )
        parser.add_argument(
            "--jersey-ocr", action="store_true",
            help="Enable jersey number OCR (adds API calls; persistent IDs "
                 "across cuts; opt-in because it slows things down).",
        )
        parser.add_argument(
            "--jersey-sample-every", type=int, default=5,
            help="Run jersey OCR every Nth frame per unlocked track.",
        )
        parser.add_argument(
            "--team-a", default=None,
            help="Name of team 0 for supervised color classification "
                 "(e.g. 'knicks'). When both --team-a and --team-b are set, "
                 "the classifier skips KMeans and matches each track to the "
                 "team with the closest canonical color signature. See "
                 "TEAM_PROFILES in team_classifier.py for the full list.",
        )
        parser.add_argument(
            "--team-b", default=None,
            help="Name of team 1 for supervised color classification "
                 "(e.g. 'sixers'). See --team-a.",
        )
        parser.add_argument(
            "--team-debug-crops", default=None,
            help="Directory to dump every chest-crop the team classifier "
                 "sampled, with V/S/accent values in each filename. Use to "
                 "verify the ROI is on the jersey body.",
        )

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
            n_teams=args.n_teams,
            debug=args.debug,
            inference_backend=args.backend,
            inference_host=args.inference_host,
            view=args.view,
            enable_jersey_ocr=args.jersey_ocr,
            jersey_sample_every=args.jersey_sample_every,
            team_a=args.team_a,
            team_b=args.team_b,
            team_classifier_debug_crops=args.team_debug_crops,
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
