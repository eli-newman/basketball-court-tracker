# -*- coding: utf-8 -*-
"""basketball_tracker_colab.ipynb

# Basketball Court Tracker - Colab Edition

Track basketball players and map their positions to a 2D court minimap
using per-frame homography from court keypoint detection.

## Features
- **Player detection** via Roboflow basketball model
- **Court keypoint detection** via Roboflow pose model (33 landmarks)
- **Per-frame homography** — handles camera pans, zooms, and cuts
- **Team classification** by jersey color (KMeans clustering)
- **2D minimap** with player dots, trails, and team colors
- **JSON/CSV export** of court coordinates per frame

## Setup
1. **Enable GPU**: Runtime > Change runtime type > T4 GPU
2. **Run all cells** in order
3. **Upload your video** or mount Google Drive
4. **Set your Roboflow API key** in the config cell
"""

#@title 1. Install Dependencies
!pip install -q supervision opencv-python-headless requests scikit-learn numpy

import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print(f"GPU: {result.stdout.strip()}" if result.returncode == 0 else "No GPU detected (will be slower)")
print("Dependencies installed!")

#@title 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#@title 3. Configuration
#@markdown ### Required Settings
ROBOFLOW_API_KEY = ""  #@param {type:"string"}
VIDEO_PATH = "/content/drive/MyDrive/basketball_clip.mp4"  #@param {type:"string"}
OUTPUT_DIR = "/content/drive/MyDrive/basketball_output"  #@param {type:"string"}

#@markdown ### Detection Settings
PLAYER_CONFIDENCE = 0.4  #@param {type:"slider", min:0.1, max:0.9, step:0.05}
COURT_CONFIDENCE = 0.3  #@param {type:"slider", min:0.1, max:0.9, step:0.05}
KEYPOINT_CONFIDENCE = 0.5  #@param {type:"slider", min:0.1, max:0.9, step:0.05}

#@markdown ### Processing Settings
FRAME_SKIP = 1  #@param {type:"integer"}
MAX_FRAMES = 0  #@param {type:"integer"}
N_TEAMS = 2  #@param {type:"slider", min:2, max:3, step:1}
DEBUG_MODE = False  #@param {type:"boolean"}

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not ROBOFLOW_API_KEY:
    print("WARNING: Set your Roboflow API key above!")
else:
    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Teams: {N_TEAMS}")

#@title 4. Court Geometry & Keypoint Mapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from collections import deque

# ── Court dimensions ─────────────────────────────────────────────────────
_COURT_WIDTH_CM = 1524
_COURT_LENGTH_CM = 2865
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
CM_TO_FT = 1.0 / 30.48

BASKET_OFFSET = 5.25
FT_LINE_DIST = 19.0
KEY_WIDTH = 16.0
KEY_TOP = (COURT_WIDTH - KEY_WIDTH) / 2
KEY_BOTTOM = KEY_TOP + KEY_WIDTH
HALF_COURT_X = COURT_LENGTH / 2
THREE_PT_RADIUS = 23.75
THREE_PT_BREAK_X = 14.0
CENTER_CIRCLE_RADIUS = 6.0
FT_CIRCLE_RADIUS = 6.0
RESTRICTED_ARC_RADIUS = 4.0

# ── Roboflow keypoint mapping (from roboflow/sports BasketballCourtConfiguration) ──
_PAINT_START_CM = (_COURT_WIDTH_CM - 488) // 2
_SIDELINE_TO_3PT_CM = 91
_BASELINE_TO_RIM_CM = 160
_STRAIGHT_3PT_CM = 424
_PAINT_LENGTH_CM = 579
_BASELINE_TO_THROW_CM = 835
_3PT_ARC_RADIUS_CM = 724

_KEYPOINTS_CM = [
    ("01", 0, 0), ("02", 0, _SIDELINE_TO_3PT_CM),
    ("04", 0, _PAINT_START_CM), ("05", 0, _PAINT_START_CM + 488),
    ("07", 0, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM), ("08", 0, _COURT_WIDTH_CM),
    ("09", _BASELINE_TO_RIM_CM, _COURT_WIDTH_CM // 2),
    ("10", _STRAIGHT_3PT_CM, _SIDELINE_TO_3PT_CM),
    ("11", _STRAIGHT_3PT_CM, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM),
    ("12", _PAINT_LENGTH_CM, _PAINT_START_CM),
    ("13", _PAINT_LENGTH_CM, _PAINT_START_CM + 244),
    ("14", _PAINT_LENGTH_CM, _PAINT_START_CM + 488),
    ("15", _BASELINE_TO_THROW_CM, 0),
    ("16", _BASELINE_TO_RIM_CM + _3PT_ARC_RADIUS_CM, _COURT_WIDTH_CM // 2),
    ("17", _BASELINE_TO_THROW_CM, _COURT_WIDTH_CM),
    ("19", _COURT_LENGTH_CM // 2, 0),
    ("21", _COURT_LENGTH_CM // 2, _COURT_WIDTH_CM // 2),
    ("23", _COURT_LENGTH_CM // 2, _COURT_WIDTH_CM),
    ("25", _COURT_LENGTH_CM - _BASELINE_TO_THROW_CM, 0),
    ("26", _COURT_LENGTH_CM - _BASELINE_TO_RIM_CM - _3PT_ARC_RADIUS_CM, _COURT_WIDTH_CM // 2),
    ("27", _COURT_LENGTH_CM - _BASELINE_TO_THROW_CM, _COURT_WIDTH_CM),
    ("28", _COURT_LENGTH_CM - _PAINT_LENGTH_CM, _PAINT_START_CM),
    ("29", _COURT_LENGTH_CM - _PAINT_LENGTH_CM, _PAINT_START_CM + 244),
    ("30", _COURT_LENGTH_CM - _PAINT_LENGTH_CM, _PAINT_START_CM + 488),
    ("31", _COURT_LENGTH_CM - _STRAIGHT_3PT_CM, _SIDELINE_TO_3PT_CM),
    ("32", _COURT_LENGTH_CM - _STRAIGHT_3PT_CM, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM),
    ("33", _COURT_LENGTH_CM - _BASELINE_TO_RIM_CM, _COURT_WIDTH_CM // 2),
    ("34", _COURT_LENGTH_CM, 0), ("35", _COURT_LENGTH_CM, _SIDELINE_TO_3PT_CM),
    ("37", _COURT_LENGTH_CM, _PAINT_START_CM),
    ("38", _COURT_LENGTH_CM, _PAINT_START_CM + 488),
    ("40", _COURT_LENGTH_CM, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM),
    ("41", _COURT_LENGTH_CM, _COURT_WIDTH_CM),
]

KEYPOINT_LABEL_MAP = {label: (x * CM_TO_FT, y * CM_TO_FT) for label, x, y in _KEYPOINTS_CM}
KEYPOINT_INDEX_MAP = {i: (x * CM_TO_FT, y * CM_TO_FT) for i, (_, x, y) in enumerate(_KEYPOINTS_CM)}


def get_court_point(name_or_index):
    if isinstance(name_or_index, int):
        return KEYPOINT_INDEX_MAP.get(name_or_index)
    if isinstance(name_or_index, str):
        return KEYPOINT_LABEL_MAP.get(name_or_index)
    return None


def court_to_minimap(x_ft, y_ft, w, h, pad=10):
    draw_w, draw_h = w - 2 * pad, h - 2 * pad
    return (int(pad + (x_ft / COURT_LENGTH) * draw_w),
            int(pad + (y_ft / COURT_WIDTH) * draw_h))


print(f"Court: {COURT_LENGTH}ft x {COURT_WIDTH}ft")
print(f"Keypoints mapped: {len(KEYPOINT_LABEL_MAP)}")

#@title 5. Court Template Generator
import cv2
import numpy as np

COURT_COLOR = (42, 100, 180)
LINE_COLOR = (255, 255, 255)
BACKBOARD_COLOR = (200, 200, 200)
RIM_COLOR = (0, 100, 255)


def generate_court_image(width=940, height=500, padding=10):
    img = np.full((height, width, 3), COURT_COLOR, dtype=np.uint8)

    def ft_to_px(x, y):
        return court_to_minimap(x, y, width, height, padding)

    def ft_to_r(r):
        return max(1, int(r * (width - 2 * padding) / COURT_LENGTH))

    t = max(1, width // 400)

    # Court outline + half court
    cv2.rectangle(img, ft_to_px(0, 0), ft_to_px(COURT_LENGTH, COURT_WIDTH), LINE_COLOR, t)
    cv2.line(img, ft_to_px(HALF_COURT_X, 0), ft_to_px(HALF_COURT_X, COURT_WIDTH), LINE_COLOR, t)
    cv2.circle(img, ft_to_px(HALF_COURT_X, COURT_WIDTH / 2), ft_to_r(CENTER_CIRCLE_RADIUS), LINE_COLOR, t)

    for side in ["left", "right"]:
        bx = BASKET_OFFSET if side == "left" else COURT_LENGTH - BASKET_OFFSET
        ft_x = FT_LINE_DIST if side == "left" else COURT_LENGTH - FT_LINE_DIST
        x_base = 0 if side == "left" else COURT_LENGTH

        # Key
        cv2.rectangle(img, ft_to_px(min(x_base, ft_x), KEY_TOP),
                       ft_to_px(max(x_base, ft_x), KEY_BOTTOM), LINE_COLOR, t)
        cv2.circle(img, ft_to_px(ft_x, COURT_WIDTH / 2), ft_to_r(FT_CIRCLE_RADIUS), LINE_COLOR, t)

        # 3pt lines
        bc = ft_to_px(bx, COURT_WIDTH / 2)
        r3 = ft_to_r(THREE_PT_RADIUS)
        cv2.line(img, ft_to_px(x_base, 3.0), ft_to_px(THREE_PT_BREAK_X if side == "left" else COURT_LENGTH - THREE_PT_BREAK_X, 3.0), LINE_COLOR, t)
        cv2.line(img, ft_to_px(x_base, COURT_WIDTH - 3.0), ft_to_px(THREE_PT_BREAK_X if side == "left" else COURT_LENGTH - THREE_PT_BREAK_X, COURT_WIDTH - 3.0), LINE_COLOR, t)
        if side == "left":
            cv2.ellipse(img, bc, (r3, r3), 0, -158, 158, LINE_COLOR, t)
        else:
            cv2.ellipse(img, bc, (r3, r3), 0, 112, 248, LINE_COLOR, t)

        # Restricted area
        cv2.ellipse(img, ft_to_px(bx, COURT_WIDTH / 2), (ft_to_r(RESTRICTED_ARC_RADIUS),) * 2,
                     0, -90 if side == "left" else 90, 90 if side == "left" else 270, LINE_COLOR, t)

        # Basket
        bb_x = bx - 0.5 if side == "left" else bx + 0.5
        cv2.line(img, ft_to_px(bb_x, COURT_WIDTH / 2 - 3), ft_to_px(bb_x, COURT_WIDTH / 2 + 3), BACKBOARD_COLOR, t + 1)
        cv2.circle(img, ft_to_px(bx, COURT_WIDTH / 2), max(ft_to_r(0.75), 2), RIM_COLOR, t)

    return img


# Preview the court
from IPython.display import display
from PIL import Image

court_img = generate_court_image()
display(Image.fromarray(cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)))
print("Court template generated")

#@title 6. Homography Engine
@dataclass
class CourtKeypoint:
    name: Union[int, str]
    pixel_x: float
    pixel_y: float
    confidence: float


class HomographyEngine:
    def __init__(self, min_kp=4, fallback_frames=30, ransac_thresh=5.0, max_reproj=10.0, margin=5.0):
        self.min_kp = min_kp
        self.fallback_frames = fallback_frames
        self.ransac_thresh = ransac_thresh
        self.max_reproj = max_reproj
        self.margin = margin
        self._last_H = None
        self._frames_since = 0

    def compute(self, keypoints):
        src, dst = [], []
        for kp in keypoints:
            pt = get_court_point(kp.name)
            if pt:
                src.append([kp.pixel_x, kp.pixel_y])
                dst.append([pt[0], pt[1]])

        if len(src) < self.min_kp:
            return self._fallback()

        H, mask = cv2.findHomography(np.array(src, np.float64), np.array(dst, np.float64), cv2.RANSAC, self.ransac_thresh)
        if H is None or not self._validate(H, np.array(src, np.float64), np.array(dst, np.float64), mask):
            return self._fallback()

        self._last_H = H.copy()
        self._frames_since = 0
        return H

    def transform_point(self, H, px, py):
        pt = cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), H)
        x, y = float(pt[0, 0, 0]), float(pt[0, 0, 1])
        if x < -self.margin or x > COURT_LENGTH + self.margin or y < -self.margin or y > COURT_WIDTH + self.margin:
            return None
        return (max(0.0, min(COURT_LENGTH, x)), max(0.0, min(COURT_WIDTH, y)))

    def reset(self):
        self._last_H = None
        self._frames_since = 0

    @property
    def has_valid(self):
        return self._last_H is not None

    def _fallback(self):
        self._frames_since += 1
        if self._last_H is not None and self._frames_since <= self.fallback_frames:
            return self._last_H.copy()
        return None

    def _validate(self, H, src, dst, mask):
        if abs(np.linalg.det(H)) < 1e-6:
            return False
        if mask is not None:
            inliers_src = src[mask.ravel() == 1]
            inliers_dst = dst[mask.ravel() == 1]
            if len(inliers_src) < self.min_kp:
                return False
            src_h = np.hstack([inliers_src, np.ones((len(inliers_src), 1))])
            proj = (H @ src_h.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            if np.mean(np.sqrt(np.sum((proj - inliers_dst) ** 2, axis=1))) > self.max_reproj:
                return False
        try:
            H_inv = np.linalg.inv(H)
            corners = np.array([[[0, 0]], [[COURT_LENGTH, 0]], [[COURT_LENGTH, COURT_WIDTH]], [[0, COURT_WIDTH]]], np.float64)
            if np.any(np.abs(cv2.perspectiveTransform(corners, H_inv)) > 1e5):
                return False
        except np.linalg.LinAlgError:
            return False
        return True


print("Homography engine ready")

#@title 7. Detection & Tracking
import base64
import requests
import supervision as sv


@dataclass
class PlayerDetection:
    bbox: tuple
    bottom_center: tuple
    center: tuple
    confidence: float
    class_name: str


def call_roboflow_api(frame, model_id, api_key, confidence):
    _, buf = cv2.imencode(".jpg", frame)
    resp = requests.post(
        f"https://detect.roboflow.com/{model_id}",
        params={"api_key": api_key, "confidence": confidence},
        data=base64.b64encode(buf).decode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    return resp.json() if resp.status_code == 200 else None


def detect_players(frame, api_key, confidence=0.4):
    result = call_roboflow_api(frame, "basketball-player-detection-3-ycjdo/6", api_key, confidence)
    if not result:
        return []
    players = []
    for pred in result.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        players.append(PlayerDetection(
            bbox=(x - w/2, y - h/2, x + w/2, y + h/2),
            bottom_center=(x, y + h/2),
            center=(x, y),
            confidence=pred.get("confidence", 0),
            class_name=pred.get("class", "player"),
        ))
    return players


def detect_court_keypoints(frame, api_key, confidence=0.3, kp_confidence=0.5):
    result = call_roboflow_api(frame, "basketball-court-detection-2/13", api_key, confidence)
    if not result:
        return []
    keypoints = []
    for pred in result.get("predictions", []):
        if "keypoints" in pred:
            for kp in pred["keypoints"]:
                conf = kp.get("confidence", 0)
                if conf < kp_confidence:
                    continue
                name = kp.get("class_name", kp.get("class", ""))
                try:
                    name = int(name)
                except (ValueError, TypeError):
                    pass
                keypoints.append(CourtKeypoint(name=name, pixel_x=kp["x"], pixel_y=kp["y"], confidence=conf))
        else:
            keypoints.append(CourtKeypoint(
                name=pred.get("class", ""),
                pixel_x=pred["x"], pixel_y=pred["y"],
                confidence=pred.get("confidence", 0),
            ))
    return keypoints


print("Detection functions ready")

#@title 8. Team Classifier
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self, n_teams=2, warmup=5):
        self.n_teams = n_teams
        self._kmeans = None
        self._samples = []
        self._warmup = warmup
        self._frame_count = 0

    def classify(self, frame, players):
        if not players:
            return []
        colors = [self._extract_color(frame, p) for p in players]
        arr = np.array(colors)
        self._frame_count += 1

        if self._kmeans is None:
            self._samples.append(arr)
            if self._frame_count >= self._warmup:
                all_colors = np.vstack(self._samples)
                if len(all_colors) >= self.n_teams:
                    self._kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
                    self._kmeans.fit(all_colors)
                    for i, c in enumerate(self._kmeans.cluster_centers_):
                        print(f"  Team {i}: HSV=({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})")
            return [-1] * len(players)

        return self._kmeans.predict(arr).tolist()

    def _extract_color(self, frame, player):
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.array([128, 128, 128], np.float32)
        bh, bw = y2 - y1, x2 - x1
        crop = frame[y1 + int(bh * 0.2):y1 + int(bh * 0.6),
                      x1 + int(bw * 0.2):x2 - int(bw * 0.2)]
        if crop.size == 0:
            crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([128, 128, 128], np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 2] > 40) & (hsv[:, :, 2] < 240)
        pixels = hsv[mask] if mask.sum() >= 10 else hsv.reshape(-1, 3)
        return pixels.mean(axis=0).astype(np.float32)


print("Team classifier ready")

#@title 9. Minimap Renderer
@dataclass
class MappedPlayer:
    track_id: int
    court_x: float
    court_y: float
    pixel_x: float
    pixel_y: float
    bbox: tuple
    confidence: float
    class_name: str
    team_id: int = -1

COLOR_TEAM_A = (255, 100, 50)
COLOR_TEAM_B = (50, 50, 255)
COLOR_UNKNOWN = (200, 200, 200)
COLOR_REFEREE = (0, 200, 200)
MINIMAP_W, MINIMAP_H, MINIMAP_PAD = 470, 250, 10
TRAIL_LENGTH = 15

court_base = generate_court_image(MINIMAP_W, MINIMAP_H, MINIMAP_PAD)
trails = {}


def render_minimap(mapped_players, h_valid=True):
    global trails
    img = court_base.copy()
    if not h_valid:
        overlay = np.full_like(img, (80, 80, 80))
        img = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
        cv2.putText(img, "NO TRACKING", (img.shape[1]//2 - 80, img.shape[0]//2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    for p in mapped_players:
        px, py = court_to_minimap(p.court_x, p.court_y, MINIMAP_W, MINIMAP_H, MINIMAP_PAD)
        color = COLOR_TEAM_A if p.team_id == 0 else COLOR_TEAM_B if p.team_id == 1 else COLOR_REFEREE if p.team_id == 2 else COLOR_UNKNOWN

        if p.track_id not in trails:
            trails[p.track_id] = deque(maxlen=TRAIL_LENGTH)
        trails[p.track_id].append((px, py))

        trail = list(trails[p.track_id])
        for i in range(1, len(trail)):
            a = i / len(trail)
            cv2.line(img, trail[i-1], trail[i], tuple(int(c * a) for c in color), 1)

        cv2.circle(img, (px, py), 5, color, -1)
        cv2.circle(img, (px, py), 5, (255, 255, 255), 1)
        cv2.putText(img, str(p.track_id % 100), (px + 7, py + 3),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return img


print("Minimap renderer ready")

#@title 10. Run Pipeline
import json
import csv
import time

assert ROBOFLOW_API_KEY, "Set your Roboflow API key in cell 3!"
assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"

# Initialize
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(f"Input: {VIDEO_PATH}")
print(f"Resolution: {video_info.width}x{video_info.height} @ {video_info.fps}fps")
print(f"Total frames: {video_info.total_frames}")

tracker = sv.ByteTrack(track_activation_threshold=PLAYER_CONFIDENCE, minimum_matching_threshold=0.8, frame_rate=int(video_info.fps))
homography = HomographyEngine(min_kp=4, fallback_frames=30)
team_clf = TeamClassifier(n_teams=N_TEAMS)
position_history = {}

# Output paths
video_out = os.path.join(OUTPUT_DIR, "output_composite.mp4")
json_out = os.path.join(OUTPUT_DIR, "output_coordinates.json")
csv_out = os.path.join(OUTPUT_DIR, "output_coordinates.csv")

# Compute output dimensions
minimap_scale = video_info.height / MINIMAP_H
output_w = video_info.width + int(MINIMAP_W * minimap_scale)

output_info = sv.VideoInfo(width=output_w, height=video_info.height, fps=video_info.fps)
box_ann = sv.BoxAnnotator(thickness=2)
label_ann = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_thickness=1, text_scale=0.5)

all_frame_data = []
frame_count = 0
valid_h_count = 0
start_time = time.time()

frame_gen = sv.get_video_frames_generator(VIDEO_PATH, stride=FRAME_SKIP)

print(f"\nProcessing{'...' if MAX_FRAMES == 0 else f' (max {MAX_FRAMES} frames)...'}")
print()

with sv.VideoSink(video_out, output_info) as sink:
    for frame in frame_gen:
        frame_count += 1
        if MAX_FRAMES > 0 and frame_count > MAX_FRAMES:
            break

        # 1. Detect court keypoints
        keypoints = detect_court_keypoints(frame, ROBOFLOW_API_KEY, COURT_CONFIDENCE, KEYPOINT_CONFIDENCE)

        # 2. Detect players
        raw_players = detect_players(frame, ROBOFLOW_API_KEY, PLAYER_CONFIDENCE)

        # 3. Classify teams
        team_ids = team_clf.classify(frame, raw_players)

        # 4. Track with ByteTrack
        if raw_players:
            xyxy = np.array([p.bbox for p in raw_players], np.float32)
            conf = np.array([p.confidence for p in raw_players], np.float32)
            sv_dets = tracker.update_with_detections(sv.Detections(xyxy=xyxy, confidence=conf))
        else:
            sv_dets = sv.Detections.empty()

        # 5. Compute homography
        H = homography.compute(keypoints)
        h_valid = H is not None
        if h_valid:
            valid_h_count += 1

        # 6. Map players to court coordinates
        mapped = []
        if H is not None:
            for i in range(len(sv_dets)):
                bbox = sv_dets.xyxy[i]
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                feet_x, feet_y = cx, y2  # bottom center = feet

                court_pos = homography.transform_point(H, feet_x, feet_y)
                if court_pos is None:
                    continue

                tid = int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else -1

                # Temporal smoothing
                if tid not in position_history:
                    position_history[tid] = deque(maxlen=5)
                position_history[tid].append(court_pos)
                xs = [p[0] for p in position_history[tid]]
                ys = [p[1] for p in position_history[tid]]
                smooth_x, smooth_y = sum(xs) / len(xs), sum(ys) / len(ys)

                # Match team_id
                best_team = -1
                if team_ids and team_ids[0] != -1:
                    best_dist = float("inf")
                    for raw, t in zip(raw_players, team_ids):
                        dx, dy = cx - raw.center[0], cy - raw.center[1]
                        d = dx*dx + dy*dy
                        if d < best_dist:
                            best_dist, best_team = d, t

                mapped.append(MappedPlayer(
                    track_id=tid, court_x=smooth_x, court_y=smooth_y,
                    pixel_x=cx, pixel_y=cy, bbox=(x1, y1, x2, y2),
                    confidence=float(sv_dets.confidence[i]) if sv_dets.confidence is not None else 0,
                    class_name="player", team_id=best_team,
                ))

        # 7. Render annotated frame
        annotated = frame.copy()
        if len(sv_dets) > 0:
            labels = [f"#{int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else -1}"
                      for i in range(len(sv_dets))]
            annotated = box_ann.annotate(annotated, sv_dets)
            annotated = label_ann.annotate(annotated, sv_dets, labels)

        if DEBUG_MODE:
            for kp in keypoints:
                cv2.drawMarker(annotated, (int(kp.pixel_x), int(kp.pixel_y)), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(annotated, str(kp.name), (int(kp.pixel_x) + 10, int(kp.pixel_y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 8. Render minimap + composite
        minimap = render_minimap(mapped, h_valid)
        minimap_scaled = cv2.resize(minimap, (int(MINIMAP_W * minimap_scale), video_info.height))
        composite = np.hstack([annotated, minimap_scaled])

        status = "TRACKING" if h_valid else "NO HOMOGRAPHY"
        cv2.putText(composite, f"{status} | {len(mapped)} players",
                     (video_info.width + 10, video_info.height - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        sink.write_frame(composite)

        # 9. Collect data
        all_frame_data.append({
            "frame": frame_count,
            "timestamp_ms": round(frame_count * 1000.0 / video_info.fps, 1),
            "homography_valid": h_valid,
            "keypoints_detected": len(keypoints),
            "players": [{"track_id": p.track_id, "court_x": round(p.court_x, 2),
                          "court_y": round(p.court_y, 2), "team_id": p.team_id}
                         for p in mapped],
        })

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  Frame {frame_count} | {frame_count/elapsed:.1f} fps | "
                  f"keypoints: {len(keypoints)} | players: {len(mapped)} | H: {h_valid}")

# Save JSON
with open(json_out, "w") as f:
    json.dump(all_frame_data, f, indent=2)

# Save CSV
with open(csv_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "timestamp_ms", "homography_valid", "track_id", "court_x", "court_y", "team_id"])
    for fd in all_frame_data:
        for p in fd["players"]:
            w.writerow([fd["frame"], fd["timestamp_ms"], fd["homography_valid"],
                        p["track_id"], p["court_x"], p["court_y"], p["team_id"]])

elapsed = time.time() - start_time
print()
print("=" * 60)
print(f"Done! {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
print(f"Homography valid: {valid_h_count}/{frame_count} ({valid_h_count/max(frame_count,1)*100:.0f}%)")
print(f"Video: {video_out}")
print(f"JSON:  {json_out}")
print(f"CSV:   {csv_out}")
print("=" * 60)

#@title 11. Preview Result
from IPython.display import HTML
from base64 import b64encode

# Show first frame of output
cap = cv2.VideoCapture(video_out)
ret, preview_frame = cap.read()
cap.release()

if ret:
    preview_frame = cv2.resize(preview_frame, (preview_frame.shape[1] // 2, preview_frame.shape[0] // 2))
    display(Image.fromarray(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)))

# Show coordinate stats
total_players = sum(len(fd["players"]) for fd in all_frame_data)
valid_frames = sum(1 for fd in all_frame_data if fd["homography_valid"])
print(f"\nStats:")
print(f"  Total player-frame detections: {total_players}")
print(f"  Frames with valid homography: {valid_frames}/{len(all_frame_data)}")
print(f"  Output files saved to: {OUTPUT_DIR}")
