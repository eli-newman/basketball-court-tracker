"""NBA court geometry constants and keypoint mapping.

All coordinates in feet. Origin (0, 0) at top-left corner of the court.
X-axis: length (0-94, baseline to baseline).
Y-axis: width (0-50, sideline to sideline).

Keypoint mapping sourced from roboflow/sports BasketballCourtConfiguration.
The Roboflow basketball-court-detection-2 model returns 33 keypoints labeled
"01" through "41" (with gaps). Each maps to a specific court landmark.
"""

from typing import Dict, List, Tuple

# ── Court dimensions ─────────────────────────────────────────────────────────
# Official NBA court in centimeters (used by Roboflow model internally)
_COURT_WIDTH_CM = 1524
_COURT_LENGTH_CM = 2865

# Convert to feet for our coordinate system
COURT_LENGTH = 94.0   # feet (≈ 2865cm / 30.48)
COURT_WIDTH = 50.0    # feet (≈ 1524cm / 30.48)
CM_TO_FT = 1.0 / 30.48

# Key dimensions
BASKET_OFFSET = 5.25     # from baseline to center of hoop (160cm ≈ 5.25ft)
FT_LINE_DIST = 19.0     # free throw line from baseline
KEY_WIDTH = 16.0         # paint width
KEY_TOP = (COURT_WIDTH - KEY_WIDTH) / 2   # 17ft
KEY_BOTTOM = KEY_TOP + KEY_WIDTH           # 33ft
HALF_COURT_X = COURT_LENGTH / 2           # 47ft
THREE_PT_RADIUS = 23.75
THREE_PT_BREAK_X = 14.0
CENTER_CIRCLE_RADIUS = 6.0
FT_CIRCLE_RADIUS = 6.0
RESTRICTED_ARC_RADIUS = 4.0
BACKBOARD_OFFSET = 4.0


# ── Roboflow keypoint model: exact mapping ───────────────────────────────────
# From roboflow/sports (feat/basketball_radar branch) BasketballCourtConfiguration.
# The model returns keypoints in this exact order with these labels.
# Coordinates below are in centimeters, converted to feet for our system.
#
# Each entry: (label_string, x_cm, y_cm, description)

_PAINT_START_CM = (_COURT_WIDTH_CM - 488) // 2  # 518cm
_SIDELINE_TO_3PT_CM = 91
_BASELINE_TO_RIM_CM = 160
_STRAIGHT_3PT_CM = 424
_PAINT_LENGTH_CM = 579
_BASELINE_TO_THROW_CM = 835
_3PT_ARC_RADIUS_CM = 724

_KEYPOINTS_CM: List[Tuple[str, float, float, str]] = [
    # Label, X_cm, Y_cm, Description
    # ── Left baseline ──
    ("01", 0, 0, "left baseline top corner"),
    ("02", 0, _SIDELINE_TO_3PT_CM, "left baseline 3pt top"),
    ("04", 0, _PAINT_START_CM, "left baseline paint top"),
    ("05", 0, _PAINT_START_CM + 488, "left baseline paint bottom"),
    ("07", 0, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM, "left baseline 3pt bottom"),
    ("08", 0, _COURT_WIDTH_CM, "left baseline bottom corner"),
    # ── Left side ──
    ("09", _BASELINE_TO_RIM_CM, _COURT_WIDTH_CM // 2, "left basket center"),
    ("10", _STRAIGHT_3PT_CM, _SIDELINE_TO_3PT_CM, "left 3pt straight top"),
    ("11", _STRAIGHT_3PT_CM, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM, "left 3pt straight bottom"),
    ("12", _PAINT_LENGTH_CM, _PAINT_START_CM, "left free throw top"),
    ("13", _PAINT_LENGTH_CM, _PAINT_START_CM + 244, "left free throw center"),
    ("14", _PAINT_LENGTH_CM, _PAINT_START_CM + 488, "left free throw bottom"),
    # ── Left side extended ──
    ("15", _BASELINE_TO_THROW_CM, 0, "left 3pt arc top sideline"),
    ("16", _BASELINE_TO_RIM_CM + _3PT_ARC_RADIUS_CM, _COURT_WIDTH_CM // 2, "left 3pt arc apex"),
    ("17", _BASELINE_TO_THROW_CM, _COURT_WIDTH_CM, "left 3pt arc bottom sideline"),
    # ── Half court ──
    ("19", _COURT_LENGTH_CM // 2, 0, "half court top"),
    ("21", _COURT_LENGTH_CM // 2, _COURT_WIDTH_CM // 2, "half court center"),
    ("23", _COURT_LENGTH_CM // 2, _COURT_WIDTH_CM, "half court bottom"),
    # ── Right side extended ──
    ("25", _COURT_LENGTH_CM - _BASELINE_TO_THROW_CM, 0, "right 3pt arc top sideline"),
    ("26", _COURT_LENGTH_CM - _BASELINE_TO_RIM_CM - _3PT_ARC_RADIUS_CM, _COURT_WIDTH_CM // 2, "right 3pt arc apex"),
    ("27", _COURT_LENGTH_CM - _BASELINE_TO_THROW_CM, _COURT_WIDTH_CM, "right 3pt arc bottom sideline"),
    # ── Right side ──
    ("28", _COURT_LENGTH_CM - _PAINT_LENGTH_CM, _PAINT_START_CM, "right free throw top"),
    ("29", _COURT_LENGTH_CM - _PAINT_LENGTH_CM, _PAINT_START_CM + 244, "right free throw center"),
    ("30", _COURT_LENGTH_CM - _PAINT_LENGTH_CM, _PAINT_START_CM + 488, "right free throw bottom"),
    ("31", _COURT_LENGTH_CM - _STRAIGHT_3PT_CM, _SIDELINE_TO_3PT_CM, "right 3pt straight top"),
    ("32", _COURT_LENGTH_CM - _STRAIGHT_3PT_CM, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM, "right 3pt straight bottom"),
    ("33", _COURT_LENGTH_CM - _BASELINE_TO_RIM_CM, _COURT_WIDTH_CM // 2, "right basket center"),
    # ── Right baseline ──
    ("34", _COURT_LENGTH_CM, 0, "right baseline top corner"),
    ("35", _COURT_LENGTH_CM, _SIDELINE_TO_3PT_CM, "right baseline 3pt top"),
    ("37", _COURT_LENGTH_CM, _PAINT_START_CM, "right baseline paint top"),
    ("38", _COURT_LENGTH_CM, _PAINT_START_CM + 488, "right baseline paint bottom"),
    ("40", _COURT_LENGTH_CM, _COURT_WIDTH_CM - _SIDELINE_TO_3PT_CM, "right baseline 3pt bottom"),
    ("41", _COURT_LENGTH_CM, _COURT_WIDTH_CM, "right baseline bottom corner"),
]

# Build the label → court coordinates (feet) mapping
# The Roboflow API returns keypoint "class" as these label strings ("01", "02", etc.)
KEYPOINT_LABEL_MAP: Dict[str, Tuple[float, float]] = {}
for _label, _x_cm, _y_cm, _desc in _KEYPOINTS_CM:
    KEYPOINT_LABEL_MAP[_label] = (_x_cm * CM_TO_FT, _y_cm * CM_TO_FT)

# Also build index-based map (keypoints come in array order 0-32)
KEYPOINT_INDEX_MAP: Dict[int, Tuple[float, float]] = {}
for _i, (_label, _x_cm, _y_cm, _desc) in enumerate(_KEYPOINTS_CM):
    KEYPOINT_INDEX_MAP[_i] = (_x_cm * CM_TO_FT, _y_cm * CM_TO_FT)

# Paint zone vertex indices (for zone detection)
LEFT_PAINT_INDICES = [2, 3, 11, 9]    # labels 04, 05, 14, 12
RIGHT_PAINT_INDICES = [21, 23, 30, 29]  # labels 28, 30, 30, 29


def get_court_point(name_or_index) -> Tuple[float, float] | None:
    """Map a Roboflow keypoint to court coordinates in feet.

    Accepts:
        - str label from API (e.g., "21" for half court center)
        - int index (array position 0-32)

    Returns (x_ft, y_ft) or None if not recognized.
    """
    if isinstance(name_or_index, int):
        return KEYPOINT_INDEX_MAP.get(name_or_index)
    if isinstance(name_or_index, str):
        return KEYPOINT_LABEL_MAP.get(name_or_index)
    return None


def court_to_minimap(
    x_ft: float,
    y_ft: float,
    minimap_w: int,
    minimap_h: int,
    padding: int = 10,
) -> Tuple[int, int]:
    """Convert court coordinates (feet) to minimap pixel coordinates."""
    draw_w = minimap_w - 2 * padding
    draw_h = minimap_h - 2 * padding
    px = int(padding + (x_ft / COURT_LENGTH) * draw_w)
    py = int(padding + (y_ft / COURT_WIDTH) * draw_h)
    return (px, py)


def minimap_scale(minimap_w: int, minimap_h: int, padding: int = 10) -> Tuple[float, float]:
    """Return (scale_x, scale_y) for feet → minimap pixels."""
    draw_w = minimap_w - 2 * padding
    draw_h = minimap_h - 2 * padding
    return (draw_w / COURT_LENGTH, draw_h / COURT_WIDTH)


def get_all_vertices_ft() -> List[Tuple[float, float]]:
    """Return all 33 court keypoint positions in feet, in model order."""
    return [(_x_cm * CM_TO_FT, _y_cm * CM_TO_FT) for _, _x_cm, _y_cm, _ in _KEYPOINTS_CM]
