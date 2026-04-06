"""NBA court geometry constants and keypoint mapping.

All coordinates in feet. Origin (0, 0) at top-left corner of the court.
X-axis: length (0-94, baseline to baseline).
Y-axis: width (0-50, sideline to sideline).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# ── Court dimensions (feet) ──────────────────────────────────────────────────

COURT_LENGTH = 94.0
COURT_WIDTH = 50.0

# Baskets are 4ft from baseline (center of rim), centered at 25ft width
BASKET_OFFSET = 4.0  # from baseline to center of hoop
BASKET_LEFT = (BASKET_OFFSET, COURT_WIDTH / 2)       # (4, 25)
BASKET_RIGHT = (COURT_LENGTH - BASKET_OFFSET, COURT_WIDTH / 2)  # (90, 25)

# Free throw line: 15ft from backboard, backboard is ~4ft from baseline
# So free throw line is at 4 + 15 = 19ft from baseline
FT_LINE_DIST = 19.0  # from nearest baseline

# Key (paint): 16ft wide, centered
KEY_WIDTH = 16.0
KEY_TOP = (COURT_WIDTH - KEY_WIDTH) / 2   # 17ft from top sideline
KEY_BOTTOM = KEY_TOP + KEY_WIDTH           # 33ft from top sideline

# Three-point arc: 23ft 9in from basket center = 23.75ft
# Corner three: 22ft, extends 14ft from baseline
THREE_PT_RADIUS = 23.75
THREE_PT_CORNER_DIST = 22.0  # corner 3 distance from basket (sideline)
THREE_PT_BREAK_X = 14.0  # where arc meets corner line (from baseline)

# Half court
HALF_COURT_X = COURT_LENGTH / 2  # 47ft
CENTER_CIRCLE_RADIUS = 6.0  # 6ft radius

# Free throw circle radius
FT_CIRCLE_RADIUS = 6.0

# Restricted area arc: 4ft radius from basket center
RESTRICTED_ARC_RADIUS = 4.0

# Backboard: 4ft from baseline
BACKBOARD_OFFSET = 4.0


# ── Named court landmarks ────────────────────────────────────────────────────
# These are the reference points we use for homography.
# Each is a (x, y) coordinate in feet on the court.

COURT_LANDMARKS: Dict[str, Tuple[float, float]] = {
    # Four corners
    "court_top_left": (0.0, 0.0),
    "court_top_right": (COURT_LENGTH, 0.0),
    "court_bottom_left": (0.0, COURT_WIDTH),
    "court_bottom_right": (COURT_LENGTH, COURT_WIDTH),

    # Half court line intersections with sidelines
    "half_court_top": (HALF_COURT_X, 0.0),
    "half_court_bottom": (HALF_COURT_X, COURT_WIDTH),
    "half_court_center": (HALF_COURT_X, COURT_WIDTH / 2),

    # Left side key (paint) corners
    "left_key_top_left": (0.0, KEY_TOP),
    "left_key_top_right": (FT_LINE_DIST, KEY_TOP),
    "left_key_bottom_left": (0.0, KEY_BOTTOM),
    "left_key_bottom_right": (FT_LINE_DIST, KEY_BOTTOM),
    "left_ft_center": (FT_LINE_DIST, COURT_WIDTH / 2),

    # Right side key (paint) corners
    "right_key_top_left": (COURT_LENGTH - FT_LINE_DIST, KEY_TOP),
    "right_key_top_right": (COURT_LENGTH, KEY_TOP),
    "right_key_bottom_left": (COURT_LENGTH - FT_LINE_DIST, KEY_BOTTOM),
    "right_key_bottom_right": (COURT_LENGTH, KEY_BOTTOM),
    "right_ft_center": (COURT_LENGTH - FT_LINE_DIST, COURT_WIDTH / 2),

    # Three-point line meets sideline (corner 3 spots)
    "left_three_top": (THREE_PT_BREAK_X, 0.0 + 3.0),  # ~3ft from sideline
    "left_three_bottom": (THREE_PT_BREAK_X, COURT_WIDTH - 3.0),
    "right_three_top": (COURT_LENGTH - THREE_PT_BREAK_X, 0.0 + 3.0),
    "right_three_bottom": (COURT_LENGTH - THREE_PT_BREAK_X, COURT_WIDTH - 3.0),

    # Baskets
    "left_basket": BASKET_LEFT,
    "right_basket": BASKET_RIGHT,
}


# ── Roboflow keypoint model class name → court landmark mapping ──────────────
# This maps the class labels from the Roboflow basketball-court-detection-2
# model to our COURT_LANDMARKS keys. Update these after inspecting actual
# model output on test frames.
#
# The Roboflow model detects court keypoints as object detection classes.
# Each detected "object" is a court landmark at a specific pixel location.

ROBOFLOW_KEYPOINT_MAP: Dict[str, str] = {
    # Format: "roboflow_class_name": "court_landmark_key"
    # These will be calibrated by running the model on test frames.
    # Below are best-guess mappings based on common Roboflow court models.
    "top-left": "court_top_left",
    "top-right": "court_top_right",
    "bottom-left": "court_bottom_left",
    "bottom-right": "court_bottom_right",
    "half-top": "half_court_top",
    "half-bottom": "half_court_bottom",
    "half-center": "half_court_center",
    "left-ft-top": "left_key_top_right",
    "left-ft-bottom": "left_key_bottom_right",
    "left-ft-center": "left_ft_center",
    "right-ft-top": "right_key_top_left",
    "right-ft-bottom": "right_key_bottom_left",
    "right-ft-center": "right_ft_center",
    "left-key-top": "left_key_top_left",
    "left-key-bottom": "left_key_bottom_left",
    "right-key-top": "right_key_top_right",
    "right-key-bottom": "right_key_bottom_right",
    "left-three-top": "left_three_top",
    "left-three-bottom": "left_three_bottom",
    "right-three-top": "right_three_top",
    "right-three-bottom": "right_three_bottom",
}


def get_court_point(roboflow_class: str) -> Tuple[float, float] | None:
    """Map a Roboflow keypoint class name to real-world court coordinates.

    Returns (x_ft, y_ft) or None if the class name is not recognized.
    """
    landmark_key = ROBOFLOW_KEYPOINT_MAP.get(roboflow_class)
    if landmark_key is None:
        return None
    return COURT_LANDMARKS.get(landmark_key)


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
