"""Generate a top-down NBA court diagram image using OpenCV drawing primitives.

No external image files needed — everything is drawn programmatically.
"""

import math

import cv2
import numpy as np

from court import (
    COURT_LENGTH,
    COURT_WIDTH,
    FT_LINE_DIST,
    KEY_TOP,
    KEY_BOTTOM,
    KEY_WIDTH,
    HALF_COURT_X,
    CENTER_CIRCLE_RADIUS,
    FT_CIRCLE_RADIUS,
    THREE_PT_RADIUS,
    THREE_PT_BREAK_X,
    BASKET_OFFSET,
    RESTRICTED_ARC_RADIUS,
    court_to_minimap,
)

# Colors (BGR)
COURT_COLOR = (42, 100, 180)       # hardwood tan
LINE_COLOR = (255, 255, 255)       # white lines
KEY_COLOR = (35, 80, 150)          # slightly darker paint
THREE_PT_COLOR = (255, 255, 255)
BACKBOARD_COLOR = (200, 200, 200)
RIM_COLOR = (0, 100, 255)          # orange rim


def generate_court_image(width: int = 940, height: int = 500, padding: int = 10) -> np.ndarray:
    """Generate a top-down NBA court diagram.

    Args:
        width: Output image width in pixels.
        height: Output image height in pixels.
        padding: Padding around the court in pixels.

    Returns:
        BGR numpy array of the court image.
    """
    img = np.full((height, width, 3), COURT_COLOR, dtype=np.uint8)

    def ft_to_px(x_ft: float, y_ft: float) -> tuple[int, int]:
        return court_to_minimap(x_ft, y_ft, width, height, padding)

    def ft_to_radius(r_ft: float) -> int:
        draw_w = width - 2 * padding
        scale = draw_w / COURT_LENGTH
        return max(1, int(r_ft * scale))

    line_t = max(1, width // 400)  # line thickness scales with image size

    # ── Court outline ────────────────────────────────────────────────────
    tl = ft_to_px(0, 0)
    tr = ft_to_px(COURT_LENGTH, 0)
    bl = ft_to_px(0, COURT_WIDTH)
    br = ft_to_px(COURT_LENGTH, COURT_WIDTH)
    cv2.rectangle(img, tl, br, LINE_COLOR, line_t)

    # ── Half court line ──────────────────────────────────────────────────
    ht = ft_to_px(HALF_COURT_X, 0)
    hb = ft_to_px(HALF_COURT_X, COURT_WIDTH)
    cv2.line(img, ht, hb, LINE_COLOR, line_t)

    # ── Center circle ────────────────────────────────────────────────────
    center = ft_to_px(HALF_COURT_X, COURT_WIDTH / 2)
    r_center = ft_to_radius(CENTER_CIRCLE_RADIUS)
    cv2.circle(img, center, r_center, LINE_COLOR, line_t)

    # ── Left key (paint) ─────────────────────────────────────────────────
    _draw_key(img, ft_to_px, ft_to_radius, line_t, side="left")

    # ── Right key (paint) ────────────────────────────────────────────────
    _draw_key(img, ft_to_px, ft_to_radius, line_t, side="right")

    # ── Three-point lines ────────────────────────────────────────────────
    _draw_three_point(img, ft_to_px, ft_to_radius, line_t, side="left")
    _draw_three_point(img, ft_to_px, ft_to_radius, line_t, side="right")

    # ── Restricted areas ─────────────────────────────────────────────────
    _draw_restricted_area(img, ft_to_px, ft_to_radius, line_t, side="left")
    _draw_restricted_area(img, ft_to_px, ft_to_radius, line_t, side="right")

    # ── Baskets ──────────────────────────────────────────────────────────
    _draw_basket(img, ft_to_px, ft_to_radius, line_t, side="left")
    _draw_basket(img, ft_to_px, ft_to_radius, line_t, side="right")

    return img


def _draw_key(img, ft_to_px, ft_to_radius, line_t, side):
    """Draw the key (paint area) for one side."""
    if side == "left":
        x_base = 0.0
        x_ft = FT_LINE_DIST
    else:
        x_base = COURT_LENGTH
        x_ft = COURT_LENGTH - FT_LINE_DIST

    # Key rectangle
    p1 = ft_to_px(min(x_base, x_ft), KEY_TOP)
    p2 = ft_to_px(max(x_base, x_ft), KEY_BOTTOM)
    cv2.rectangle(img, p1, p2, LINE_COLOR, line_t)

    # Free throw circle (top half or bottom half depending on side)
    center = ft_to_px(x_ft, COURT_WIDTH / 2)
    r = ft_to_radius(FT_CIRCLE_RADIUS)
    cv2.circle(img, center, r, LINE_COLOR, line_t)


def _draw_three_point(img, ft_to_px, ft_to_radius, line_t, side):
    """Draw the three-point arc for one side."""
    if side == "left":
        basket_x = BASKET_OFFSET
    else:
        basket_x = COURT_LENGTH - BASKET_OFFSET

    basket_center = ft_to_px(basket_x, COURT_WIDTH / 2)
    r = ft_to_radius(THREE_PT_RADIUS)

    # The arc — compute angles
    # Corner three lines (straight portions along sideline)
    corner_y_top = (COURT_WIDTH / 2) - THREE_PT_RADIUS
    corner_y_bottom = (COURT_WIDTH / 2) + THREE_PT_RADIUS

    # The straight corner portions
    if side == "left":
        # Top corner line (sideline to where arc starts)
        cv2.line(img, ft_to_px(0, 3.0), ft_to_px(THREE_PT_BREAK_X, 3.0), LINE_COLOR, line_t)
        # Bottom corner line
        cv2.line(img, ft_to_px(0, COURT_WIDTH - 3.0), ft_to_px(THREE_PT_BREAK_X, COURT_WIDTH - 3.0), LINE_COLOR, line_t)
        # Arc: from ~top to ~bottom
        start_angle = -90 - 68  # roughly covers the arc
        end_angle = 90 + 68
        cv2.ellipse(img, basket_center, (r, r), 0, start_angle, end_angle, LINE_COLOR, line_t)
    else:
        cv2.line(img, ft_to_px(COURT_LENGTH, 3.0), ft_to_px(COURT_LENGTH - THREE_PT_BREAK_X, 3.0), LINE_COLOR, line_t)
        cv2.line(img, ft_to_px(COURT_LENGTH, COURT_WIDTH - 3.0), ft_to_px(COURT_LENGTH - THREE_PT_BREAK_X, COURT_WIDTH - 3.0), LINE_COLOR, line_t)
        cv2.ellipse(img, basket_center, (r, r), 0, 112, 248, LINE_COLOR, line_t)


def _draw_restricted_area(img, ft_to_px, ft_to_radius, line_t, side):
    """Draw the restricted area arc near the basket."""
    if side == "left":
        basket_x = BASKET_OFFSET
    else:
        basket_x = COURT_LENGTH - BASKET_OFFSET

    center = ft_to_px(basket_x, COURT_WIDTH / 2)
    r = ft_to_radius(RESTRICTED_ARC_RADIUS)

    if side == "left":
        cv2.ellipse(img, center, (r, r), 0, -90, 90, LINE_COLOR, line_t)
    else:
        cv2.ellipse(img, center, (r, r), 0, 90, 270, LINE_COLOR, line_t)


def _draw_basket(img, ft_to_px, ft_to_radius, line_t, side):
    """Draw the backboard and rim."""
    if side == "left":
        basket_x = BASKET_OFFSET
        bb_x = BASKET_OFFSET - 0.5
    else:
        basket_x = COURT_LENGTH - BASKET_OFFSET
        bb_x = COURT_LENGTH - BASKET_OFFSET + 0.5

    # Backboard (3ft wide line)
    bb_top = ft_to_px(bb_x, COURT_WIDTH / 2 - 3)
    bb_bottom = ft_to_px(bb_x, COURT_WIDTH / 2 + 3)
    cv2.line(img, bb_top, bb_bottom, BACKBOARD_COLOR, line_t + 1)

    # Rim (circle, 9in radius ≈ 0.75ft)
    rim_center = ft_to_px(basket_x, COURT_WIDTH / 2)
    rim_r = ft_to_radius(0.75)
    cv2.circle(img, rim_center, max(rim_r, 2), RIM_COLOR, line_t)


def generate_half_court_image(
    side: str,
    width: int = 470,
    height: int = 500,
    padding: int = 10,
) -> np.ndarray:
    """Generate a top-down view of one half of the court (47ft × 50ft).

    The active half fills the canvas, doubling the on-screen scale of a
    full-court minimap. The half-court line is drawn at the canvas edge
    closest to the inactive half (right edge for side="left", left edge
    for side="right"), with a fade so it's clear that's the boundary.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    img = np.full((height, width, 3), COURT_COLOR, dtype=np.uint8)

    from court import court_to_minimap_half

    def ft_to_px(x_ft: float, y_ft: float) -> tuple[int, int]:
        return court_to_minimap_half(x_ft, y_ft, side, width, height, padding)

    def ft_to_radius(r_ft: float) -> int:
        # Half-court x range is 47ft; the canvas width represents 47ft.
        draw_w = width - 2 * padding
        scale = draw_w / (COURT_LENGTH / 2)
        return max(1, int(r_ft * scale))

    line_t = max(1, width // 400)

    # ── Outline of just this half ────────────────────────────────────────
    if side == "left":
        x_lo, x_hi = 0.0, COURT_LENGTH / 2
    else:
        x_lo, x_hi = COURT_LENGTH / 2, COURT_LENGTH

    cv2.rectangle(img, ft_to_px(x_lo, 0), ft_to_px(x_hi, COURT_WIDTH), LINE_COLOR, line_t)

    # ── Half-court line (the inner edge of this half) + center circle arc
    cv2.line(
        img, ft_to_px(HALF_COURT_X, 0), ft_to_px(HALF_COURT_X, COURT_WIDTH),
        LINE_COLOR, line_t,
    )
    # Half of the center circle that's on this side
    center = ft_to_px(HALF_COURT_X, COURT_WIDTH / 2)
    r_center = ft_to_radius(CENTER_CIRCLE_RADIUS)
    if side == "left":
        cv2.ellipse(img, center, (r_center, r_center), 0, 90, 270, LINE_COLOR, line_t)
    else:
        cv2.ellipse(img, center, (r_center, r_center), 0, -90, 90, LINE_COLOR, line_t)

    # ── Paint, 3pt arc, restricted area, basket — only for this side ─────
    _draw_key(img, ft_to_px, ft_to_radius, line_t, side=side)
    _draw_three_point(img, ft_to_px, ft_to_radius, line_t, side=side)
    _draw_restricted_area(img, ft_to_px, ft_to_radius, line_t, side=side)
    _draw_basket(img, ft_to_px, ft_to_radius, line_t, side=side)

    return img


if __name__ == "__main__":
    court = generate_court_image()
    cv2.imwrite("court_template.png", court)
    print(f"Court template saved: {court.shape[1]}x{court.shape[0]}")
    half_l = generate_half_court_image("left")
    half_r = generate_half_court_image("right")
    cv2.imwrite("court_template_half_left.png", half_l)
    cv2.imwrite("court_template_half_right.png", half_r)
    print(f"Half-court templates: {half_l.shape[1]}x{half_l.shape[0]}")
