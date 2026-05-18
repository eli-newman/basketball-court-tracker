"""Team classification by jersey color, aggregated per track.

Per-frame classification has two failure modes that show up in real footage:

1. **Occlusion contamination.** When a defender stands tight on an offensive
   player, their bboxes overlap. The chest crop for player A then includes
   pixels from player B's jersey — both end up with the same median color,
   both get assigned the same team.

2. **Per-frame jitter.** A single bad frame (motion blur, weird angle,
   referee crossing through) flips a player's team_id back and forth.

This module fixes both by:
  - Masking out pixels that fall inside *other* players' bboxes when
    extracting a player's chest color (occlusion-aware extraction).
  - Accumulating chest color samples per `track_id` over time, and
    fitting KMeans on **one median color per track**, not per-frame.
  - Returning a stable team_id per track that's recomputed on every refit
    rather than on every frame.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from detector import PlayerDetection


# Sentinel for "no team yet" — used until enough data is collected to fit.
UNKNOWN_TEAM = -1


# ── Flash-frame gate ────────────────────────────────────────────────────────
# Anomalous-lighting frames (replay flashes, transition graphics, scoreboard
# overlays bleeding into the broadcast feed) produce chest crops that are
# wildly desaturated relative to normal frames. If we let those crops into
# the rolling sample buffer, the per-track median feature drifts toward the
# "washed-out" point in feature space, and the next anchored re-assignment
# pass can flip a player to the wrong team.
#
# Threshold rationale (verified on the Knicks/Sixers clip's frame 90):
#   - normal arena lighting:  mean_V ≈ 130-140, mean_S ≈ 95-105
#   - bright flash / replay:  mean_V ≈ 180-200, mean_S ≈ 50-80
# 170 V triggers the gate alone (any normal broadcast frame sits well
# under 170 because the crowd + court keep the mean down). Saturation
# is a SECONDARY signal — we only use it when V is already moderately
# elevated, to catch the "less violent flash" case (V≈155 + S≈40).
# Pure-saturation isn't enough because synthetic-color test frames
# with large white areas can hit mean_S≈50 even though they're stable.
_FLASH_MEAN_V_HIGH = 170.0       # very bright frame by itself
_FLASH_MEAN_V_BORDERLINE = 155.0 # bright-ish — only flag if also desat
_FLASH_MEAN_S_LOW = 50.0         # paired-with-borderline-V threshold


def is_anomalous_frame(frame: np.ndarray) -> bool:
    """True iff the frame has unusually bright or unusually unsaturated
    color statistics — the kind of frame that produces unreliable chest
    crops and should be skipped for color-sample aggregation.

    The check is intentionally cheap (one HSV conversion, two means) so
    it can run unconditionally on every frame.
    """
    if frame is None or frame.size == 0:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[..., 2].mean())
    mean_s = float(hsv[..., 1].mean())
    if mean_v > _FLASH_MEAN_V_HIGH:
        return True
    if mean_v > _FLASH_MEAN_V_BORDERLINE and mean_s < _FLASH_MEAN_S_LOW:
        return True
    return False


# ── Canonical NBA team color signatures ─────────────────────────────────────
# Each profile is a 4-D vector matching the feature order produced by
# `_extract_jersey_color`:
#
#   [body_v (0-255), body_s (0-255), accent_red_frac, accent_blue_frac]
#
# Body lightness (V) dominates the matching — it's the single most reliable
# signal because almost every NBA matchup has one team in light jerseys and
# one in dark. Saturation distinguishes "white" (low-S) from "saturated dark
# color" (high-S). Red/blue accent fractions are tiebreakers for matchups
# where bodies are at similar lightness.
#
# Most teams have TWO common uniforms — light/home and dark/road. The lookup
# accepts either the bare team name (defaults to ROAD/dark for clarity) or
# a "<team>-<variant>" form. Variants: "road", "home", "city", "statement".
# Unknown variant → fall back to bare name.
#
# Anchor V values are calibrated against ACTUAL OBSERVED broadcast samples
# from the shoulders ROI — not synthetic pure-color values:
#
#   - "Dark" jersey body (navy, black, wine, purple, green) → V≈75-95
#   - "Light" jersey body (white, yellow) → V≈140-170
#
# Real shoulder crops include skin around the neckline + shadows under the
# chin + court reflection, which compresses both ends of the V range. A
# perfectly-white pixel is V=255, but a "white jersey shoulders" sample
# averages to V≈150. Same with navy — the pure pixels are V≈40 but the
# crop averages to V≈80.
#
# The midpoint between "dark" and "light" sits around V=115. Anchors are
# spaced symmetrically around that midpoint so distance to either anchor
# resolves cleanly.
TEAM_PROFILES: Dict[str, np.ndarray] = {
    # ── Eastern Conference ──────────────────────────────────────────────────
    "knicks":         np.array([85, 130, 0.05, 0.40], dtype=np.float32),  # navy road
    "knicks-road":    np.array([85, 130, 0.05, 0.40], dtype=np.float32),
    "knicks-home":    np.array([155, 50, 0.10, 0.15], dtype=np.float32),  # white

    "sixers":         np.array([155, 50, 0.30, 0.20], dtype=np.float32),  # white home
    "sixers-home":    np.array([155, 50, 0.30, 0.20], dtype=np.float32),
    "sixers-road":    np.array([95, 160, 0.45, 0.15], dtype=np.float32),  # red road

    "celtics":        np.array([90, 150, 0.05, 0.05], dtype=np.float32),  # green road
    "celtics-road":   np.array([90, 150, 0.05, 0.05], dtype=np.float32),
    "celtics-home":   np.array([160, 45, 0.10, 0.10], dtype=np.float32),

    "bulls":          np.array([90, 160, 0.55, 0.05], dtype=np.float32),  # red road
    "bulls-road":     np.array([90, 160, 0.55, 0.05], dtype=np.float32),
    "bulls-home":     np.array([160, 50, 0.40, 0.10], dtype=np.float32),

    "heat":           np.array([85, 170, 0.55, 0.05], dtype=np.float32),
    "heat-road":      np.array([85, 170, 0.55, 0.05], dtype=np.float32),
    "heat-home":      np.array([155, 50, 0.35, 0.10], dtype=np.float32),

    "bucks":          np.array([85, 150, 0.05, 0.05], dtype=np.float32),
    "bucks-home":     np.array([160, 45, 0.10, 0.10], dtype=np.float32),

    "pistons":        np.array([95, 130, 0.40, 0.40], dtype=np.float32),
    "hawks":          np.array([90, 160, 0.50, 0.10], dtype=np.float32),
    "magic":          np.array([95, 150, 0.10, 0.55], dtype=np.float32),
    "pacers":         np.array([95, 150, 0.10, 0.55], dtype=np.float32),
    "nets":           np.array([60, 40, 0.05, 0.05], dtype=np.float32),
    "nets-home":      np.array([160, 35, 0.05, 0.05], dtype=np.float32),
    "raptors":        np.array([90, 160, 0.55, 0.10], dtype=np.float32),
    "cavs":           np.array([80, 150, 0.45, 0.10], dtype=np.float32),
    "wizards":        np.array([95, 140, 0.35, 0.40], dtype=np.float32),
    "hornets":        np.array([85, 140, 0.05, 0.25], dtype=np.float32),

    # ── Western Conference ──────────────────────────────────────────────────
    "lakers":         np.array([95, 150, 0.05, 0.10], dtype=np.float32),
    "lakers-home":    np.array([170, 180, 0.10, 0.10], dtype=np.float32),  # gold
    "warriors":       np.array([90, 160, 0.10, 0.55], dtype=np.float32),
    "warriors-home":  np.array([160, 50, 0.10, 0.35], dtype=np.float32),
    "nuggets":        np.array([90, 160, 0.10, 0.55], dtype=np.float32),
    "mavs":           np.array([90, 170, 0.10, 0.60], dtype=np.float32),
    "mavs-home":      np.array([160, 45, 0.10, 0.45], dtype=np.float32),
    "kings":          np.array([90, 150, 0.10, 0.10], dtype=np.float32),
    "thunder":        np.array([90, 170, 0.10, 0.55], dtype=np.float32),
    "rockets":        np.array([90, 170, 0.55, 0.10], dtype=np.float32),
    "jazz":           np.array([85, 170, 0.10, 0.35], dtype=np.float32),
    "suns":           np.array([95, 150, 0.10, 0.10], dtype=np.float32),
    "blazers":        np.array([85, 160, 0.45, 0.10], dtype=np.float32),
    "spurs":          np.array([70, 40, 0.05, 0.05], dtype=np.float32),
    "spurs-home":     np.array([160, 30, 0.05, 0.05], dtype=np.float32),
    "grizzlies":      np.array([90, 160, 0.10, 0.50], dtype=np.float32),
    "pelicans":       np.array([60, 60, 0.05, 0.05], dtype=np.float32),
    "wolves":         np.array([65, 60, 0.05, 0.35], dtype=np.float32),
    "clippers":       np.array([90, 170, 0.40, 0.35], dtype=np.float32),
}

# Per-dimension weights for distance: body_v dominates because lightness is
# the most stable cross-clip signal. S is half as important — it disambig-
# uates white (low-S) from saturated colors but is noisier than V. Accent
# fractions are tiebreakers (weight 1.0) but max out at ~0.6 so their
# squared contribution is bounded.
_FEATURE_WEIGHTS = np.array([0.040, 0.010, 1.0, 1.0], dtype=np.float32)

# Display BGR for each team (used by team_colors_bgr in anchored mode).
TEAM_DISPLAY_BGR: Dict[str, Tuple[int, int, int]] = {
    "knicks":    (200, 100, 30),
    "sixers":    (40, 40, 220),
    "celtics":   (40, 180, 40),
    "bulls":     (60, 40, 200),
    "heat":      (40, 60, 200),
    "bucks":     (40, 100, 40),
    "pistons":   (180, 40, 40),
    "hawks":     (40, 40, 220),
    "magic":     (200, 100, 40),
    "pacers":    (40, 80, 220),
    "nets":      (60, 30, 30),
    "raptors":   (40, 40, 200),
    "cavs":      (40, 30, 140),
    "wizards":   (180, 40, 40),
    "hornets":   (200, 40, 160),
    "lakers":    (200, 40, 160),
    "warriors":  (180, 100, 30),
    "nuggets":   (140, 60, 40),
    "mavs":      (220, 100, 40),
    "kings":     (180, 40, 140),
    "thunder":   (200, 120, 40),
    "rockets":   (40, 40, 200),
    "jazz":      (60, 60, 60),
    "suns":      (40, 100, 220),
    "blazers":   (40, 40, 200),
    "spurs":     (180, 180, 180),
    "grizzlies": (140, 100, 60),
    "pelicans":  (60, 30, 100),
    "wolves":    (140, 100, 30),
    "clippers":  (200, 40, 60),
}


def resolve_team_profile(name: str) -> Optional[np.ndarray]:
    """Look up a team's canonical color signature, case-insensitive.

    Returns None if the name isn't in the registry — caller decides whether
    that's a hard error or a "fall back to KMeans" signal.
    """
    if not name:
        return None
    key = name.strip().lower()
    return TEAM_PROFILES.get(key)


def _nearest_anchor(X: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """For each row in X, return the index of the closest anchor by weighted L2.

    The weights live in `_FEATURE_WEIGHTS` and are calibrated to make body_v
    dominate the distance — lightness is the most stable signal across
    broadcast clips, and saturation / accent fractions only tiebreak.

    Args:
        X: (n_tracks, F) feature matrix.
        anchors: (n_teams, F) anchor matrix.

    Returns:
        (n_tracks,) integer labels in [0, n_teams).
    """
    # Broadcast diff to (n_tracks, n_teams, F), square, weight, sum.
    diff = X[:, None, :] - anchors[None, :, :]
    sq = diff * diff
    weighted = sq * _FEATURE_WEIGHTS[None, None, :]
    distances = weighted.sum(axis=2)  # (n_tracks, n_teams)
    return np.argmin(distances, axis=1)


class TeamClassifier:
    """Classifies players into teams based on chest color aggregated per track."""

    def __init__(
        self,
        n_teams: int = 2,
        warmup_frames: int = 15,
        min_saturation: int = 60,
        samples_per_track: int = 30,
        refit_every: int = 10,
        min_samples_to_classify: int = 3,
        team_anchors: Optional[Sequence[np.ndarray]] = None,
        team_names: Optional[Sequence[str]] = None,
        debug_crop_dir: Optional[str] = None,
    ):
        """
        Args:
            n_teams: Number of teams to cluster (2 = just teams, 3 = teams + refs).
            warmup_frames: Don't fit KMeans until at least this many frames have
                been seen. Bigger = more stable initial fit, slower to label.
            min_saturation: HSV-S threshold for "team color" pixels. Drops
                washed-out whites/grays so the team accent dominates the median.
            samples_per_track: Rolling window of per-frame color samples kept
                for each tracked player. Old samples drop off as new ones land.
            refit_every: Refit KMeans (re-cluster all tracks) every N frames
                once warmed up. Keeps assignments stable but adaptive to new
                tracks that appear later.
            min_samples_to_classify: A track needs at least this many color
                samples before it's eligible to be clustered.
            team_anchors: Optional list of N=n_teams canonical color signatures
                (each a 7-D vector matching the feature order of
                `_extract_jersey_color`). If provided, the classifier skips
                KMeans entirely and assigns each track to the anchor with the
                highest cosine similarity to its median. This is the
                "supervised" mode — use it whenever you know which two teams
                are playing (which is always true for NBA broadcasts).
            team_names: Optional list of N display names aligned with
                `team_anchors`. Used by `team_colors_bgr` to look up canonical
                BGR colors in `TEAM_DISPLAY_BGR`.
        """
        self.n_teams = n_teams
        self._kmeans: Optional[KMeans] = None
        self._calibrated = False
        self._warmup_frames = warmup_frames
        self._min_saturation = min_saturation
        self._samples_per_track = samples_per_track
        self._refit_every = refit_every
        self._min_samples_to_classify = min_samples_to_classify

        # Supervised mode state. When anchors are present we don't need a
        # warmup at all — the anchors are calibrated — so collapse the
        # warmup + min-samples thresholds to 1. New tracks get classified
        # the moment they're detected rather than sitting unclassified
        # for 15+ frames.
        self._team_anchors: Optional[np.ndarray] = None
        self._team_names: Optional[List[str]] = None
        if team_anchors is not None:
            self._warmup_frames = 1
            self._min_samples_to_classify = 1
            self._refit_every = 1
            anchors = np.stack([np.asarray(a, dtype=np.float32) for a in team_anchors], axis=0)
            if anchors.shape[0] != n_teams:
                raise ValueError(
                    f"team_anchors has {anchors.shape[0]} rows but n_teams={n_teams}"
                )
            self._team_anchors = anchors
            if team_names is not None:
                if len(team_names) != n_teams:
                    raise ValueError(
                        f"team_names has {len(team_names)} entries but n_teams={n_teams}"
                    )
                self._team_names = list(team_names)

        # Per-track state
        self._track_samples: Dict[int, Deque[np.ndarray]] = {}
        # Single-slot fallback used ONLY for tracks that have no clean
        # sample yet (heavily-occluded from the moment they appear, e.g.
        # a player in a tight defensive set). Cleared the instant any
        # clean (occlusion-masked) sample lands. Keeps the contaminated
        # bootstrap signal from poisoning the long-term median.
        self._track_bootstrap: Dict[int, np.ndarray] = {}
        self._track_assignments: Dict[int, int] = {}  # track_id -> team_id
        # Set per-call by `_extract_jersey_color` so `classify()` can
        # route the returned feature to the right buffer.
        self._last_extract_was_fallback: bool = False

        # Bookkeeping
        self._frame_count = 0
        self._frames_since_refit = 0
        # Diagnostic: how many frames the flash gate rejected during the
        # run. Surfaced via `n_flash_frames_skipped` for the end-of-run
        # summary so we can tell if the gate fired too often (would
        # indicate the thresholds are too tight for this broadcast).
        self._flash_frames_skipped = 0

        # Debug: dump every chest crop we sample to disk. When set, each
        # sampled ROI is written to `<dir>/track_<id>_<frame>.png` with the
        # extracted feature in the filename. Lets us visually verify that
        # the ROI is on the jersey body and not on the head/background.
        self._debug_crop_dir: Optional[str] = debug_crop_dir
        if debug_crop_dir:
            import os as _os
            _os.makedirs(debug_crop_dir, exist_ok=True)

    def classify(
        self,
        frame: np.ndarray,
        players: List[PlayerDetection],
    ) -> List[int]:
        """Extract per-player chest color, accumulate, return current team_ids.

        Args:
            frame: BGR frame.
            players: tracked players (track_id should be set; -1 untracked is
                tolerated but those players are skipped from the aggregator).

        Returns:
            team_id per player in the same order as the input. -1 until the
            track has enough samples and KMeans has been fit.
        """
        if not players:
            return []
        self._frame_count += 1

        # Flash gate: skip sample accumulation entirely on anomalous-
        # lighting frames (replay flashes, transition graphics). The
        # crops on those frames are wildly desaturated and would shift
        # each track's rolling-median feature in the wrong direction —
        # which then causes a wrong-team flip on the very next normal
        # frame. We still RETURN current assignments built from the
        # existing buffer so the rest of the pipeline keeps a stable
        # team_id stream.
        if is_anomalous_frame(frame):
            self._flash_frames_skipped += 1
            return [
                self._track_assignments.get(p.track_id, UNKNOWN_TEAM)
                for p in players
            ]

        # 1) Collect this frame's chest colors, masking each player's other
        #    overlapping bboxes out so defender/offensive-player overlap can't
        #    contaminate the sample. Samples are routed to the main buffer
        #    when extracted with the occlusion mask (clean) or to the per-
        #    track single-slot bootstrap when the mask had to be dropped.
        all_bboxes = [p.bbox for p in players]
        for i, player in enumerate(players):
            if player.track_id < 0:
                continue
            others = all_bboxes[:i] + all_bboxes[i + 1:]
            self._last_extract_was_fallback = False  # reset per call
            color = self._extract_jersey_color(frame, player, others)
            if color is None:
                continue
            if self._last_extract_was_fallback:
                # Bootstrap only — store as the single-slot fallback,
                # but ONLY if we don't already have any clean samples.
                # Once a clean sample exists, fallbacks are ignored.
                if not self._track_samples.get(player.track_id):
                    self._track_bootstrap[player.track_id] = color
            else:
                buf = self._track_samples.setdefault(
                    player.track_id, deque(maxlen=self._samples_per_track),
                )
                buf.append(color)
                # First clean sample for this track — drop the
                # (possibly contaminated) bootstrap.
                self._track_bootstrap.pop(player.track_id, None)

        # 2) Refit KMeans periodically once warmed up.
        self._frames_since_refit += 1
        if (
            self._frame_count >= self._warmup_frames
            and self._frames_since_refit >= self._refit_every
        ):
            self._refit()
            self._frames_since_refit = 0

        # 3) Return current assignments in input order.
        return [
            self._track_assignments.get(p.track_id, UNKNOWN_TEAM)
            for p in players
        ]

    def _extract_jersey_color(
        self,
        frame: np.ndarray,
        player: PlayerDetection,
        other_bboxes: List[tuple],
    ) -> Optional[np.ndarray]:
        """Extract a 4-D body-color feature from the player's shoulders.

        Returns ``[body_v, body_s, accent_red, accent_blue]``:

          - **body_v** (0-255): median HSV-V of body pixels. The primary
            discriminator. Light jersey (white/gold) ≈ 200+. Dark jersey
            (navy/black/wine) ≈ 50-100. This single dimension separates
            most NBA matchups — one team always wears light, one dark.
          - **body_s** (0-255): median HSV-S of body pixels. Light/white
            jerseys are low-S; saturated colored bodies (navy, purple,
            green) are high-S. Helps distinguish "white vs colored" when
            both bodies are at similar brightness.
          - **accent_red, accent_blue**: proportion of saturated chest
            pixels that fall into the red or blue hue ranges. Used as a
            tiebreaker when both teams have similar body lightness
            (e.g., two-light matchup Sixers @ Mavs).

        The ROI is **shoulders** (top 10-30% vertical, center 40% horiz),
        which is the cleanest piece of jersey: above the number, below
        the neck/face. Previously we sampled the JERSEY NUMBER band,
        which mixed body color and number color into the median and
        smeared the two teams together.

        Skin pixels (faces, arms) are masked out by a coarse skin-tone
        gate so the body sample is dominated by the actual fabric.
        """
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        fh, fw = frame.shape[:2]
        x1 = max(0, min(x1, fw - 1))
        x2 = max(0, min(x2, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        y2 = max(0, min(y2, fh - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        # Shoulders/upper-chest ROI: 22-42% vertical, center 40% horizontal.
        # This lands BELOW the head/neck (top ~20% of body) and ABOVE the
        # jersey number (typically at 40-55%). Pure body color, no
        # contamination from face skin or the number itself.
        #
        # Empirically (logging on broadcast clips), 10-30% was too high —
        # it mostly captured skin + hair, and white jerseys read with the
        # same V as navy because both samples were ~50% face.
        bh = y2 - y1
        bw = x2 - x1
        cy1 = y1 + int(bh * 0.22)
        cy2 = y1 + int(bh * 0.42)
        cx1 = x1 + int(bw * 0.30)
        cx2 = x2 - int(bw * 0.30)
        if cy2 <= cy1 or cx2 <= cx1:
            cy1, cy2, cx1, cx2 = y1, y2, x1, x2

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Skin mask: coarse skin-tone filter covering light AND dark skin.
        # Skin pixels in HSV: hue H in [0, 25] (red-orange wraparound) OR
        # H in [170, 180] (red wraparound on the other side), with moderate
        # saturation. Critically the V range is wide — dark skin in
        # broadcast shadows reads as V≈50-90, the SAME range as a navy
        # jersey, so without a wide V floor the skin mask misses dark
        # faces entirely and the body-V median collapses to "navy" for
        # whichever Sixers player got cropped with their head in frame.
        # V ceiling is 200, NOT 255, so we don't accidentally mask out
        # very-bright orange numbers (which sit at V≈200-230, S≈220+).
        skin_mask = (
            ((h <= 25) | (h >= 170))
            & (s >= 20) & (s <= 200)
            & (v >= 40) & (v <= 200)
        )

        # Occlusion mask: drop pixels inside any other player's bbox.
        H, W = crop.shape[:2]
        occlusion = np.ones((H, W), dtype=bool)
        if other_bboxes:
            for ox1, oy1, ox2, oy2 in other_bboxes:
                ix1 = max(0, int(ox1) - cx1)
                iy1 = max(0, int(oy1) - cy1)
                ix2 = min(W, int(ox2) - cx1)
                iy2 = min(H, int(oy2) - cy1)
                if ix2 > ix1 and iy2 > iy1:
                    occlusion[iy1:iy2, ix1:ix2] = False

        # Valid body pixels: not skin, not occluded, not pure black or pure
        # white (extreme shadows / rim-light saturation).
        body = (~skin_mask) & occlusion & (v >= 20) & (v <= 250)

        # Reject the whole sample if too much of the ROI is skin — that
        # means the ROI mis-targeted the head/face and the remaining body
        # pixels are tiny edge slivers that don't represent the jersey.
        # Without this guard, samples like "75% face + 25% jersey
        # background sliver" let dark-skin tones masquerade as navy body.
        roi_area = H * W
        if skin_mask.sum() > 0.5 * roi_area:
            return None

        # Tier 1: strict — occlusion-masked, ≥15% body pixels. The
        # cleanest signal; everything we'd want to base a long-term
        # classification on.
        if body.sum() >= max(20, int(0.15 * roi_area)):
            self._last_extract_was_fallback = False
        else:
            # Tier 2: lenient occlusion-masked — ≥5% body. Still
            # uncontaminated by adjacent players (the mask did its
            # job), but the sample is small. Useful when a player is
            # close to another but not directly occluded.
            if body.sum() >= max(10, int(0.05 * roi_area)):
                self._last_extract_was_fallback = False
            else:
                # Tier 3: drop the occlusion mask entirely (bootstrap-
                # only). The sample WILL include neighbor-jersey pixels
                # if there's overlap, so it's tagged as a fallback —
                # the caller stores it in a separate single-slot buffer
                # that's used ONLY when no clean (tier-1/2) sample
                # exists for this track yet. Once any clean sample
                # lands, the fallback is discarded.
                body_no_occl = (~skin_mask) & (v >= 20) & (v <= 250)
                if body_no_occl.sum() < max(20, int(0.15 * roi_area)):
                    # Even without occlusion masking there's not
                    # enough body — ROI mostly off-frame or in shadow.
                    return None
                body = body_no_occl
                self._last_extract_was_fallback = True

        body_v = float(np.median(v[body]))
        body_s = float(np.median(s[body]))

        # Accent tiebreakers: fraction of *saturated* chest pixels that
        # are red or blue. Use a tighter saturation gate here so the
        # accent count reflects real number/trim pixels, not washed-out
        # body color.
        sat = body & (s >= max(80, self._min_saturation))
        total_sat = int(sat.sum()) + 1
        # Red wraps around the hue circle.
        accent_red = float(
            (((h <= 8) | (h >= 168)) & sat).sum()
        ) / total_sat
        accent_blue = float(
            ((h >= 95) & (h <= 130) & sat).sum()
        ) / total_sat

        feature = np.array(
            [body_v, body_s, accent_red, accent_blue],
            dtype=np.float32,
        )

        # Debug: save the crop we just measured, with the feature in the
        # filename, so we can verify visually what's being sampled.
        if self._debug_crop_dir:
            import os as _os
            stem = (
                f"track_{player.track_id:03d}_"
                f"f{self._frame_count:04d}_"
                f"V{int(body_v):03d}_S{int(body_s):03d}_"
                f"r{int(accent_red * 100):02d}_b{int(accent_blue * 100):02d}.png"
            )
            cv2.imwrite(_os.path.join(self._debug_crop_dir, stem), crop)

        return feature

    def _refit(self):
        """Recluster tracks: one median sample per track, fit, assign.

        Branches on `_team_anchors`:
          - Anchored: nearest-anchor by cosine similarity (no KMeans).
          - Unanchored: KMeans over per-track medians.
        """
        # Build "track_id -> median color" using only tracks with enough
        # samples. A track that has only the bootstrap fallback (no clean
        # samples) still gets included, using that single sample — that's
        # the whole point of the bootstrap: get the player ONTO a team
        # even if their first frames are heavily occluded. As soon as
        # the first clean sample lands, the bootstrap is cleared and
        # this loop ignores it.
        track_ids: List[int] = []
        track_colors: List[np.ndarray] = []
        seen = set()
        for tid, samples in self._track_samples.items():
            if len(samples) < self._min_samples_to_classify:
                continue
            track_ids.append(tid)
            track_colors.append(np.median(np.stack(samples, axis=0), axis=0))
            seen.add(tid)
        for tid, bootstrap in self._track_bootstrap.items():
            if tid in seen:
                continue   # clean sample already won this track
            track_ids.append(tid)
            track_colors.append(bootstrap)

        if not track_colors:
            return

        X = np.stack(track_colors, axis=0)

        if self._team_anchors is not None:
            # Supervised: each track → nearest team anchor by weighted L2.
            # The weights make body lightness dominate — that's the most
            # robust cross-clip signal (broadcast lighting varies, but
            # navy is always darker than white).
            labels = _nearest_anchor(X, self._team_anchors)
            self._track_assignments = {
                int(tid): int(label) for tid, label in zip(track_ids, labels)
            }
            self._calibrated = True
            self._log_anchored_assignments(X, track_ids, labels)
            return

        # Unsupervised: KMeans. Needs at least n_teams distinct tracks.
        if len(track_colors) < self.n_teams:
            return

        kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
        kmeans.fit(X)
        labels = kmeans.predict(X)

        self._kmeans = kmeans
        self._track_assignments = {
            int(tid): int(label) for tid, label in zip(track_ids, labels)
        }
        self._calibrated = True

        # Debug log: show the (V, S, accent) signature of each cluster center
        centers = kmeans.cluster_centers_
        for i, center in enumerate(centers):
            count = sum(1 for v in self._track_assignments.values() if v == i)
            print(
                f"  Team {i}: V={center[0]:5.1f} S={center[1]:5.1f} "
                f"r={center[2]:.2f} b={center[3]:.2f}  [{count} tracks]"
            )

    def _log_anchored_assignments(
        self,
        X: Optional[np.ndarray] = None,
        track_ids: Optional[List[int]] = None,
        labels: Optional[np.ndarray] = None,
    ):
        """Print per-team track counts + (when debug data passed) per-track
        feature vectors so we can see what the classifier is actually doing.
        """
        for i in range(self.n_teams):
            count = sum(1 for v in self._track_assignments.values() if v == i)
            name = (
                self._team_names[i]
                if self._team_names is not None
                else f"team {i}"
            )
            print(f"  {name}: {count} tracks")
        if X is not None and track_ids is not None and labels is not None:
            for tid, feat, lab in sorted(
                zip(track_ids, X, labels), key=lambda r: r[2],
            ):
                team = (
                    self._team_names[int(lab)]
                    if self._team_names is not None
                    else f"team{int(lab)}"
                )
                print(
                    f"    track {tid:>3} → {team:<10} "
                    f"V={feat[0]:5.1f} S={feat[1]:5.1f} "
                    f"r={feat[2]:.2f} b={feat[3]:.2f}"
                )

    def refit(self):
        """Public force-refit (e.g., after a detected scene change)."""
        self._refit()
        self._frames_since_refit = 0

    def reset_track_state(self):
        """Clear per-track sample buffers + assignments. Keeps the fitted
        model (anchors or KMeans) so the next shot's tracks can be
        classified immediately without re-warming.

        Use on a camera cut: when the tracker resets its IDs, the next
        new player gets an old track ID. Without this we'd average that
        new player's chest-color samples with the previous player's,
        producing a polluted median that drifts into unknown/wrong team.
        """
        self._track_samples.clear()
        self._track_bootstrap.clear()
        self._track_assignments.clear()
        # Keep _kmeans, _team_anchors, _calibrated — the model is global
        # to the clip, only per-track samples are per-shot.

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def n_flash_frames_skipped(self) -> int:
        """How many frames the flash gate rejected during the run.

        End-of-run summary surfaces this so we can tell if the gate
        fired too often (would indicate thresholds are too tight for
        this broadcast).
        """
        return self._flash_frames_skipped

    @property
    def team_colors_bgr(self) -> List[Tuple[int, int, int]]:
        """Display BGR per team.

        Anchored mode: use canonical NBA team colors via TEAM_DISPLAY_BGR.
        Unsupervised mode: synthesize a BGR triple by inverting the
        cluster center's HSV signature — dark clusters get a dark dot,
        light clusters get a light dot. Approximate but informative.
        """
        # Anchored mode: known team identities, return canonical colors.
        if self._team_anchors is not None:
            out: List[Tuple[int, int, int]] = []
            for i in range(self.n_teams):
                if self._team_names is not None:
                    # Look up by both bare and trimmed forms ("knicks-road"
                    # → fall back to "knicks" if no exact match).
                    raw = self._team_names[i].lower()
                    bgr = (
                        TEAM_DISPLAY_BGR.get(raw)
                        or TEAM_DISPLAY_BGR.get(raw.split("-")[0])
                    )
                    if bgr is not None:
                        out.append(bgr)
                        continue
                out.append((200, 200, 200))
            return out

        # Unsupervised mode: synthesize BGR from the cluster center V & S.
        if not self._calibrated or self._kmeans is None:
            return [(200, 200, 200)] * self.n_teams
        out = []
        for center in self._kmeans.cluster_centers_:
            v = int(np.clip(center[0], 0, 255))
            s = int(np.clip(center[1], 0, 255))
            # Light/low-S → near-white; dark/high-S → near-black.
            gray = max(0, min(255, v - s // 4))
            out.append((gray, gray, gray))
        return out
