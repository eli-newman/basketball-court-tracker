# Basketball Court Tracker — Spec

## Vision
Any basketball video goes in → a 2D minimap with player dots on a top-down court diagram comes out, plus full coordinate data. Works regardless of camera pans, zooms, or cuts because we recalculate court-to-pixel mapping every frame.

## Architecture

### Pipeline
1. **Court Keypoint Detection** — Roboflow model detects court landmarks (corners, free throw lines, 3pt arc points)
2. **Player Detection** — Roboflow model detects all players per frame
3. **Homography** — Compute per-frame 3x3 transform matrix from detected keypoints → known court coordinates
4. **Tracking** — ByteTrack assigns persistent IDs across frames
5. **Coordinate Mapping** — Transform player pixel positions to real court coordinates (feet)
6. **Visualization** — Render 2D minimap with player dots + side-by-side with original video

### Court Coordinate System
- NBA court: 94ft x 50ft
- Origin (0, 0) at top-left corner
- X-axis: length (0-94ft, baseline to baseline)
- Y-axis: width (0-50ft, sideline to sideline)

### Per-Frame Output
- Player court positions (x, y) in feet
- Player track IDs (persistent across frames)
- Homography validity flag
- Number of keypoints detected

### Output
1. `output_composite.mp4` — side-by-side: annotated original + 2D minimap
2. `output_coordinates.json` — per-frame player positions in court coordinates
3. `output_coordinates.csv` — same data in CSV format

## Tech Stack
- Python 3.10+
- OpenCV (video I/O, homography, drawing)
- Roboflow API (player detection + court keypoint detection)
- supervision (ByteTrack tracking)
- numpy (math)
- scikit-learn (team color clustering)

## Models
- Player detection: `basketball-player-detection-3-ycjdo/6` (Roboflow)
- Court keypoints: `basketball-court-detection-2/13` (Roboflow)

## Target
- Works on any basketball footage (broadcast, gym camera)
- Handles camera pans, zooms, cuts via per-frame homography
- Offline processing (not real-time)
