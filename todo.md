# Basketball Court Tracker — TODO

## Done
- [x] court.py — NBA court geometry, landmarks, keypoint map
- [x] court_template.py — programmatic court diagram generation
- [x] tests/test_court.py — 13 passing tests
- [x] homography.py — per-frame homography compute/validate/cache/transform
- [x] tests/test_homography.py — 9 passing tests
- [x] config.py — centralized config with CLI args
- [x] detector.py — Roboflow API player + court keypoint detection
- [x] tracker.py — ByteTrack player tracking
- [x] mapper.py — homography + detection → court coords + smoothing
- [x] visualizer.py — minimap + overlay + composite renderers
- [x] pipeline.py — full frame loop orchestrator
- [x] main.py — CLI entry point

## Next: Calibration & Testing
- [ ] Run court keypoint model on test frames to get actual class names
- [ ] Update ROBOFLOW_KEYPOINT_MAP in court.py with real class names
- [ ] End-to-end test on a basketball clip
- [ ] Fix any mapping issues from actual model output

## Polish
- [ ] Team color clustering (KMeans on jersey crops)
- [ ] Proper track_id propagation through mapper (currently uses bbox hash)
- [ ] CSV export verification
- [ ] Debug mode overlay testing
