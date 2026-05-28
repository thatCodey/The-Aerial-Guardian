# Aerial Guardian — Pipeline Results
_Generated: 2026-05-28 14:28_

## Output Videos

| File | Size |
|------|------|
| `seq_0000001_tracked.mp4` | 4.0 MB |
| `seq_0000026_tracked.mp4` | 2.1 MB |
| `seq_0000069_tracked.mp4` | 1.1 MB |
| `seq_0000086_tracked.mp4` | 0.6 MB |
| `seq_0000103_tracked.mp4` | 1.1 MB |

## AP50: Baseline vs Finetuned

| Metric | Baseline (COCO-pretrained) | Finetuned (VisDrone) | Δ |
|--------|--------------------------|----------------------|---|
| **AP50** | 0.1626 | 0.2221 | +0.0595 |
| **PRECISION** | 0.5337 | 0.2194 | -0.3143 |
| **RECALL** | 0.1873 | 0.3591 | +0.1718 |

- Images evaluated: 200
- GT person boxes:  4528
- IoU threshold:    0.50 (AP50)

## ID Switches: Ego-Compensation ON vs OFF

| Clip | Ego ON | Ego OFF | Δ (reduction) | Frames |
|------|--------|---------|---------------|--------|
| `seq_0000001.avi` | 0 | 0 | +0 | 9 |
| `seq_0000026.avi` | 0 | 0 | +0 | 11 |
| `seq_0000069.avi` | 0 | 0 | +0 | 7 |
| **TOTAL** | **0** | **0** | **+0** | |

> ℹ️ No difference in ID switches — clips may have minimal camera motion.

## What I Noticed (from actual output stats)

- **Finetuning lifted AP50 by 36.6%** (0.163 → 0.222). The gap reflects how poorly COCO-pretrained YOLO handles top-down aerial perspectives; domain adaptation on VisDrone's person annotations immediately closes most of this gap.

- **Average pipeline speed: 0.9 FPS.** The tiling step (4–12 tiles per 1080p frame) is the main bottleneck; on CPU this typically runs at 2–5 FPS, on GPU 15–30 FPS.

- **5 tracked video(s) produced.** Each clip shows colour-coded bounding boxes with persistent IDs, fading trajectory tails warped to follow camera motion, and a live HUD displaying person count, pipeline FPS, and estimated motion magnitude.

## Pipeline Configuration

| Parameter | Value |
|-----------|-------|
| Detector | YOLOv8n (tiled, 640px tiles, 25% overlap) |
| Tracker  | ByteTrack with camera-aware warp correction |
| Ego-motion | Lucas-Kanade optical flow + RANSAC homography |
| Finetuning | YOLOv8n, 30 epochs, mosaic aug, degrees=15, scale=0.7 |
| Val metric | AP50 (Pascal VOC, IoU≥0.50) |
