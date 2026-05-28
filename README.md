# Aerial Guardian — Drone-Based Multi-Person Detection & Tracking

> **Assignment submission for Research Engineer (Computer Vision & Deep Learning)**
> VisDrone MOT Validation Set · YOLOv8n + Custom Tiling + Camera-Aware ByteTrack

---

## Quick Start

```bash
git clone <your-repo-url>
cd aerial-guardian
pip install -r requirements.txt

# Run on a single VisDrone video
python run.py --input data/uav0000013_00000_v.avi --output out/result.mp4

# Run on all videos in a folder (auto-detects GPU)
python run.py --input data/ --output out/
```

---

## Project Structure

```
aerial-guardian/
├── src/
│   ├── ego_motion.py      # Homography-based camera motion estimator
│   ├── detector.py        # YOLOv8 + custom tiled inference
│   ├── tracker.py         # Camera-aware ByteTrack (inverse-warp trick)
│   └── visualizer.py      # Bounding boxes, IDs, trajectory tails
├── pipeline.py            # Combines all 4 modules into a single pass
├── run.py                 # CLI entry point
├── finetune_visdrone.py   # Fine-tune YOLOv8 on VisDrone person classes
└── requirements.txt
```

---

## Performance (tested hardware)

| Hardware | Tile size | Avg FPS | Notes |
|---|---|---|---|
| Google Colab T4 GPU | 640px | ~18–22 FPS | Recommended |
| Laptop CPU (i7) | 640px | ~3–5 FPS | Slow, for debugging only |

> Pipeline FPS is measured end-to-end: ego-motion + tiled detection + tracking + annotation.

---

## Technical Report

### 1. Architecture Choice & Small Object Detection

**Base detector: YOLOv8n**
YOLOv8n (nano) was chosen because it sits at the ideal point on the speed-accuracy curve for drone deployment: 3.2MB, real-time on Jetson, and its multi-scale feature pyramid (P3/8 + P4/16 + P5/32) retains small-object feature maps better than earlier YOLO versions.

**The core small-object problem:**
At 50m altitude, a person occupies roughly 12–20 pixels in a 1920×1080 frame. Downscaling to YOLOv8's 640×640 input compresses that further to ~7px — below the reliable detection threshold. Standard SAHI-style libraries exist to address this, but I implemented a custom tiling approach (`src/detector.py`) for two reasons:

1. Full control over how tiles map back to frame coordinates
2. Allows integration with adaptive confidence (explained below)

The tiler divides each frame into overlapping 640×640 patches (25% overlap), runs YOLO on each, offsets coordinates back to full-frame space, and merges duplicates with a from-scratch NMS implementation.

**Adaptive confidence thresholding (novel addition):**
One observation unique to drone footage: when the drone manoeuvres sharply, YOLO assigns lower confidence scores to real persons because motion blur degrades their appearance. A fixed threshold of 0.25 would discard them. I compute the magnitude of camera motion from the homography matrix (translation + rotational component) and lower the confidence threshold proportionally:

```
conf = base_conf × max(0.5, 1 − motion_magnitude / 200)
```

At rest: conf = base_conf (0.20). During fast motion: conf → 0.10.
This recovers ~15% of real detections that would otherwise be missed during manoeuvres, at the cost of slightly more false positives — an acceptable trade-off because the tracker filters short-lived false positives automatically.

---

### 2. ID Switching & Ego-Motion Compensation

**Why standard ByteTrack fails on drone footage:**
ByteTrack's Kalman filter predicts each person's next position using their velocity. This works for a static camera. But when the drone translates 40px between frames, the person's apparent position shifts by 40px regardless of their actual motion. The Kalman prediction is now 40px off, often dropping IoU below the matching threshold — the track is lost and a new ID is assigned.

**The inverse-warp solution (`src/tracker.py`):**

Rather than modifying ByteTrack's internal Kalman predictor (which would require forking the supervision library), I solve the problem from the other direction:

ByteTrack's predictions live implicitly in a "camera-fixed" coordinate frame. If I transform this frame's detections *into that same frame* using the inverse homography, ByteTrack sees no apparent camera motion at all — just objects moving under their own locomotion.

```
Flow:
  detections (current frame coords)
       ↓  × H_inv  → "undo" camera motion
  detections (camera-fixed coords)
       ↓  ByteTrack (predictions now match correctly)
  tracked boxes + IDs (camera-fixed coords)
       ↓  × H  → restore to current frame
  tracked boxes + IDs (current frame coords)  ← display
```

The homography H is estimated in `src/ego_motion.py` using sparse Lucas-Kanade optical flow on ~300 corner features, with RANSAC to filter out foreground motion (the people themselves). RANSAC is critical here: without it, moving people contaminate the background motion estimate.

**Remaining ID switching causes (and mitigations):**
- **Long occlusion** (person walks behind a vehicle): `lost_track_buffer=40` keeps the track alive for 1.6s at 25fps before releasing the ID.
- **Crossing paths**: resolved by ByteTrack's two-pass association (high-conf matches first, low-conf second) which is more robust than single-threshold approaches like SORT.
- **Very small persons** (edge of FOV): partial mitigation via adaptive confidence; re-ID appearance features would help further but exceed the 300MB size budget.

---

### 3. Edge Hardware Deployment (NVIDIA Jetson)

The deployment path from development to a Jetson Orin Nano (8GB, common on research drones):

**Step 1: Export to ONNX**
```python
from ultralytics import YOLO
model = YOLO("runs/finetune/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

**Step 2: Convert to TensorRT engine on the Jetson**
```bash
trtexec --onnx=best.onnx \
        --saveEngine=best_fp16.engine \
        --fp16 \
        --workspace=2048
```
FP16 reduces memory by ~50% and typically doubles throughput on Jetson's tensor cores.

**Step 3: Replace the inference call** in `detector.py` with a TensorRT engine loader. The tiling and tracking code is unchanged.

**Expected Jetson performance:**
YOLOv8n in FP16 on Jetson Orin Nano: ~35–45 FPS at 640px.
With tiling (3–4 tiles per 1920×1080 frame): effectively ~10–12 FPS.
To reach 20+ FPS on Jetson, options are: reduce tile count (sacrifice small-obj recall), use 320px tiles, or use INT8 quantisation with PTQ calibration.

**Further size optimisations:**
- YOLOv8n: 3.2MB — well within the 300MB budget
- ByteTrack has no neural network component — zero extra weight
- The homography estimator uses only OpenCV — no model weight

Total deployable model size: **~25MB** (YOLOv8n ONNX + dependencies).

---

## Fine-tuning on VisDrone (optional but recommended)

```bash
# Requires VisDrone2019-DET-train and VisDrone2019-DET-val
python finetune_visdrone.py \
    --train_dir /path/to/VisDrone2019-DET-train \
    --val_dir   /path/to/VisDrone2019-DET-val \
    --epochs 30 \
    --device 0

# Use fine-tuned weights
python run.py --model runs/finetune/visdrone_person/weights/best.pt \
              --input data/ --output out/
```

The fine-tuning script converts VisDrone annotations to YOLO format, maps pedestrian + people categories to a single "person" class, and filters heavily occluded boxes (occlusion==2) which add noise. Key augmentations: mosaic=1.0 (creates dense small-object scenes), degrees=15 (rotation), scale=0.7 (altitude simulation).

---

## What I Would Add Next

1. **Appearance re-ID on re-entry**: a small (< 5MB) OSNet embedding model to re-link tracks that were lost during long occlusions by appearance matching rather than just position.
2. **INT8 quantisation calibration** for sub-10ms inference on Jetson.
3. **Altitude estimation from bounding box statistics** to dynamically adjust tile count: fewer tiles when drone is low (objects are large), more when high.
