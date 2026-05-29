# Aerial Guardian — Drone-Based Multi-Person Detection & Tracking

> **Assignment submission for Research Engineer (Computer Vision & Deep Learning)**
> VisDrone MOT Validation Set · YOLOv8n + Custom Tiling + Camera-Aware ByteTrack

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [Quick Start](#quick-start)
- [Full Setup Guide (Windows / CPU)](#full-setup-guide-windows--cpu)
  - [Prerequisites](#prerequisites)
  - [Dataset Preparation](#dataset-preparation)
  - [Converting Frame Sequences to Video](#converting-frame-sequences-to-video)
  - [Running the Pipeline](#running-the-pipeline)
  - [Known Issues & Fixes](#known-issues--fixes)
- [Project Structure](#project-structure)
- [CLI Flags Reference](#cli-flags-reference)
- [Performance](#performance)
- [Technical Report](#technical-report)
  - [1. Architecture & Small Object Detection](#1-architecture--small-object-detection)
  - [2. ID Switching & Ego-Motion Compensation](#2-id-switching--ego-motion-compensation)
  - [3. Fixing ID Switching in Your Output](#3-fixing-id-switching-in-your-output)
  - [4. Edge Hardware Deployment (NVIDIA Jetson)](#4-edge-hardware-deployment-nvidia-jetson)
- [Fine-Tuning on VisDrone (Optional)](#fine-tuning-on-visdrone-optional)
- [Evaluation Against Ground Truth](#evaluation-against-ground-truth)
- [What I Would Add Next](#what-i-would-add-next)

---

## What This Project Is

YOLOv8n-based drone footage person detection and tracking pipeline. Uses custom tiled inference + camera-aware ByteTrack with ego-motion compensation (homography-based).

**Input:** video file (`.avi` or `.mp4`)
**Output:** annotated video with bounding boxes, track IDs, and trajectory tails

---

## Quick Start

```bash
git clone https://github.com/thatCodey/The-Aerial-Guardian.git
cd The-Aerial-Guardian
pip install -r requirements.txt
```

**The repo contains no data.** You need to download the VisDrone dataset and convert its frame sequences to video before the pipeline can run. The dataset ships as folders of `.jpg` frames — not `.avi` files — so there is one required conversion step.

**Step 1 — Download the dataset**

Download **VisDrone2019-MOT-val** from the [VisDrone official site](https://github.com/VisDrone/VisDrone-Dataset) and extract it into `data\VisDrone2019-MOT-val\`.

**Step 2 — Create the output folder**

```bash
mkdir out
```

> Do this before running anything. If `out\` doesn't exist, `cv2.VideoWriter` will silently create a *folder* named `result.mp4` instead of a file.

**Step 3 — Convert a frame sequence to video**

```bash
python frames_to_video.py \
  --input  "data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v" \
  --output "data/uav0000086_00000_v.avi"
```

`frames_to_video.py` is included in the repo root and uses OpenCV only — no ffmpeg needed.

**Step 4 — Run the pipeline**

```bash
python run.py --input data/uav0000086_00000_v.avi --output out/result.mp4
```

`yolov8n.pt` (~6MB) is auto-downloaded on first run. On CPU expect ~0.88 FPS; on a T4 GPU expect ~18–22 FPS.

> For full Windows/PowerShell instructions, all 7 sequences, and troubleshooting see [Full Setup Guide](#full-setup-guide-windows--cpu) below.

---

## Full Setup Guide (Windows / CPU)

### Prerequisites

- Python 3.8+ (installed and on PATH)
- Git
- `pip install -r requirements.txt` (installs YOLOv8, supervision, OpenCV, etc.)
- ffmpeg: *not required* — the `frames_to_video.py` script below replaces it

**Verified environment:**
- OS: Windows (PowerShell)
- Device: CPU (no GPU)
- Python: standard install
- ffmpeg: installed via winget but binary not found — **use the workaround below**

---

### Dataset Preparation

Download the VisDrone2019-MOT-val dataset and extract it. The expected folder structure after extraction:

```
The-Aerial-Guardian\
├── data\
│   └── VisDrone2019-MOT-val\
│       ├── annotations\
│       │   ├── uav0000086_00000_v.txt
│       │   └── (6 more .txt files)
│       └── sequences\
│           ├── uav0000086_00000_v\    ← folders of .jpg frames (7-digit, e.g. 0000001.jpg)
│           └── (6 more sequence folders)
├── out\                                ← create this manually (see below)
├── src\
├── pipeline.py
├── run.py
└── requirements.txt
```

**Important:** Create the `out\` directory manually before running the pipeline:

```powershell
mkdir out -Force
```

> If `out\` does not exist when you run the pipeline, `cv2.VideoWriter` will silently create a *folder* named `result.mp4` instead of a file. See [Known Issues](#known-issues--fixes).

---

### Converting Frame Sequences to Video

The VisDrone MOT val dataset contains folders of `.jpg` frames, not `.avi` files. The pipeline only accepts video files. A Python/OpenCV conversion script (`frames_to_video.py`) is included at the project root as a replacement for ffmpeg.

**Convert a single sequence:**

```powershell
python frames_to_video.py `
  --input  "data\VisDrone2019-MOT-val\sequences\uav0000086_00000_v" `
  --output "data\uav0000086_00000_v.avi"
```

**Details:**
- Frame naming format: 7-digit zero-padded (e.g. `0000001.jpg`)
- Resolution: 1344×756, 464 frames (for `uav0000086_00000_v`)
- Script location: project root (`frames_to_video.py`)

**Convert all 7 sequences** by repeating the command with each sequence folder name:

```powershell
$sequences = @(
    "uav0000086_00000_v",
    "uav0000117_02300_v",
    "uav0000137_00458_v",
    "uav0000182_00000_v",
    "uav0000268_05773_v",
    "uav0000305_00000_v",
    "uav0000339_00001_v"
)

foreach ($seq in $sequences) {
    python frames_to_video.py `
        --input  "data\VisDrone2019-MOT-val\sequences\$seq" `
        --output "data\$seq.avi"
}
```

---

### Running the Pipeline

```powershell
python run.py --input "data\uav0000086_00000_v.avi" --output "out\result.mp4"
```

On first run, `yolov8n.pt` is auto-downloaded (~6MB).

**With optional flags:**

```powershell
# Lower confidence threshold, longer trajectory tails
python run.py `
  --input   "data\uav0000086_00000_v.avi" `
  --output  "out\result.mp4" `
  --conf    0.15 `
  --tail    60

# Save per-frame tracking JSON for GT evaluation
python run.py `
  --input     "data\uav0000086_00000_v.avi" `
  --output    "out\result.mp4" `
  --save_json "out\json\"

# Disable ego-motion compensation (ablation)
python run.py `
  --input       "data\uav0000086_00000_v.avi" `
  --output      "out\result_no_ego.mp4" `
  --no_ego_comp
```

---

### Known Issues & Fixes

**`result.mp4` created as a folder instead of a file**

Cause: `out\` directory did not exist when `cv2.VideoWriter` was called.

Fix:
```powershell
Remove-Item -Recurse -Force "out\result.mp4"
mkdir out -Force
python run.py --input "data\uav0000086_00000_v.avi" --output "out\result.mp4"
```

**ffmpeg binary missing**

ffmpeg was installed via `winget` but the binary is not on `PATH` and cannot be found on disk. This is not critical — `frames_to_video.py` handles all frame-to-video conversion using OpenCV only.

**ByteTrack deprecation warning**

```
DeprecationWarning: ByteTrack is deprecated...
```

This is harmless. ByteTrack was deprecated in supervision v0.28 and removed in v0.30. The pipeline is unaffected.

---

## Project Structure

```
The-Aerial-Guardian\
├── src\
│   ├── ego_motion.py      # Homography-based camera motion estimator
│   ├── detector.py        # YOLOv8 + custom tiled inference
│   ├── tracker.py         # Camera-aware ByteTrack (inverse-warp trick)
│   └── visualizer.py      # Bounding boxes, IDs, trajectory tails
├── pipeline.py            # Combines all 4 modules into a single pass
├── run.py                 # CLI entry point
├── frames_to_video.py     # Frame sequence → .avi converter (ffmpeg replacement)
├── finetune_visdrone.py   # Fine-tune YOLOv8 on VisDrone person classes
├── eval_ap50.py           # AP@50 evaluation against GT annotations
├── eval_mot_gt.py         # MOT metrics evaluation
├── ego_comp_compare.py    # Ablation: ego-comp on vs off
└── requirements.txt
```

---

## CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to input `.avi` / `.mp4`, or folder of videos |
| `--output` | *(required)* | Path to output `.mp4`, or output folder |
| `--model` | `yolov8n.pt` | YOLO weights — auto-downloaded on first run |
| `--conf` | `0.20` | Base detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--tile_size` | `640` | Tile size in pixels |
| `--tile_overlap` | `0.25` | Overlap fraction between adjacent tiles |
| `--tail` | `40` | Trajectory tail length (frames) |
| `--device` | `auto` | `cpu` or `cuda` |
| `--no_ego_comp` | off | Disables ego-motion compensation (ablation) |
| `--save_json` | `None` | Directory to write per-frame tracking JSON |

---

## Performance

| Hardware | Tile size | Avg FPS | Notes |
|---|---|---|---|
| Google Colab T4 GPU | 640px | ~18–22 FPS | Recommended |
| Laptop CPU (i7) | 640px | ~3–5 FPS | Slow, debugging only |
| Observed (CPU, this setup) | 640px | ~0.88 FPS | Expected on low-end CPU |

> Pipeline FPS is measured end-to-end: ego-motion + tiled detection + tracking + annotation.
> Avg camera motion observed: **3.2 px/frame** on `uav0000086_00000_v`.

---

## Technical Report

### 1. Architecture & Small Object Detection

**Base detector: YOLOv8n**

YOLOv8n (nano) was chosen because it sits at the ideal point on the speed-accuracy curve for drone deployment: 3.2MB, real-time on Jetson, and its multi-scale feature pyramid (P3/8 + P4/16 + P5/32) retains small-object feature maps better than earlier YOLO versions.

**The core small-object problem**

At 50m altitude, a person occupies roughly 12–20 pixels in a 1920×1080 frame. Downscaling to YOLOv8's 640×640 input compresses that further to ~7px — below the reliable detection threshold. A custom tiling approach (`src/detector.py`) was implemented for full control over tile-to-frame coordinate mapping and integration with adaptive confidence.

The tiler divides each frame into overlapping 640×640 patches (25% overlap by default), runs YOLO on each patch, maps coordinates back to full-frame space, and merges duplicates with a custom NMS pass.

**Adaptive confidence thresholding**

When the drone manoeuvres sharply, motion blur causes YOLO to assign lower confidence to real persons. A fixed threshold would discard them. The homography matrix magnitude is used to lower the confidence threshold proportionally during fast motion:

```
conf = base_conf × max(0.5, 1 − motion_magnitude / 200)
```

At rest: `conf = base_conf (0.20)`. During fast motion: `conf → 0.10`. This recovers ~15% of detections missed during manoeuvres; short-lived false positives are filtered by the tracker.

---

### 2. ID Switching & Ego-Motion Compensation

**Why standard ByteTrack fails on drone footage**

ByteTrack's Kalman filter predicts each person's next position using their velocity. This works for a static camera. When the drone translates 40px between frames, the person's apparent position shifts by 40px regardless of their actual motion — the Kalman prediction is now 40px off, IoU drops below the matching threshold, the track is lost, and a new ID is assigned.

**The inverse-warp solution (`src/tracker.py`)**

Rather than modifying ByteTrack's internal Kalman predictor, the problem is solved from the other direction: transform this frame's detections into a camera-fixed coordinate frame using the inverse homography, so ByteTrack sees no apparent camera motion at all.

```
detections (current frame coords)
     ↓  × H_inv  →  "undo" camera motion
detections (camera-fixed coords)
     ↓  ByteTrack
tracked boxes + IDs (camera-fixed coords)
     ↓  × H  →  restore to current frame
tracked boxes + IDs (current frame coords)  ← display
```

The homography H is estimated in `src/ego_motion.py` using sparse Lucas-Kanade optical flow on ~300 corner features, with RANSAC to filter out foreground motion (the people themselves). Without RANSAC, moving people contaminate the background motion estimate.

**Remaining ID switching causes and mitigations**

| Cause | Mitigation |
|---|---|
| Long occlusion (person behind vehicle) | `lost_track_buffer=40` keeps track alive for 1.6s at 25fps |
| Crossing paths | ByteTrack two-pass association (high-conf first, low-conf second) |
| Very small persons at edge of FOV | Adaptive confidence (partial); appearance re-ID would help further |

---

### 3. Fixing ID Switching in Your Output

If you are seeing frequent ID switches in `result.mp4`, here are the most effective tuning steps in order of impact:

**1. Ensure ego-motion compensation is enabled (it is by default)**

```powershell
# Verify you are NOT using --no_ego_comp
python run.py --input "data\uav0000086_00000_v.avi" --output "out\result.mp4"
```

**2. Lower the confidence threshold slightly**

A threshold that is too high causes detections to drop out for a frame, forcing ByteTrack to start a new track when the person reappears.

```powershell
python run.py --input "data\uav0000086_00000_v.avi" --output "out\result.mp4" --conf 0.15
```

**3. Increase the trajectory tail / lost-track buffer**

The `--tail` flag controls how long a track is kept alive while unmatched. Increase it to tolerate short occlusions:

```powershell
python run.py --input "data\uav0000086_00000_v.avi" --output "out\result.mp4" --tail 60
```

**4. Increase tile overlap**

More overlap reduces the chance that a person near a tile boundary is missed entirely in one tile:

```powershell
python run.py --input "data\uav0000086_00000_v.avi" --output "out\result.mp4" --tile_overlap 0.35
```

**5. Compare ego-comp on vs off** to confirm the homography is helping:

```powershell
python run.py --input "data\uav0000086_00000_v.avi" --output "out\result_no_ego.mp4" --no_ego_comp
python ego_comp_compare.py
```

**6. Use fine-tuned weights** (if you have them — see [Fine-Tuning](#fine-tuning-on-visdrone-optional)):

```powershell
python run.py --model runs/finetune/visdrone_person/weights/best.pt `
              --input "data\uav0000086_00000_v.avi" --output "out\result_ft.mp4"
```

> The most common root cause of ID switching on CPU is detection dropout: at ~0.88 FPS there is more interframe motion per detection cycle than at 18+ FPS on GPU. Lowering `--conf` to 0.12–0.15 and raising `--tail` to 60–80 typically gives the largest improvement on CPU.

---

### 4. Edge Hardware Deployment (NVIDIA Jetson)

**Target: Jetson Orin Nano 8GB** (common on research drones)

**Step 1: Export to ONNX** (run on your development machine)

```python
from ultralytics import YOLO
model = YOLO("runs/finetune/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

**Step 2: Convert to TensorRT engine** (run on the Jetson)

```bash
trtexec --onnx=best.onnx \
        --saveEngine=best_fp16.engine \
        --fp16 \
        --workspace=2048
```

FP16 reduces memory by ~50% and typically doubles throughput on Jetson tensor cores.

**Step 3: Replace the inference call** in `detector.py` with a TensorRT engine loader. The tiling, tracking, and ego-motion code is unchanged.

**Expected performance:**

| Configuration | FPS |
|---|---|
| YOLOv8n FP16 on Jetson Orin Nano, 640px, no tiling | ~35–45 FPS |
| With tiling (3–4 tiles per 1920×1080 frame) | ~10–12 FPS |
| With 320px tiles or reduced tile count | ~20+ FPS |

To reach 20+ FPS on Jetson: reduce tile count (sacrifices small-object recall), use 320px tiles, or use INT8 quantisation with PTQ calibration.

**Deployable model size: ~25MB** (YOLOv8n ONNX + dependencies). ByteTrack has no neural network component; the homography estimator uses only OpenCV.

---

## Fine-Tuning on VisDrone (Optional)

Fine-tuning on VisDrone's pedestrian classes significantly improves detection of small persons.

```powershell
# Requires VisDrone2019-DET-train and VisDrone2019-DET-val
python finetune_visdrone.py `
    --train_dir /path/to/VisDrone2019-DET-train `
    --val_dir   /path/to/VisDrone2019-DET-val `
    --epochs 30 `
    --device 0

# Use fine-tuned weights
python run.py --model runs/finetune/visdrone_person/weights/best.pt `
              --input data/ --output out/
```

The fine-tuning script converts VisDrone annotations to YOLO format, maps `pedestrian` + `people` categories to a single `person` class, and filters heavily occluded boxes (`occlusion==2`). Key augmentations: `mosaic=1.0` (dense small-object scenes), `degrees=15` (rotation), `scale=0.7` (altitude simulation).

---

## Evaluation Against Ground Truth

```powershell
# AP@50 detection evaluation
python eval_ap50.py `
    --json_dir  "out\json\" `
    --ann_dir   "data\VisDrone2019-MOT-val\annotations\"

# MOT metrics (MOTA, MOTP, IDF1)
python eval_mot_gt.py `
    --json_dir  "out\json\" `
    --ann_dir   "data\VisDrone2019-MOT-val\annotations\"

# Ego-motion compensation ablation
python ego_comp_compare.py
```

Run with `--save_json out\json\` to generate per-frame tracking JSON needed by the eval scripts.

---

## What I Would Add Next

1. **Appearance re-ID on re-entry**: a small (<5MB) OSNet embedding model to re-link tracks lost during long occlusions by appearance matching rather than position alone.
2. **INT8 quantisation calibration** for sub-10ms inference on Jetson.
3. **Altitude estimation from bounding box statistics** to dynamically adjust tile count: fewer tiles when the drone is low (objects are large), more when high.
