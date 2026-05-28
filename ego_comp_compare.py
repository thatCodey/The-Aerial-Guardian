"""
ego_comp_compare.py — Compare ID Switches: Ego-Motion Compensation ON vs OFF
=============================================================================

Runs 2-3 video clips through the pipeline twice:
  1. Ego-comp ON  (default — H is computed and passed to tracker)
  2. Ego-comp OFF (H is always passed as None, so tracker sees raw detections)

Counts ID switches in each case and prints a comparison table.

ID Switch Definition:
    A new tracker_id appears for an object that was previously assigned
    a different ID — i.e., the tracker "lost" someone and reacquired them
    with a new number. We detect this by tracking bounding-box centroid
    assignments frame-over-frame and counting re-IDs.

Usage:
    python ego_comp_compare.py --input data/ --clips 3 --device 0
    python ego_comp_compare.py --input data/video.avi --device cpu
"""

import argparse
import cv2
import time
import json
import sys
import os
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import supervision as sv

# ── ensure src is on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.ego_motion import EgoMotionCompensator
from src.detector   import TiledPersonDetector
from src.tracker    import CameraAwareByteTrack
from src.visualizer import Visualizer


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


# ─────────────────────────────────────────────────────────────────────────────
# Core processing function (ego-comp toggled via flag)
# ─────────────────────────────────────────────────────────────────────────────

def process_clip(
    video_path: str,
    model_path: str = "yolov8n.pt",
    device: str = "cpu",
    ego_comp: bool = True,
    max_frames: int = 500,
    conf: float = 0.20,
    iou: float = 0.45,
) -> dict:
    """
    Process a video clip and return ID switch statistics.
    
    Args:
        ego_comp: If True, pass the estimated H to the tracker.
                  If False, always pass H=None (disable compensation).
    
    Returns dict with:
        id_switches:  number of times a new tracker_id appeared for a bbox
                      that had a nearby existing ID in the previous frame.
        total_ids:    total unique tracker IDs assigned over the clip.
        frames:       frames processed.
        avg_fps:      pipeline speed.
    """
    ego    = EgoMotionCompensator()
    det    = TiledPersonDetector(model_path, conf, iou, 640, 0.25, device)
    trk    = CameraAwareByteTrack(track_activation_threshold=conf)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps_q        = deque(maxlen=30)
    frame_count  = 0

    # Track ID continuity to count switches
    # prev_assignments: {track_id -> centroid (cx, cy)} from last frame
    prev_assignments: dict = {}
    all_ids_seen:     set  = set()
    id_switches:      int  = 0

    print(f"  ego_comp={'ON ' if ego_comp else 'OFF'}  clip={Path(video_path).name}")

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        H, motion_mag = ego.update(gray)
        detections    = det.detect(frame, motion_magnitude=motion_mag)

        # Ego-comp toggle: pass H only if enabled
        tracked = trk.update(detections, H if ego_comp else None)

        dt = time.perf_counter() - t0
        fps_q.append(dt)
        frame_count += 1

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            prev_assignments = {}
            continue

        # ── ID switch counting ────────────────────────────────────────
        curr_assignments = {}
        for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            curr_assignments[int(tid)] = (cx, cy)

        # A switch happens when a new ID appears but there was a very close
        # previous ID that disappeared (another ID "took over" a position).
        # Simpler heuristic: count IDs that are new this frame AND there was
        # a nearby ID (<50px) in the previous frame under a different number.
        if prev_assignments:
            prev_centroids = list(prev_assignments.values())
            prev_ids_arr   = list(prev_assignments.keys())
            for tid, (cx, cy) in curr_assignments.items():
                if tid not in all_ids_seen:
                    # New ID appeared — check if it's close to a now-gone ID
                    for p_id, (px, py) in prev_assignments.items():
                        if p_id not in curr_assignments:
                            dist = np.sqrt((cx-px)**2 + (cy-py)**2)
                            if dist < 60:
                                id_switches += 1
                                break

        all_ids_seen.update(curr_assignments.keys())
        prev_assignments = curr_assignments

    cap.release()

    avg_fps = (1.0 / (sum(fps_q) / len(fps_q))) if fps_q else 0.0
    total_ids = len(all_ids_seen)

    return {
        "id_switches": id_switches,
        "total_ids":   total_ids,
        "frames":      frame_count,
        "avg_fps":     avg_fps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ego-motion compensation comparison")
    p.add_argument("--input",      required=True,  help="Video file or folder")
    p.add_argument("--model",      default="yolov8n.pt")
    p.add_argument("--clips",      type=int, default=3,   help="Max clips to test from folder")
    p.add_argument("--max_frames", type=int, default=500, help="Max frames per clip")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--conf",       type=float, default=0.20)
    p.add_argument("--out_json",   default="ego_comp_results.json")
    args = p.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        videos = sorted(f for f in inp.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        videos = videos[:args.clips]
    else:
        videos = [inp]

    if not videos:
        print("No video files found.")
        return

    print(f"\n{'='*70}")
    print(f" Ego-Motion Compensation Comparison  ({len(videos)} clips)")
    print(f"{'='*70}")
    print(f" Model : {args.model}")
    print(f" Device: {args.device}")
    print(f" Max frames/clip: {args.max_frames}")
    print(f"{'='*70}\n")

    results = []

    for v in videos:
        print(f"\n── Clip: {v.name} ──────────────────────────────────────")
        stats_on  = process_clip(str(v), args.model, args.device, ego_comp=True,
                                  max_frames=args.max_frames, conf=args.conf)
        stats_off = process_clip(str(v), args.model, args.device, ego_comp=False,
                                  max_frames=args.max_frames, conf=args.conf)
        results.append({
            "clip":       v.name,
            "ego_on":     stats_on,
            "ego_off":    stats_off,
        })

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(" RESULTS: ID Switches — Ego-Comp ON vs OFF")
    print(f"{'='*70}")
    header = f"{'Clip':<35}  {'Ego ON':>8}  {'Ego OFF':>8}  {'Delta':>8}  {'Frames':>7}"
    print(header)
    print("-"*70)

    total_on, total_off = 0, 0
    for r in results:
        name    = r["clip"][:34]
        on_sw   = r["ego_on"]["id_switches"]
        off_sw  = r["ego_off"]["id_switches"]
        delta   = off_sw - on_sw
        frames  = r["ego_on"]["frames"]
        total_on  += on_sw
        total_off += off_sw
        print(f"{name:<35}  {on_sw:>8}  {off_sw:>8}  {delta:>+8}  {frames:>7}")

    print("-"*70)
    print(f"{'TOTAL':<35}  {total_on:>8}  {total_off:>8}  {total_off-total_on:>+8}")
    print(f"{'='*70}\n")
    print("Positive delta = ego-comp reduced ID switches (good).")

    # Save JSON for report generation
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Results → {args.out_json}")


if __name__ == "__main__":
    main()
