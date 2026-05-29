"""
eval_mot_gt.py
Ground-truth-based evaluator for the Aerial Guardian pipeline.

Reads VisDrone MOT annotations and pipeline output to compute:
  - Per-sequence ID switches (ego-comp ON vs OFF)
  - Per-frame detection counts (GT vs predicted)
  - Summary table

VisDrone MOT annotation format (one line per object per frame):
  frame_id, object_id, x, y, w, h, score, category, truncation, occlusion

Person categories: 1=pedestrian, 2=people
score=0 -> ignored region, skip
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict


PERSON_CATS = {1, 2}  # pedestrian, people


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loader
# ─────────────────────────────────────────────────────────────────────────────

def load_gt(ann_path):
    """
    Returns dict: frame_id (1-indexed) -> list of [x1, y1, x2, y2, track_id]
    Only valid (score=1) person (cat 1,2) annotations kept.
    """
    gt = defaultdict(list)
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            obj_id   = int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            score    = int(parts[6])
            category = int(parts[7])
            if score == 0:
                continue
            if category not in PERSON_CATS:
                continue
            gt[frame_id].append([x, y, x + w, y + h, obj_id])
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_thr=0.5):
    """
    Returns list of (gt_idx, pred_idx) matched pairs using greedy IoU matching.
    """
    if not gt_boxes or not pred_boxes:
        return []
    ious = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            ious[i, j] = iou(g, p)
    matched = []
    used_pred = set()
    # Greedy: sort by IoU descending
    pairs = sorted(
        [(ious[i, j], i, j) for i in range(len(gt_boxes)) for j in range(len(pred_boxes))],
        reverse=True,
    )
    used_gt = set()
    for score, gi, pi in pairs:
        if score < iou_thr:
            break
        if gi in used_gt or pi in used_pred:
            continue
        matched.append((gi, pi))
        used_gt.add(gi)
        used_pred.add(pi)
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# ID switch counter
# ─────────────────────────────────────────────────────────────────────────────

def count_id_switches(gt, pred_per_frame, iou_thr=0.5):
    """
    gt: dict frame_id -> list [x1,y1,x2,y2, gt_track_id]
    pred_per_frame: dict frame_id -> list [x1,y1,x2,y2, pred_track_id]

    For each GT track, record what predicted ID it was matched to last frame.
    If it changes -> ID switch.
    """
    gt_to_pred_last = {}   # gt_track_id -> last matched pred_track_id
    switches = 0

    all_frames = sorted(set(gt.keys()) | set(pred_per_frame.keys()))
    for frame_id in all_frames:
        gt_boxes = gt.get(frame_id, [])
        pred_boxes = pred_per_frame.get(frame_id, [])

        if not gt_boxes or not pred_boxes:
            continue

        matches = match_boxes(gt_boxes, pred_boxes, iou_thr)

        for gi, pi in matches:
            gt_id   = gt_boxes[gi][4]
            pred_id = pred_boxes[pi][4]

            if gt_id in gt_to_pred_last:
                if gt_to_pred_last[gt_id] != pred_id:
                    switches += 1
            gt_to_pred_last[gt_id] = pred_id

    return switches


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline output reader
# ─────────────────────────────────────────────────────────────────────────────

def read_pipeline_json(json_path):
    """
    Reads the per-frame tracking JSON saved by run.py (--save_json flag).
    Falls back to empty if file not found.
    Format: list of frames, each frame = list of {x1,y1,x2,y2,track_id}
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        data = json.load(f)
    result = {}
    for frame in data:
        fid  = frame["frame"]
        dets = []
        for d in frame.get("detections", []):
            dets.append([d["x1"], d["y1"], d["x2"], d["y2"], d["track_id"]])
        result[fid] = dets
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir",    default="VisDrone2019-MOT-val/VisDrone2019-MOT-val/annotations")
    ap.add_argument("--on_dir",     default="out_mot_json",      help="JSON output dir, ego-comp ON")
    ap.add_argument("--off_dir",    default="out_mot_json_off",  help="JSON output dir, ego-comp OFF")
    ap.add_argument("--out",        default="mot_eval_results.json")
    ap.add_argument("--iou_thr",    type=float, default=0.5)
    args = ap.parse_args()

    SEQUENCES = [
        "uav0000305_00000_v",
        "uav0000137_00458_v",
        "uav0000086_00000_v",
        "uav0000268_05773_v",
    ]

    results = {}
    print(f"\n{'Sequence':<30} {'GT tracks':>10} {'IDs ON':>8} {'IDs OFF':>8} {'Delta':>8}")
    print("-" * 68)

    for seq in SEQUENCES:
        ann_path  = os.path.join(args.ann_dir, f"{seq}.txt")
        on_path   = os.path.join(args.on_dir,  f"{seq}.json")
        off_path  = os.path.join(args.off_dir, f"{seq}.json")

        if not os.path.exists(ann_path):
            print(f"  [SKIP] {seq}: annotation not found")
            continue

        gt = load_gt(ann_path)
        gt_track_count = len(set(v[4] for boxes in gt.values() for v in boxes))

        on_pred  = read_pipeline_json(on_path)
        off_pred = read_pipeline_json(off_path)

        sw_on  = count_id_switches(gt, on_pred,  args.iou_thr) if on_pred  else "N/A"
        sw_off = count_id_switches(gt, off_pred, args.iou_thr) if off_pred else "N/A"

        delta = ""
        if isinstance(sw_on, int) and isinstance(sw_off, int):
            delta = f"{sw_off - sw_on:+d}"

        results[seq] = {
            "gt_tracks": gt_track_count,
            "id_switches_on":  sw_on,
            "id_switches_off": sw_off,
        }
        print(f"  {seq:<28} {gt_track_count:>10} {str(sw_on):>8} {str(sw_off):>8} {delta:>8}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
