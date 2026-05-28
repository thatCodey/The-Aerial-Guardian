"""
eval_ap50.py — Evaluate AP50 on VisDrone val images Before and After Finetuning
================================================================================

Computes person detection AP50 on VisDrone val set using:
  - Baseline:   yolov8n.pt          (COCO pretrained)
  - Finetuned:  runs/finetune/visdrone_person/weights/best.pt

VisDrone val annotation format (per line):
    x, y, w, h, score, category, truncation, occlusion
where category 1=pedestrian, 2=people (both mapped to class 0 "person").

AP50 Calculation:
    For each image, run the model to get boxes + confidences.
    Match predictions to GT boxes by IoU >= 0.50 (Pascal VOC style).
    Compute Average Precision (area under precision-recall curve).
    Average over all images with at least 1 GT box.

Usage:
    python eval_ap50.py --val_dir data/ --finetuned runs/finetune/visdrone_person/weights/best.pt
    python eval_ap50.py --val_dir data/ --finetuned runs/finetune/visdrone_person/weights/best.pt --device 0
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# VisDrone annotation loading
# ─────────────────────────────────────────────────────────────────────────────

PERSON_CATEGORIES = {1, 2}   # pedestrian, people


def load_gt_boxes(ann_path: Path) -> np.ndarray:
    """
    Load ground-truth bounding boxes from one VisDrone annotation file.
    Returns Nx4 array of [x1, y1, x2, y2] pixel coords, person-class only.
    Skips score=0 (ignored) and heavily occluded boxes.
    """
    boxes = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score    = int(parts[4])
            category = int(parts[5])
            occlusion = int(parts[7])
            if score == 0 or category not in PERSON_CATEGORIES:
                continue
            if occlusion == 2:   # heavily occluded — skip
                continue
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes [x1y1x2y2]. Returns MxN matrix."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Average Precision (Pascal VOC 11-point interpolation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Compute AP using 11-point interpolation (classic VOC method)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = precisions[recalls >= thr]
        ap += prec_at_rec.max() if len(prec_at_rec) > 0 else 0.0
    return ap / 11.0


def evaluate_model(model_path: str, val_dir: str, device: str, conf: float = 0.10,
                   iou_thr: float = 0.50, max_images: int = 500) -> dict:
    """
    Run AP50 evaluation on the VisDrone val set.
    
    Returns dict: {ap50, precision, recall, num_images, num_gt}
    """
    from ultralytics import YOLO
    
    print(f"\n[Eval] Loading {model_path}")
    model = YOLO(model_path)

    val_path = Path(val_dir)
    
    # Try several known VisDrone val layout options
    img_dirs  = [
        val_path / "images",
        val_path / "VisDrone2019-DET-val" / "images",
        val_path,
    ]
    ann_dirs  = [
        val_path / "annotations",
        val_path / "VisDrone2019-DET-val" / "annotations",
        val_path / "annotations",
    ]
    
    img_dir = None
    ann_dir = None
    for i_dir, a_dir in zip(img_dirs, ann_dirs):
        if i_dir.exists() and a_dir.exists():
            imgs = list(i_dir.glob("*.jpg")) + list(i_dir.glob("*.png"))
            if imgs:
                img_dir = i_dir
                ann_dir = a_dir
                break

    if img_dir is None:
        # Last resort: look for any images in val_dir
        imgs = list(val_path.glob("**/*.jpg"))[:max_images]
        if not imgs:
            print("[Eval] ERROR: No images found in val_dir. Cannot evaluate.")
            return {"ap50": 0.0, "precision": 0.0, "recall": 0.0,
                    "num_images": 0, "num_gt": 0, "error": "no_images"}
        img_dir = imgs[0].parent
        ann_dir = img_dir.parent / "annotations"

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    images = images[:max_images]
    print(f"[Eval] {len(images)} images in {img_dir}")

    all_tp, all_fp, all_fn = 0, 0, 0
    all_conf_tp = []   # (confidence, is_tp)
    total_gt    = 0

    for img_path in tqdm(images, desc=f"Evaluating {Path(model_path).stem}"):
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        gt_boxes = load_gt_boxes(ann_path)
        if len(gt_boxes) == 0:
            continue
        total_gt += len(gt_boxes)

        # Run model
        results = model(
            str(img_path),
            conf=conf,
            classes=[0],  # person
            device=device,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            all_fn += len(gt_boxes)
            continue

        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_conf  = results.boxes.conf.cpu().numpy()

        # Match preds to GT by IoU
        iou_mat  = box_iou(pred_boxes, gt_boxes)   # P x G
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # Sort predictions by confidence descending
        sorted_idx = np.argsort(-pred_conf)
        for idx in sorted_idx:
            ious = iou_mat[idx]
            best_gt = np.argmax(ious)
            if ious[best_gt] >= iou_thr and not gt_matched[best_gt]:
                all_conf_tp.append((pred_conf[idx], 1))
                gt_matched[best_gt] = True
            else:
                all_conf_tp.append((pred_conf[idx], 0))

        all_fn += int((~gt_matched).sum())

    if not all_conf_tp:
        return {"ap50": 0.0, "precision": 0.0, "recall": 0.0,
                "num_images": len(images), "num_gt": total_gt}

    # Sort by confidence descending → compute cumulative P/R
    all_conf_tp.sort(key=lambda x: -x[0])
    tps_arr    = np.array([x[1] for x in all_conf_tp])
    cum_tp     = np.cumsum(tps_arr)
    cum_fp     = np.cumsum(1 - tps_arr)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    recalls    = cum_tp / (total_gt + 1e-6)

    ap50 = compute_ap(precisions, recalls)
    final_prec = precisions[-1] if len(precisions) > 0 else 0.0
    final_rec  = recalls[-1]    if len(recalls)    > 0 else 0.0

    return {
        "ap50":       float(ap50),
        "precision":  float(final_prec),
        "recall":     float(final_rec),
        "num_images": len(images),
        "num_gt":     total_gt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate AP50 before/after finetuning")
    p.add_argument("--val_dir",    required=True)
    p.add_argument("--baseline",   default="yolov8n.pt")
    p.add_argument("--finetuned",  default="runs/finetune/visdrone_person/weights/best.pt")
    p.add_argument("--device",     default="0")
    p.add_argument("--conf",       type=float, default=0.10)
    p.add_argument("--max_images", type=int,   default=500)
    p.add_argument("--out_json",   default="ap50_results.json")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f" AP50 Evaluation — Baseline vs Finetuned")
    print(f"{'='*60}")

    baseline_results = evaluate_model(
        args.baseline, args.val_dir, args.device, args.conf, max_images=args.max_images
    )
    
    finetuned_path = Path(args.finetuned)
    if finetuned_path.exists():
        finetuned_results = evaluate_model(
            args.finetuned, args.val_dir, args.device, args.conf, max_images=args.max_images
        )
    else:
        print(f"\n[WARN] Finetuned model not found at {args.finetuned}")
        print("       Run finetune_visdrone.py first. Skipping finetuned eval.")
        finetuned_results = {"ap50": 0.0, "precision": 0.0, "recall": 0.0,
                              "num_images": 0, "num_gt": 0, "error": "not_found"}

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" {'Metric':<20}  {'Baseline':>12}  {'Finetuned':>12}  {'Δ':>8}")
    print(f"{'-'*60}")
    
    for metric in ["ap50", "precision", "recall"]:
        bv = baseline_results.get(metric, 0.0)
        fv = finetuned_results.get(metric, 0.0)
        delta = fv - bv
        print(f" {metric.upper():<20}  {bv:>12.4f}  {fv:>12.4f}  {delta:>+8.4f}")

    print(f"{'-'*60}")
    print(f" Images evaluated: {baseline_results.get('num_images', 0)}")
    print(f" GT boxes used:    {baseline_results.get('num_gt', 0)}")
    print(f"{'='*60}\n")

    results = {
        "baseline":  baseline_results,
        "finetuned": finetuned_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] → {args.out_json}")


if __name__ == "__main__":
    main()
