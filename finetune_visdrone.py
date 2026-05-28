"""
finetune_visdrone.py — Fine-tune YOLOv8 on VisDrone Person Classes
===================================================================

WHY FINE-TUNE? (know this cold for interviews)
    The default YOLOv8 weights are trained on COCO — photos scraped from
    the internet, mostly at eye level, with persons at 1–3m distance.

    VisDrone footage is completely different:
        - Top-down view (birds-eye perspective)
        - People appear as small blobs, not full-body silhouettes
        - Crowded scenes with heavy occlusion
        - Image quality degrades at altitude (atmospheric blur)

    A model that's never seen aerial footage will underperform.
    Fine-tuning on VisDrone's training set teaches the model what
    "person" looks like from 30–100 metres up.

VISDRONE CLASSES (important — don't mix these up):
    VisDrone has 10 object categories. We only care about:
        1 → pedestrian  (individual person, clearly visible)
        2 → people      (group or partially occluded person)

    We map BOTH to class 0 ("person") for our single-class model.

DATASET FORMAT:
    VisDrone annotation format (per line in .txt files):
        <x>, <y>, <w>, <h>, <score>, <category>, <truncation>, <occlusion>
    where x,y,w,h are in pixels and category is 1-indexed.

    YOLO format (what ultralytics expects):
        <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    where everything is normalised to [0,1] relative to image size.

HOW TO USE:
    1. Download VisDrone Task 1 (Image Detection) dataset:
       https://github.com/VisDrone/VisDrone-Dataset
       You need: VisDrone2019-DET-train  and  VisDrone2019-DET-val

    2. Run:
       python finetune_visdrone.py \
           --train_dir /path/to/VisDrone2019-DET-train \
           --val_dir   /path/to/VisDrone2019-DET-val \
           --output    ./runs/finetune

    3. Use the best weights:
       python run.py --model runs/finetune/weights/best.pt --input ...
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml
from PIL import Image
from tqdm import tqdm


# VisDrone categories we keep, mapped to our single class 0
PERSON_CATEGORIES = {1, 2}   # 1=pedestrian, 2=people


# ──────────────────────────────────────────────────────────────────────────
# Dataset conversion: VisDrone → YOLO format
# ──────────────────────────────────────────────────────────────────────────

def convert_annotation(ann_path: Path, img_path: Path, out_label_path: Path):
    """
    Converts one VisDrone annotation file to YOLO format.
    Skips boxes with score=0 (ignored regions) and non-person categories.
    Also skips heavily occluded boxes (occlusion == 2) — they add noise.
    """
    try:
        img = Image.open(img_path)
        img_w, img_h = img.size
    except Exception:
        return 0

    lines_out = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue

            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score    = int(parts[4])
            category = int(parts[5])
            # truncation = int(parts[6])
            occlusion = int(parts[7])

            # Skip ignored regions and non-person classes
            if score == 0 or category not in PERSON_CATEGORIES:
                continue
            # Skip heavily occluded (>75%) — annotation is unreliable
            if occlusion == 2:
                continue
            # Skip zero-area boxes
            if w <= 0 or h <= 0:
                continue

            # Convert to YOLO normalised centre format
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h

            # Clamp to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            wn = max(0.0, min(1.0, wn))
            hn = max(0.0, min(1.0, hn))

            lines_out.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, 'w') as f:
        f.write('\n'.join(lines_out))

    return len(lines_out)


def convert_split(visdrone_dir: str, yolo_dir: Path, split_name: str) -> int:
    """Convert one split (train/val) from VisDrone to YOLO layout."""
    src = Path(visdrone_dir)
    img_src  = src / 'images'
    ann_src  = src / 'annotations'

    img_dst  = yolo_dir / 'images' / split_name
    lbl_dst  = yolo_dir / 'labels' / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images = sorted(img_src.glob('*.jpg')) + sorted(img_src.glob('*.png'))
    total_boxes = 0

    for img_path in tqdm(images, desc=f"Converting {split_name}"):
        ann_path = ann_src / (img_path.stem + '.txt')
        if not ann_path.exists():
            continue

        # Copy image
        shutil.copy(img_path, img_dst / img_path.name)

        # Convert annotation
        out_label = lbl_dst / (img_path.stem + '.txt')
        total_boxes += convert_annotation(ann_path, img_path, out_label)

    print(f"  [{split_name}] {len(images)} images, {total_boxes} person boxes")
    return total_boxes


def build_yaml(yolo_dir: Path, output_yaml: Path):
    """Write the dataset YAML that Ultralytics expects."""
    config = {
        'path':  str(yolo_dir.resolve()),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    1,
        'names': ['person'],
    }
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[YAML] Written → {output_yaml}")


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────

def train(data_yaml: str, output_dir: str, epochs: int = 30, imgsz: int = 640, device: str = "0"):
    """
    Fine-tunes YOLOv8n on the converted VisDrone dataset.

    Key hyperparameter choices:
        - epochs=30: Enough to adapt without overfitting on VisDrone's ~6,000 images
        - imgsz=640: Matches tile size in our detector for consistency
        - mosaic=1.0: YOLO's mosaic augmentation: pastes 4 images into 1.
                      This artificially creates images with many small people — exactly
                      what drone footage looks like. Very important for this dataset.
        - degrees=15: Random rotation up to 15°. Drone cameras rotate; this teaches
                      the model to recognise rotated people.
        - scale=0.7:  Random scale augmentation — simulates different altitudes.
    """
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")   # start from COCO pretrained weights

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=device,
        project=output_dir,
        name="visdrone_person",
        # ── Augmentation settings important for aerial data ───────────
        mosaic=1.0,       # combine 4 images — creates dense small-object scenes
        degrees=15.0,     # rotation — drones aren't always level
        scale=0.7,        # scale jitter — simulates altitude variation
        flipud=0.5,       # vertical flip — drone can approach from any angle
        translate=0.2,    # translation
        # ── Keep small boxes — critical for drone footage ─────────────
        box=7.5,          # box regression loss weight
        cls=0.5,          # classification loss weight (only 1 class, less critical)
    )
    print(f"\n[Train] Best weights saved in: {output_dir}/visdrone_person/weights/best.pt")


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on VisDrone person classes")
    parser.add_argument("--train_dir",  required=True, help="Path to VisDrone2019-DET-train")
    parser.add_argument("--val_dir",    required=True, help="Path to VisDrone2019-DET-val")
    parser.add_argument("--yolo_dir",   default="./data/visdrone_yolo", help="Where to write converted dataset")
    parser.add_argument("--output",     default="./runs/finetune",       help="Training output directory")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--device",     default="0", help="GPU id or 'cpu'")
    args = parser.parse_args()

    yolo_dir  = Path(args.yolo_dir)
    data_yaml = yolo_dir / "visdrone_person.yaml"

    # Step 1: Convert dataset
    print("=" * 60)
    print("Step 1: Converting VisDrone → YOLO format")
    convert_split(args.train_dir, yolo_dir, "train")
    convert_split(args.val_dir,   yolo_dir, "val")
    build_yaml(yolo_dir, data_yaml)

    # Step 2: Train
    print("\n" + "=" * 60)
    print("Step 2: Fine-tuning YOLOv8n")
    train(str(data_yaml), args.output, args.epochs, args.imgsz, args.device)


if __name__ == "__main__":
    main()
