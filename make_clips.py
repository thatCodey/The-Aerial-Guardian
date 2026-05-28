"""
make_clips.py — Assemble VisDrone DET val images into short .avi clips
=======================================================================

VisDrone DET val images are named: {sequence}_{frame}_{drone}_{id}.jpg
Images from the same sequence are consecutive frames from one video.
We group them by sequence prefix and write each group as a video.

Usage:
    python make_clips.py --img_dir VisDrone2019-DET-val/images --out_dir data/ --fps 10
"""

import argparse
import cv2
import sys
from pathlib import Path
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="VisDrone2019-DET-val/images")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--fps",     type=float, default=10.0)
    p.add_argument("--max_clips", type=int, default=10, help="Max clips to create")
    args = p.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group images by sequence prefix (first field before '_')
    groups = defaultdict(list)
    for img in sorted(img_dir.glob("*.jpg")):
        # Name format: {seq}_{frame}_{drone}_{id}.jpg
        # seq is the first 7-char prefix e.g. "0000001"
        parts = img.stem.split("_")
        seq = parts[0]
        groups[seq].append(img)

    print(f"Found {len(groups)} sequences, {sum(len(v) for v in groups.values())} images total")

    created = 0
    for seq, imgs in sorted(groups.items()):
        if created >= args.max_clips:
            break

        imgs = sorted(imgs)
        if len(imgs) < 5:
            continue

        # Read first image to get resolution
        first = cv2.imread(str(imgs[0]))
        if first is None:
            continue
        h, w = first.shape[:2]

        out_path = out_dir / f"seq_{seq}.avi"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            args.fps,
            (w, h),
        )

        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                writer.write(frame)

        writer.release()
        print(f"  Created {out_path.name}  ({len(imgs)} frames, {w}x{h})")
        created += 1

    print(f"\nDone. Created {created} video clips in {out_dir}/")


if __name__ == "__main__":
    main()
