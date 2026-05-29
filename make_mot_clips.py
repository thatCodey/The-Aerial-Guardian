"""
make_mot_clips.py
Assembles VisDrone2019-MOT-val image sequences into AVI video files.
Each sequence directory (0000001.jpg ... 0000NNN.jpg) -> one .avi clip.
"""
import cv2
import os
import glob
import argparse

SELECTED = {
    "uav0000305_00000_v",
    "uav0000137_00458_v",
    "uav0000086_00000_v",
    "uav0000268_05773_v",
}

def make_clips(seq_root, out_dir, fps=25):
    os.makedirs(out_dir, exist_ok=True)
    seq_dirs = sorted(d for d in os.listdir(seq_root)
                      if os.path.isdir(os.path.join(seq_root, d))
                      and d in SELECTED)
    print(f"Found {len(seq_dirs)} sequences to process.")

    for seq in seq_dirs:
        seq_path = os.path.join(seq_root, seq)
        frames = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
        if not frames:
            print(f"  [SKIP] {seq}: no jpg frames found")
            continue

        # Read first frame to get dimensions
        first = cv2.imread(frames[0])
        if first is None:
            print(f"  [SKIP] {seq}: cannot read first frame")
            continue
        h, w = first.shape[:2]

        out_path = os.path.join(out_dir, f"{seq}.avi")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (w, h),
        )
        for f in frames:
            img = cv2.imread(f)
            if img is not None:
                writer.write(img)
        writer.release()
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  [OK] {seq}: {len(frames)} frames -> {out_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_root", default="VisDrone2019-MOT-val/VisDrone2019-MOT-val/sequences")
    ap.add_argument("--out_dir", default="data_mot")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    make_clips(args.seq_root, args.out_dir, args.fps)
