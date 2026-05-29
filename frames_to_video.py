import cv2
import os
import glob
import argparse

def frames_to_video(frames_dir, output_path, fps=25):
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frames:
        print("No .jpg frames found in", frames_dir)
        return

    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    print(f"Found {len(frames)} frames at {w}x{h}, writing to {output_path} ...")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, f in enumerate(frames):
        frame = cv2.imread(f)
        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames)} frames done")

    out.release()
    print("Done:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing .jpg frames")
    parser.add_argument("--output", required=True, help="Output .avi path")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    frames_to_video(args.input, args.output, args.fps)
