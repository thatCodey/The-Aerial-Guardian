"""
run.py — Entry point for Aerial Guardian pipeline.

USAGE:
    # Single video
    python run.py --input data_mot/uav0000305_00000_v.avi --output out_mot/result.mp4

    # All videos in a folder
    python run.py --input data_mot/ --output out_mot/

    # With fine-tuned weights
    python run.py --model runs/finetune/visdrone_person/weights/best.pt \
                  --input data_mot/ --output out_mot_finetuned/

    # Ego-comp OFF (ablation study, save JSON for GT evaluation)
    python run.py --input data_mot/ --output out_mot_noego/ \
                  --no_ego_comp --save_json out_mot_json_off
"""

import argparse
from pathlib import Path
import torch
from pipeline import AerialGuardianPipeline

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def parse_args():
    p = argparse.ArgumentParser(description="Aerial Guardian -- Drone Person Tracker")
    p.add_argument("--input",        required=True)
    p.add_argument("--output",       required=True)
    p.add_argument("--model",        default="yolov8n.pt")
    p.add_argument("--conf",         type=float, default=0.20)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--tile_size",    type=int,   default=640)
    p.add_argument("--tile_overlap", type=float, default=0.25)
    p.add_argument("--tail",         type=int,   default=40)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_ego_comp",  action="store_true",
                   help="Disable ego-motion compensation (ablation study)")
    p.add_argument("--save_json",    default=None,
                   help="Directory to save per-frame tracking JSON (for GT evaluation)")
    return p.parse_args()


def run_one(args, inp: str, out: str):
    pipe = AerialGuardianPipeline(
        model_path=args.model,
        base_conf=args.conf,
        iou_thr=args.iou,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tail_length=args.tail,
        device=args.device,
        ego_comp=not args.no_ego_comp,
    )
    return pipe.process_video(inp, out, json_out=args.save_json)


def main():
    args = parse_args()
    ego_str = "OFF (ablation)" if args.no_ego_comp else "ON"
    print(f"Device: {args.device}  |  Model: {args.model}")
    print(f"Tile: {args.tile_size}px  Overlap: {args.tile_overlap}  Conf: {args.conf}")
    print(f"Ego-motion compensation: {ego_str}\n")

    inp = Path(args.input)
    out = Path(args.output)

    if inp.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        videos = sorted(f for f in inp.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        if not videos:
            print("No video files found.")
            return
        all_stats = []
        for v in videos:
            print(f"\n{'─'*60}")
            print(f"Processing: {v.name}")
            s = run_one(args, str(v), str(out / (v.stem + "_tracked.mp4")))
            all_stats.append(s)
        mean_fps = sum(s["avg_fps"] for s in all_stats) / len(all_stats)
        print(f"\n{'='*60}")
        print(f"Batch done. Mean pipeline FPS: {mean_fps:.2f}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        run_one(args, str(inp), str(out))


if __name__ == "__main__":
    main()
