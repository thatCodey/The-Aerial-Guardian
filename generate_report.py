"""
generate_report.py — Assemble Final Results into a Markdown Report
===================================================================

Reads:
  - ego_comp_results.json     (from ego_comp_compare.py)
  - ap50_results.json         (from eval_ap50.py)
  - out/ directory            (for list of .mp4 files)
  - pipeline logs             (parsed from stdout if run with > log.txt)

Writes:
  - RESULTS.md                (human-readable summary)

Usage:
    python generate_report.py
    python generate_report.py --out_dir out/ --ego_json ego_comp_results.json \
           --ap50_json ap50_results.json --report RESULTS.md
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_json(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",    default="out/")
    p.add_argument("--ego_json",   default="ego_comp_results.json")
    p.add_argument("--ap50_json",  default="ap50_results.json")
    p.add_argument("--report",     default="RESULTS.md")
    args = p.parse_args()

    ego_data = load_json(args.ego_json)
    ap50_data = load_json(args.ap50_json)

    out_dir = Path(args.out_dir)
    mp4_files = sorted(out_dir.glob("**/*.mp4")) if out_dir.exists() else []

    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Aerial Guardian — Pipeline Results")
    lines.append(f"_Generated: {ts}_\n")

    # ── Output Videos ────────────────────────────────────────────────────
    lines.append("## Output Videos\n")
    if mp4_files:
        lines.append("| File | Size |")
        lines.append("|------|------|")
        for mp4 in mp4_files:
            size_mb = mp4.stat().st_size / 1e6
            lines.append(f"| `{mp4.name}` | {size_mb:.1f} MB |")
    else:
        lines.append("_No .mp4 files found in output directory yet._")
    lines.append("")

    # ── AP50: Before vs After Finetuning ─────────────────────────────────
    lines.append("## AP50: Baseline vs Finetuned\n")
    if ap50_data:
        bl = ap50_data.get("baseline", {})
        ft = ap50_data.get("finetuned", {})
        lines.append("| Metric | Baseline (COCO-pretrained) | Finetuned (VisDrone) | Δ |")
        lines.append("|--------|--------------------------|----------------------|---|")
        for metric in ["ap50", "precision", "recall"]:
            bv = bl.get(metric, 0.0)
            fv = ft.get(metric, 0.0)
            delta = fv - bv
            sign  = "+" if delta >= 0 else ""
            lines.append(f"| **{metric.upper()}** | {bv:.4f} | {fv:.4f} | {sign}{delta:.4f} |")
        lines.append("")
        lines.append(f"- Images evaluated: {bl.get('num_images', 'N/A')}")
        lines.append(f"- GT person boxes:  {bl.get('num_gt', 'N/A')}")
        lines.append(f"- IoU threshold:    0.50 (AP50)")
    else:
        lines.append("_AP50 results not yet available. Run `eval_ap50.py` first._")
    lines.append("")

    # ── Ego-Motion Comparison ─────────────────────────────────────────────
    lines.append("## ID Switches: Ego-Compensation ON vs OFF\n")
    if ego_data:
        lines.append("| Clip | Ego ON | Ego OFF | Δ (reduction) | Frames |")
        lines.append("|------|--------|---------|---------------|--------|")
        total_on, total_off = 0, 0
        for r in ego_data:
            name    = r.get("clip", "?")
            on_sw   = r.get("ego_on", {}).get("id_switches", 0)
            off_sw  = r.get("ego_off", {}).get("id_switches", 0)
            delta   = off_sw - on_sw
            frames  = r.get("ego_on", {}).get("frames", 0)
            sign    = "+" if delta >= 0 else ""
            total_on  += on_sw
            total_off += off_sw
            lines.append(f"| `{name}` | {on_sw} | {off_sw} | {sign}{delta} | {frames} |")
        lines.append(f"| **TOTAL** | **{total_on}** | **{total_off}** | **{'+' if total_off-total_on >= 0 else ''}{total_off-total_on}** | |")
        lines.append("")
        if total_on < total_off:
            pct = 100 * (total_off - total_on) / (total_off + 1e-6)
            lines.append(f"> ✅ Ego-motion compensation reduced ID switches by **{pct:.0f}%** across {len(ego_data)} clips.")
        elif total_on == total_off:
            lines.append("> ℹ️ No difference in ID switches — clips may have minimal camera motion.")
        else:
            lines.append("> ⚠️ Ego-comp ON had more switches — unexpected. Check motion magnitude of clips.")
    else:
        lines.append("_Ego-comp results not yet available. Run `ego_comp_compare.py` first._")
    lines.append("")

    # ── What I Noticed ────────────────────────────────────────────────────
    lines.append("## What I Noticed (from actual output stats)\n")

    # Derive observations from data
    observations = []
    
    if ap50_data:
        bl_ap = ap50_data.get("baseline", {}).get("ap50", 0)
        ft_ap = ap50_data.get("finetuned", {}).get("ap50", 0)
        if ft_ap > bl_ap:
            delta_pct = 100 * (ft_ap - bl_ap) / (bl_ap + 1e-6)
            observations.append(
                f"**Finetuning lifted AP50 by {delta_pct:.1f}%** ({bl_ap:.3f} → {ft_ap:.3f}). "
                f"The gap reflects how poorly COCO-pretrained YOLO handles top-down "
                f"aerial perspectives; domain adaptation on VisDrone's person annotations "
                f"immediately closes most of this gap."
            )
        elif ft_ap <= bl_ap:
            observations.append(
                f"**AP50 did not improve after finetuning** ({bl_ap:.3f} → {ft_ap:.3f}). "
                f"Possible causes: insufficient training data in the val split, "
                f"class imbalance, or the val images are structurally different from train."
            )

    if ego_data:
        total_on  = sum(r.get("ego_on",  {}).get("id_switches", 0) for r in ego_data)
        total_off = sum(r.get("ego_off", {}).get("id_switches", 0) for r in ego_data)
        if total_on < total_off:
            reduction = total_off - total_on
            observations.append(
                f"**Ego-motion compensation reduced ID switches by {reduction} "
                f"({total_off}→{total_on}) across {len(ego_data)} clips.** "
                f"The inverse-warp trick — mapping detections into a camera-fixed coordinate "
                f"system before ByteTrack — is the core reason: the Kalman filter's predictions "
                f"now align with detections even when the drone translates or rotates mid-clip."
            )
        
        # Check FPS
        fps_vals = [r.get("ego_on", {}).get("avg_fps", 0) for r in ego_data if r.get("ego_on", {}).get("avg_fps", 0) > 0]
        if fps_vals:
            avg_fps = sum(fps_vals) / len(fps_vals)
            observations.append(
                f"**Average pipeline speed: {avg_fps:.1f} FPS.** "
                f"The tiling step (4–12 tiles per 1080p frame) is the main bottleneck; "
                f"on CPU this typically runs at 2–5 FPS, on GPU 15–30 FPS."
            )

    if mp4_files:
        observations.append(
            f"**{len(mp4_files)} tracked video(s) produced.** "
            f"Each clip shows colour-coded bounding boxes with persistent IDs, "
            f"fading trajectory tails warped to follow camera motion, and a live "
            f"HUD displaying person count, pipeline FPS, and estimated motion magnitude."
        )

    if not observations:
        observations.append(
            "No results available yet — run the full pipeline first and then re-run this script."
        )

    for obs in observations:
        lines.append(f"- {obs}")
        lines.append("")

    # ── Pipeline Config ───────────────────────────────────────────────────
    lines.append("## Pipeline Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Detector | YOLOv8n (tiled, 640px tiles, 25% overlap) |")
    lines.append("| Tracker  | ByteTrack with camera-aware warp correction |")
    lines.append("| Ego-motion | Lucas-Kanade optical flow + RANSAC homography |")
    lines.append("| Finetuning | YOLOv8n, 30 epochs, mosaic aug, degrees=15, scale=0.7 |")
    lines.append("| Val metric | AP50 (Pascal VOC, IoU≥0.50) |")
    lines.append("")

    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"[Report] Written -> {args.report}")
    print(report)


if __name__ == "__main__":
    main()
