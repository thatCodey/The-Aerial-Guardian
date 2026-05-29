"""
run_all_experiments.py
Runs all Aerial Guardian experiments in sequence on MOT val clips:
  1. Baseline (ego-comp ON)  → out_mot/          + JSON
  2. Ego-comp OFF (ablation) → out_mot_noego/     + JSON
  3. Finetuned model         → out_mot_finetuned/ + JSON
  4. GT evaluation           → mot_eval_results.json
  5. Print summary table
"""

import subprocess
import sys
import os
import json
from pathlib import Path

PYTHON = sys.executable
BASE_MODEL      = "yolov8n.pt"
FINETUNED_MODEL = "runs/finetune/visdrone_person/weights/best.pt"
INPUT_DIR       = "data_mot"
DEVICE          = "cpu"

# Sequences to use for ego-comp ablation (2 shortest = faster)
EGO_COMP_SEQS = [
    "uav0000305_00000_v",
    "uav0000137_00458_v",
]

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"CMD:  {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"[WARNING] Step '{label}' exited with code {result.returncode}")
    return result.returncode

def main():
    # ── Step 1: Baseline ego-comp ON, all 4 clips ──────────────────────────
    run([PYTHON, "run.py",
         "--input", INPUT_DIR,
         "--output", "out_mot",
         "--model", BASE_MODEL,
         "--save_json", "out_mot_json",
         "--device", DEVICE],
        "Baseline (ego-comp ON) — all 4 clips")

    # ── Step 2: Ego-comp OFF ablation, 2 short clips ───────────────────────
    # Run only the 2 shortest sequences for ego-comp comparison
    os.makedirs("out_mot_noego", exist_ok=True)
    os.makedirs("out_mot_json_off", exist_ok=True)
    for seq in EGO_COMP_SEQS:
        src = str(Path(INPUT_DIR) / f"{seq}.avi")
        dst = str(Path("out_mot_noego") / f"{seq}_tracked.mp4")
        if not os.path.exists(src):
            print(f"[SKIP] {src} not found")
            continue
        run([PYTHON, "run.py",
             "--input", src,
             "--output", dst,
             "--model", BASE_MODEL,
             "--no_ego_comp",
             "--save_json", "out_mot_json_off",
             "--device", DEVICE],
            f"Ego-comp OFF — {seq}")

    # ── Step 3: Finetuned model, all 4 clips ──────────────────────────────
    ft_model = FINETUNED_MODEL
    if not os.path.exists(ft_model):
        # Try alternate path
        alts = list(Path("runs").rglob("best.pt"))
        if alts:
            ft_model = str(alts[0])
            print(f"[INFO] Using finetuned weights: {ft_model}")
        else:
            print("[WARNING] Finetuned weights not found — skipping Step 3")
            ft_model = None

    if ft_model:
        run([PYTHON, "run.py",
             "--input", INPUT_DIR,
             "--output", "out_mot_finetuned",
             "--model", ft_model,
             "--save_json", "out_mot_json_finetuned",
             "--device", DEVICE],
            "Finetuned model — all 4 clips")

    # ── Step 4: GT evaluation ─────────────────────────────────────────────
    run([PYTHON, "eval_mot_gt.py",
         "--ann_dir", "VisDrone2019-MOT-val/VisDrone2019-MOT-val/annotations",
         "--on_dir",  "out_mot_json",
         "--off_dir", "out_mot_json_off",
         "--out",     "mot_eval_results.json"],
        "GT ID-switch evaluation")

    # ── Step 5: Print summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    # FPS from baseline JSON logs
    for out_dir, label in [("out_mot", "Baseline"), ("out_mot_finetuned", "Finetuned")]:
        mp4s = list(Path(out_dir).glob("*.mp4")) if Path(out_dir).exists() else []
        print(f"{label}: {len(mp4s)} output videos in {out_dir}/")

    if os.path.exists("mot_eval_results.json"):
        with open("mot_eval_results.json") as f:
            res = json.load(f)
        print("\nID Switch Comparison (ego-comp ON vs OFF):")
        print(f"{'Sequence':<32} {'GT tracks':>10} {'Switches ON':>12} {'Switches OFF':>13} {'Reduction':>10}")
        print("-" * 80)
        for seq, r in res.items():
            sw_on  = r.get("id_switches_on",  "N/A")
            sw_off = r.get("id_switches_off", "N/A")
            gt_t   = r.get("gt_tracks", "?")
            if isinstance(sw_on, int) and isinstance(sw_off, int) and sw_off > 0:
                pct = f"{(sw_off - sw_on) / sw_off * 100:.1f}%"
            else:
                pct = "N/A"
            print(f"  {seq:<30} {gt_t:>10} {str(sw_on):>12} {str(sw_off):>13} {pct:>10}")

    print("\nAll experiments complete. Ready to update RESULTS.md and push.")

if __name__ == "__main__":
    main()
