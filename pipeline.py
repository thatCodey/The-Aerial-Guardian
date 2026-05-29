"""
pipeline.py — Aerial Guardian: Full Inference Pipeline
=======================================================

Optional flags:
  ego_comp  (bool, default True)  — set False to disable camera-motion
                                    compensation (used for ablation study)
  json_out  (str, default None)   — path to a directory; if set, writes
                                    per-frame tracking JSON for GT evaluation

Brings together all 4 components in the right order:

    Frame
      │
      ▼
   EgoMotionCompensator  ──► H (homography), magnitude
      │
      ▼
   TiledPersonDetector(magnitude)  ──► raw detections
      │
      ▼
   CameraAwareByteTrack(detections, H)  ──► tracked detections + IDs
      │
      ▼
   Visualizer  ──► annotated frame
      │
      ▼
   Output video
"""

import cv2
import json
import os
import time
from collections import deque
from src.ego_motion import EgoMotionCompensator
from src.detector   import TiledPersonDetector
from src.tracker    import CameraAwareByteTrack
from src.visualizer import Visualizer


class AerialGuardianPipeline:
    def __init__(
        self,
        model_path:   str   = "yolov8n.pt",
        base_conf:    float = 0.20,
        iou_thr:      float = 0.45,
        tile_size:    int   = 640,
        tile_overlap: float = 0.25,
        tail_length:  int   = 40,
        device:       str   = "cpu",
        ego_comp:     bool  = True,
    ):
        self.ego      = EgoMotionCompensator()
        self.det      = TiledPersonDetector(model_path, base_conf, iou_thr,
                                            tile_size, tile_overlap, device)
        self.trk      = CameraAwareByteTrack(track_activation_threshold=base_conf)
        self.vis      = Visualizer(tail_length=tail_length)
        self._fps_q   = deque(maxlen=30)
        self.ego_comp = ego_comp  # False -> disable inverse-warp trick

    # ─────────────────────────────────────────────────────────────────────
    # Single-frame entry point
    # ─────────────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        t0 = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1 ─ Estimate how much the camera moved
        H, motion_mag = self.ego.update(gray)

        # 2 ─ Detect persons (confidence adapts to motion_mag)
        detections = self.det.detect(frame, motion_magnitude=motion_mag)

        # 3 ─ Track (pass H=None to disable ego-motion compensation)
        H_for_tracker = H if self.ego_comp else None
        tracked = self.trk.update(detections, H_for_tracker)

        # 4 ─ Annotate (always draw with H so tails warp correctly)
        dt = time.perf_counter() - t0
        self._fps_q.append(dt)
        fps = 1.0 / (sum(self._fps_q) / len(self._fps_q))

        annotated = self.vis.update_and_draw(frame, tracked, H, fps, motion_mag)

        return annotated, tracked, fps, motion_mag

    # ─────────────────────────────────────────────────────────────────────
    # Video file processing
    # ─────────────────────────────────────────────────────────────────────

    def process_video(self, input_path: str, output_path: str,
                      json_out: str = None) -> dict:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {input_path}")

        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_res  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps,
            (W, H_res),
        )

        print(f"[Pipeline] {total} frames  |  {W}x{H_res}  |  {src_fps:.1f} src-fps")
        print(f"[Pipeline] ego_comp={self.ego_comp}")
        fps_log, motion_log, idx = [], [], 0
        json_frames = []   # per-frame tracking records

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ann, tracked, fps, motion = self.process_frame(frame)
            writer.write(ann)
            fps_log.append(fps)
            motion_log.append(motion)
            idx += 1

            # Build per-frame JSON record (1-indexed frame to match GT)
            if json_out is not None:
                dets = []
                if tracked is not None and tracked.xyxy is not None:
                    for k, box in enumerate(tracked.xyxy):
                        tid = int(tracked.tracker_id[k]) if tracked.tracker_id is not None else -1
                        dets.append({
                            "x1": int(box[0]), "y1": int(box[1]),
                            "x2": int(box[2]), "y2": int(box[3]),
                            "track_id": tid,
                        })
                json_frames.append({"frame": idx, "detections": dets})

            if idx % 100 == 0:
                print(f"  {idx}/{total}  FPS={fps:.1f}  motion={motion:.1f}px")

        cap.release()
        writer.release()

        # Save JSON if requested
        if json_out is not None:
            os.makedirs(json_out, exist_ok=True)
            stem = os.path.splitext(os.path.basename(input_path))[0]
            jpath = os.path.join(json_out, f"{stem}.json")
            with open(jpath, "w") as jf:
                json.dump(json_frames, jf)
            print(f"[JSON] saved -> {jpath}")

        stats = {
            "frames":        idx,
            "avg_fps":       sum(fps_log) / len(fps_log) if fps_log else 0,
            "avg_motion_px": sum(motion_log) / len(motion_log) if motion_log else 0,
        }
        print(f"\n[Done] avg pipeline FPS: {stats['avg_fps']:.2f}")
        print(f"[Done] avg camera motion: {stats['avg_motion_px']:.1f} px/frame")
        print(f"[Done] saved -> {output_path}")
        return stats

    def reset(self):
        """Reset state between videos."""
        self.ego.prev_gray = None
        self.trk.reset()
        self.vis.reset()
