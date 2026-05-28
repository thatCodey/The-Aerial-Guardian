"""
pipeline.py — Aerial Guardian: Full Inference Pipeline
=======================================================

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
import time
from collections import deque
from src.ego_motion import EgoMotionCompensator
from src.detector   import TiledPersonDetector
from src.tracker    import CameraAwareByteTrack
from src.visualizer import Visualizer


class AerialGuardianPipeline:
    def __init__(
        self,
        model_path:  str   = "yolov8n.pt",
        base_conf:   float = 0.20,
        iou_thr:     float = 0.45,
        tile_size:   int   = 640,
        tile_overlap: float = 0.25,
        tail_length: int   = 40,
        device:      str   = "cpu",
    ):
        self.ego    = EgoMotionCompensator()
        self.det    = TiledPersonDetector(model_path, base_conf, iou_thr,
                                          tile_size, tile_overlap, device)
        self.trk    = CameraAwareByteTrack(track_activation_threshold=base_conf)
        self.vis    = Visualizer(tail_length=tail_length)
        self._fps_q = deque(maxlen=30)

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

        # 3 ─ Track with camera-motion compensation
        tracked = self.trk.update(detections, H)

        # 4 ─ Annotate
        dt = time.perf_counter() - t0
        self._fps_q.append(dt)
        fps = 1.0 / (sum(self._fps_q) / len(self._fps_q))

        annotated = self.vis.update_and_draw(frame, tracked, H, fps, motion_mag)

        return annotated, tracked, fps, motion_mag

    # ─────────────────────────────────────────────────────────────────────
    # Video file processing
    # ─────────────────────────────────────────────────────────────────────

    def process_video(self, input_path: str, output_path: str) -> dict:
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

        print(f"[Pipeline] {total} frames  |  {W}×{H_res}  |  {src_fps:.1f} src-fps")
        fps_log, motion_log, idx = [], [], 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ann, tracked, fps, motion = self.process_frame(frame)
            writer.write(ann)
            fps_log.append(fps)
            motion_log.append(motion)
            idx += 1

            if idx % 100 == 0:
                print(f"  {idx}/{total}  FPS={fps:.1f}  motion={motion:.1f}px")

        cap.release()
        writer.release()

        stats = {
            "frames":       idx,
            "avg_fps":      sum(fps_log) / len(fps_log) if fps_log else 0,
            "avg_motion_px": sum(motion_log) / len(motion_log) if motion_log else 0,
        }
        print(f"\n[Done] avg pipeline FPS: {stats['avg_fps']:.2f}")
        print(f"[Done] avg camera motion: {stats['avg_motion_px']:.1f} px/frame")
        print(f"[Done] saved → {output_path}")
        return stats

    def reset(self):
        """Reset state between videos."""
        self.ego.prev_gray = None
        self.trk.reset()
        self.vis.reset()
