"""
src/visualizer.py — Annotation & Display Utilities
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque


PALETTE = [
    (50,  205, 50),   # lime green
    (30,  144, 255),  # dodger blue
    (255, 140, 0),    # dark orange
    (186, 85,  211),  # medium orchid
    (255, 215, 0),    # gold
    (0,   206, 209),  # dark turquoise
    (220, 20,  60),   # crimson
    (127, 255, 0),    # chartreuse
]


def _color(track_id: int) -> tuple:
    return PALETTE[int(track_id) % len(PALETTE)]


class TrajectoryManager:
    """
    Maintains rolling trajectory tails for each track.
    Handles ego-motion warping: when the camera moves, old trajectory
    points are warped into the new frame's coordinate space so the
    tail stays glued to the person, not floating in space.
    """

    def __init__(self, tail_length: int = 40):
        self.tail_length = tail_length
        self.tails: dict = defaultdict(lambda: deque(maxlen=tail_length))

    def warp_existing_tails(self, H: np.ndarray | None):
        """
        When the camera moves (H is not None), transform all existing tail
        points into the new frame's coordinate system.

        Without this, tail lines drift away from the person because the
        person moved in the frame (camera motion) but the old tail points
        didn't.
        """
        if H is None:
            return
        from src.ego_motion import EgoMotionCompensator
        for tid in list(self.tails.keys()):
            pts = list(self.tails[tid])
            warped = EgoMotionCompensator.warp_points(pts, H)
            self.tails[tid] = deque(warped, maxlen=self.tail_length)

    def update(self, tracked: sv.Detections):
        """Add the current centre-point of each tracked person to their tail."""
        if tracked.tracker_id is None:
            return
        for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            self.tails[tid].append((cx, cy))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for tid, pts in self.tails.items():
            pts = list(pts)
            if len(pts) < 2:
                continue
            color = _color(tid)
            for j in range(1, len(pts)):
                alpha     = j / len(pts)
                fade      = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, pts[j - 1], pts[j], fade, thickness, cv2.LINE_AA)
        return frame

    def reset(self):
        self.tails.clear()


class Visualizer:
    def __init__(self, tail_length: int = 40):
        self.traj = TrajectoryManager(tail_length=tail_length)

    def update_and_draw(
        self,
        frame:    np.ndarray,
        tracked:  sv.Detections,
        H:        np.ndarray | None = None,
        fps:      float = 0.0,
        motion:   float = 0.0,
    ) -> np.ndarray:
        """
        Full annotation pass: warp old tails, update tails, draw everything.
        Returns an annotated copy of frame.
        """
        out = frame.copy()

        # 1. Warp old tail points to current frame
        self.traj.warp_existing_tails(H)

        # 2. Append new centre-points
        self.traj.update(tracked)

        # 3. Draw tails
        self.traj.draw(out)

        # 4. Draw bounding boxes + ID labels
        if tracked.tracker_id is not None:
            for i, (bbox, tid) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                x1, y1, x2, y2 = map(int, bbox)
                color = _color(tid)
                conf  = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

                # Box
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                # Label background + text
                label = f"#{tid}  {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, cv2.FILLED)
                cv2.putText(out, label, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 5. HUD
        count = len(tracked) if tracked.tracker_id is not None else 0
        hud_lines = [
            f"Persons: {count}",
            f"FPS: {fps:.1f}",
            f"Motion: {motion:.1f}px",
        ]
        for k, line in enumerate(hud_lines):
            cv2.putText(out, line, (10, 35 + k * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return out

    def reset(self):
        self.traj.reset()
