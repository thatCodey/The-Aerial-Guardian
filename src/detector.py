"""
src/detector.py — Person Detector with Adaptive Tiling
=======================================================

THE SMALL OBJECT PROBLEM (memorise this for interviews):
    YOLOv8 runs at a fixed input resolution — typically 640×640.
    If your source frame is 1920×1080, it gets DOWNSCALED to 640×640.
    A person who was 20 pixels tall in the original frame is now
    ~12 pixels tall — below the threshold where YOLO reliably fires.

    Drone footage makes this worse: people at 50m altitude can be
    just 8–15 pixels tall in the raw frame.

THE TILING FIX:
    Instead of shrinking the whole frame, we cut it into overlapping
    640×640 tiles and run YOLO on each tile separately.
    A person who was 12px in the full frame is now 12 * (1920/640) = 36px
    in their local tile — 3× more detectable.

    After detecting in each tile, we offset the coordinates back to
    the full-frame system and run NMS to merge duplicate detections
    at tile boundaries.

WHY WE WROTE THIS OURSELVES (not using the sahi library):
    The sahi library works, but it's a black box. We wrote this
    ourselves because:
    1. We can tune it exactly to the VisDrone resolution and person sizes
    2. We can explain every line in an interview
    3. We added motion-adaptive confidence — if the drone is moving
       fast, we lower the confidence threshold to catch blurry detections
       that YOLO gives low scores to.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

PERSON_CLASS_COCO = 0   # class index in COCO-trained YOLO


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """
    Pure-numpy Non-Maximum Suppression.
    Returns indices of boxes to keep.

    Why we need this after tiling:
        The same person can appear in 2 overlapping tiles, giving us
        2 detections for 1 person. NMS removes the duplicate by keeping
        the higher-confidence box and removing any box with IoU > iou_thr.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]     # sort highest confidence first

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the top box with all remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        # Keep boxes with low overlap (they're different people)
        order = order[np.where(iou <= iou_thr)[0] + 1]

    return np.array(keep, dtype=int)


class TiledPersonDetector:
    def __init__(
        self,
        model_path:   str   = "yolov8n.pt",
        base_conf:    float = 0.20,
        iou_thr:      float = 0.45,
        tile_size:    int   = 640,
        overlap:      float = 0.25,
        device:       str   = "cpu",
    ):
        """
        Args:
            base_conf:  Default confidence threshold. We lower it dynamically
                        when the camera is moving fast (see detect() method).
            tile_size:  Size of each square tile in pixels. Should match YOLO's
                        training resolution (640 for standard YOLOv8).
            overlap:    Fraction of overlap between adjacent tiles (0.0–0.5).
                        Higher overlap = fewer missed persons at boundaries,
                        but more tiles to process = slower.
        """
        print(f"[Detector] Loading {model_path} on {device} …")
        self.model      = YOLO(model_path)
        self.base_conf  = base_conf
        self.iou_thr    = iou_thr
        self.tile_size  = tile_size
        self.overlap    = overlap
        self.device     = device

    def detect(self, frame: np.ndarray, motion_magnitude: float = 0.0) -> sv.Detections:
        """
        Run tiled detection on a single frame.

        ADAPTIVE CONFIDENCE (the key innovation here):
            When the drone moves fast, YOLO assigns LOWER confidence scores
            to persons because:
            - Motion blur makes features less sharp
            - Rapid perspective change distorts the person's appearance

            If we keep a fixed threshold (say 0.25), we miss real people
            during fast drone movement. So we dynamically lower the threshold
            proportional to motion magnitude.

            Formula:  conf = base_conf * max(0.5, 1 - motion_magnitude / 200)
            At motion=0   → conf = base_conf (e.g. 0.20)
            At motion=100 → conf = base_conf * 0.5 (e.g. 0.10)
            At motion=200+ → conf clamped at base_conf * 0.5

            The 0.5 floor prevents the threshold from going absurdly low.
        """
        # ── Adaptive confidence ───────────────────────────────────────────
        scale = max(0.5, 1.0 - motion_magnitude / 200.0)
        conf  = self.base_conf * scale

        # ── Generate tile coordinates ─────────────────────────────────────
        h, w  = frame.shape[:2]
        tiles = self._get_tiles(h, w)

        all_boxes  = []
        all_scores = []

        for (x1, y1, x2, y2) in tiles:
            tile = frame[y1:y2, x1:x2]

            results = self.model(
                tile,
                conf=conf,
                iou=self.iou_thr,
                classes=[PERSON_CLASS_COCO],
                imgsz=self.tile_size,
                device=self.device,
                verbose=False,
            )[0]

            if results.boxes is None or len(results.boxes) == 0:
                continue

            boxes  = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()

            # Offset tile-local coordinates back to full-frame coordinates
            boxes[:, 0] += x1;  boxes[:, 2] += x1
            boxes[:, 1] += y1;  boxes[:, 3] += y1

            # Clip to frame boundaries (tiles at edges may exceed frame)
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

            all_boxes.append(boxes)
            all_scores.append(scores)

        if not all_boxes:
            return sv.Detections.empty()

        all_boxes  = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)

        # ── Cross-tile NMS ────────────────────────────────────────────────
        keep = _nms(all_boxes, all_scores, iou_thr=self.iou_thr)

        return sv.Detections(
            xyxy=all_boxes[keep],
            confidence=all_scores[keep],
            class_id=np.zeros(len(keep), dtype=int),
        )

    def _get_tiles(self, h: int, w: int) -> list[tuple]:
        """
        Generates a list of (x1, y1, x2, y2) tile rectangles covering the frame.

        Uses a sliding window with `overlap` fraction of overlap between
        adjacent tiles. The last tile in each row/column is snapped to the
        frame edge rather than going out of bounds.
        """
        stride = int(self.tile_size * (1 - self.overlap))
        tiles  = []

        y = 0
        while y < h:
            y2 = min(y + self.tile_size, h)
            y1 = max(0, y2 - self.tile_size)   # snap last tile to frame edge

            x = 0
            while x < w:
                x2 = min(x + self.tile_size, w)
                x1 = max(0, x2 - self.tile_size)

                tiles.append((x1, y1, x2, y2))

                if x2 == w:
                    break
                x += stride

            if y2 == h:
                break
            y += stride

        return tiles
