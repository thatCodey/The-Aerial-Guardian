"""
src/tracker.py — Camera-Aware ByteTrack
========================================

THE TRACKING PROBLEM ON DRONE FOOTAGE:

    ByteTrack uses a Kalman filter internally.
    The Kalman filter says: "Track #5 was at pixel (300, 400) last frame,
    moving at (+5, +2) px/frame, so it'll be at (305, 402) this frame."

    This works great for a STATIC camera. But our drone moved.
    If the drone translated 40px to the right, the person who was at (300,400)
    now appears at (260, 400) — even if they didn't move at all.
    The Kalman predicts (305,402), the detection is at (260,400),
    IoU between prediction and detection = ~0. ByteTrack loses the track.

THE INVERSE-WARP TRICK (the key insight of this file):

    We can't easily modify ByteTrack's internal Kalman predictor.
    But we can solve the same problem from the OTHER direction:

    Instead of making ByteTrack's PREDICTIONS match the current frame,
    we make the DETECTIONS match the coordinate space that ByteTrack expects.

    ByteTrack's predictions live in a "camera-fixed" coordinate system
    (where the background doesn't move). If we apply the INVERSE homography
    to this frame's detections, we "undo" the camera's motion — putting
    the detections back into the same coordinate space as the predictions.

    After ByteTrack assigns IDs, we warp the bounding boxes BACK to the
    actual current-frame coordinates for display.

    Visually:
        curr frame detections
                ↓  apply H_inv  (undo camera motion)
        camera-fixed detections
                ↓  ByteTrack (predictions now match!)
        tracked IDs in camera-fixed coords
                ↓  apply H  (restore to actual frame)
        tracked IDs in curr frame coords  ← what we display

    This is mathematically equivalent to correcting the Kalman predictor,
    but achieved without touching ByteTrack's internals.
"""

import cv2
import numpy as np
import supervision as sv


def _warp_detections(detections: sv.Detections, H: np.ndarray) -> sv.Detections:
    """
    Warps detection bounding boxes through homography H.
    Warps all 4 corners (not just centre) to handle perspective distortion.
    """
    if len(detections) == 0 or H is None:
        return detections

    warped_boxes = []
    for x1, y1, x2, y2 in detections.xyxy:
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        ).reshape(-1, 1, 2)
        w_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        warped_boxes.append([
            w_corners[:, 0].min(), w_corners[:, 1].min(),
            w_corners[:, 0].max(), w_corners[:, 1].max(),
        ])

    return sv.Detections(
        xyxy=np.array(warped_boxes, dtype=np.float32),
        confidence=detections.confidence,
        class_id=detections.class_id,
    )


class CameraAwareByteTrack:
    """
    Wraps supervision's ByteTrack to handle ego-motion using the
    inverse-warp trick described in this file's docstring.

    Usage:
        tracker = CameraAwareByteTrack()
        tracked = tracker.update(detections, H)  # H from EgoMotionCompensator
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.20,
        lost_track_buffer:          int   = 40,
        minimum_matching_threshold: float = 0.8,
        frame_rate:                 int   = 25,
    ):
        """
        Args:
            track_activation_threshold: Min confidence to START a new track.
                                        ByteTrack internally uses TWO thresholds:
                                        high-conf detections match to existing tracks first,
                                        then low-conf detections get a second pass.
                                        This sets the floor.
            lost_track_buffer:  How many frames to keep a track "alive" after its
                                person disappears (e.g. behind an obstacle). 40 frames
                                at 25fps = 1.6 seconds before the ID is released.
            minimum_matching_threshold: Min IoU for a detection to be matched to
                                        an existing track. Lower = more permissive.
        """
        self._byte_track = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(
        self,
        detections: sv.Detections,
        H: np.ndarray | None = None,
    ) -> sv.Detections:
        """
        Update the tracker with new detections, compensating for camera motion.

        Args:
            detections: sv.Detections from this frame (full-frame coordinates).
            H:          Homography from EgoMotionCompensator (prev→curr frame).
                        Pass None if unavailable (first frame, estimation failed).

        Returns:
            sv.Detections with .tracker_id field populated.
            Bounding boxes are in current-frame coordinates (correct for display).
        """
        if len(detections) == 0:
            # Still need to tick the tracker to age out lost tracks
            return self._byte_track.update_with_detections(sv.Detections.empty())

        if H is not None:
            # ── Step 1: Compute inverse homography ────────────────────────
            # H maps prev_frame → curr_frame
            # H_inv maps curr_frame → prev_frame (camera-fixed space)
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = None
        else:
            H_inv = None

        # ── Step 2: Warp detections into camera-fixed coordinates ─────────
        dets_for_tracker = _warp_detections(detections, H_inv)

        # ── Step 3: Run ByteTrack in camera-fixed space ───────────────────
        tracked_warped = self._byte_track.update_with_detections(dets_for_tracker)

        if len(tracked_warped) == 0 or tracked_warped.tracker_id is None:
            return tracked_warped

        # ── Step 4: Warp tracked boxes BACK to current-frame coordinates ──
        # The tracker_id assignments are still correct; we just fix the coordinates.
        tracked_curr = _warp_detections(tracked_warped, H)

        # Re-attach tracker IDs (warp creates a new Detections object)
        tracked_curr.tracker_id = tracked_warped.tracker_id
        return tracked_curr

    def reset(self):
        """Reset tracker state (call between videos)."""
        self._byte_track.reset()
