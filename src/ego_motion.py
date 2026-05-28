"""
src/ego_motion.py — Camera Motion Estimator
============================================

CORE IDEA (explain this in your interview):
    A drone video has TWO kinds of motion mixed together:
        1. Objects moving (people walking)
        2. The camera itself moving (drone flying)

    Standard trackers like ByteTrack assume only (1) is happening.
    They use a Kalman filter that says: "this person was at position X
    moving at velocity V, so next frame they'll be at X + V."

    But if the camera moved by 50 pixels to the right, the person
    appears 50 pixels to the LEFT — even if they didn't move at all.
    The Kalman prediction is now completely wrong, causing ID switches.

    This module estimates EXACTLY how the camera moved between frames
    using sparse optical flow on background pixels, expressed as a
    3x3 homography matrix H.

WHY OPTICAL FLOW + RANSAC (interview answer):
    - We track ~300 feature points between frames using Lucas-Kanade
    - But some of those points are ON people (foreground, not camera motion)
    - RANSAC automatically filters those out by finding the majority
      consensus (background pixels >> foreground pixels in drone footage)
    - Result: H describes pure camera motion, not object motion
"""

import cv2
import numpy as np


class EgoMotionCompensator:
    def __init__(self, max_corners: int = 300, quality: float = 0.01, min_dist: int = 7):
        """
        Args:
            max_corners: How many feature points to track. More = more accurate
                         but slower. 300 is a good balance.
            quality:     Minimum quality of corner features (0-1). Lower values
                         find weaker corners — useful for texture-poor aerial views.
            min_dist:    Minimum pixel distance between tracked points. Prevents
                         clustering all points in one region.
        """
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_dist,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=(21, 21),  # Larger window = handles faster motion better
            maxLevel=3,        # Pyramid levels — helps with large displacements
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray: np.ndarray | None = None

    def update(self, curr_gray: np.ndarray) -> tuple[np.ndarray | None, float]:
        """
        Call once per frame. Returns:
            H     — 3x3 homography (None if first frame or estimation fails)
            magnitude — scalar: how much the camera moved in pixels.
                        Used downstream to adapt detection sensitivity.
        """
        H, magnitude = None, 0.0

        if self.prev_gray is not None:
            H = self._estimate_homography(self.prev_gray, curr_gray)
            if H is not None:
                magnitude = self._motion_magnitude(H)

        self.prev_gray = curr_gray.copy()
        return H, magnitude

    def _estimate_homography(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray | None:
        # Step 1: Find strong corners in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev, mask=None, **self.feature_params)
        if prev_pts is None or len(prev_pts) < 8:
            return None

        # Step 2: Track them into current frame using Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, curr, prev_pts, None, **self.lk_params
        )
        if curr_pts is None:
            return None

        # Step 3: Keep only successfully tracked points
        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < 4:
            return None

        # Step 4: Fit homography with RANSAC — outliers (people, noise) are rejected
        H, inlier_mask = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)

        # Sanity check: if too few inliers, the homography is unreliable
        if inlier_mask is None or inlier_mask.sum() < 10:
            return None

        return H

    @staticmethod
    def _motion_magnitude(H: np.ndarray) -> float:
        """
        Extracts a scalar 'how much did the camera move' from H.

        H has the form:  | r00 r01 tx |
                         | r10 r11 ty |
                         | p0  p1  1  |

        tx, ty = translation in pixels.
        We also check the rotational/scale component via the Frobenius
        distance from the identity matrix.
        """
        tx, ty = H[0, 2], H[1, 2]
        translation = np.sqrt(tx ** 2 + ty ** 2)

        # Frobenius norm of (rotation submatrix - identity) captures rotation/shear
        rot_deviation = np.linalg.norm(H[:2, :2] - np.eye(2), ord='fro')

        # Weighted sum: translation dominates, rotation adds smaller component
        return float(translation + rot_deviation * 20.0)

    @staticmethod
    def warp_points(points: list[tuple], H: np.ndarray) -> list[tuple]:
        """Transforms a list of (x, y) pixel coords through homography H."""
        if not points or H is None:
            return points
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return [(int(p[0][0]), int(p[0][1])) for p in warped]

    @staticmethod
    def warp_boxes(xyxy: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warps an array of bounding boxes [x1,y1,x2,y2] through homography H.

        We warp all 4 corners of each box (not just the centre) because
        perspective transforms can skew boxes. The new box is the axis-
        aligned bounding rect of the 4 warped corners.
        """
        if len(xyxy) == 0 or H is None:
            return xyxy
        result = []
        for x1, y1, x2, y2 in xyxy:
            corners = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            ).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            result.append([
                warped[:, 0].min(), warped[:, 1].min(),
                warped[:, 0].max(), warped[:, 1].max(),
            ])
        return np.array(result, dtype=np.float32)
