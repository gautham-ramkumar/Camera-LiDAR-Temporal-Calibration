"""Line features: LSD (with Canny fallback) + LiDAR range-edge points."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def image_line_mask(
    gray: np.ndarray,
    *,
    use_lsd: bool = True,
    canny_low: int = 50,
    canny_high: int = 150,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Binary mask (0/255) of image line structure.

    Prefers OpenCV LSD when available; falls back to Canny.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    drew = False
    if use_lsd:
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lines, *_ = lsd.detect(gray)
            if lines is not None:
                for ln in lines:
                    x1, y1, x2, y2 = ln[0]
                    cv2.line(
                        mask,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        255,
                        line_thickness,
                    )
                drew = True
        except Exception:
            drew = False

    if not drew:
        edges = cv2.Canny(gray, canny_low, canny_high)
        mask = (edges > 0).astype(np.uint8) * 255
    return mask


def dist_transform_from_mask(mask: np.ndarray) -> np.ndarray:
    """Raw-pixel DT; edge pixels are 0."""
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)


def split_lidar_lines_hv(
    points: np.ndarray,
    vertical_z_thresh: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rough H/V split of LiDAR edge points (paper [19]-style cue).

    Vertical: large |Δz| along local neighborhoods approximated by |z| spread
    vs XY — here a simple |z| vs radial heuristic for densified edges.
    """
    if points.shape[0] == 0:
        z = np.zeros((0, 3))
        return z, z
    pts = np.asarray(points, dtype=np.float64)
    # Near-vertical structure: larger |z| relative to ground plane distance
    r_xy = np.linalg.norm(pts[:, :2], axis=1) + 1e-6
    vert_score = np.abs(pts[:, 2]) / r_xy
    is_v = vert_score > vertical_z_thresh
    return pts[is_v], pts[~is_v]


def apply_line_dt_to_pipeline(
    pipeline,
    frame_id: int,
    *,
    use_lsd: bool = True,
) -> np.ndarray:
    """
    Recompute edges/DT for one frame using LSD (fallback Canny).

    Mutates ``pipeline.edge_maps`` / ``dist_transforms``.
    """
    gray = pipeline.images[frame_id]
    mask = image_line_mask(
        gray,
        use_lsd=use_lsd,
        canny_low=getattr(pipeline, "canny_low", 50),
        canny_high=getattr(pipeline, "canny_high", 150),
    )
    pipeline.edge_maps[frame_id] = mask
    dt = dist_transform_from_mask(mask)
    pipeline.dist_transforms[frame_id] = dt
    return dt
