"""Lightweight dynamic-point filter via adjacent-frame NN (paper-style)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from .geometry import transform_points
from .imu import IMUPreintegrator
from .pose_warp import relative_motion


def filter_dynamic_points(
    points_curr: np.ndarray,
    points_prev: np.ndarray,
    T_prev_to_curr: np.ndarray,
    *,
    base_thresh_m: float = 0.35,
    range_scale: float = 0.02,
    min_keep_ratio: float = 0.25,
) -> np.ndarray:
    """
    Keep points in ``points_curr`` that match a warped previous cloud.

    Dynamic / inconsistent points have large NN distance after warping
    prev → curr. Threshold grows with range (distant objects need slack).
    If too few points remain, return the original cloud (safety).
    """
    curr = np.asarray(points_curr, dtype=np.float64)
    prev = np.asarray(points_prev, dtype=np.float64)
    if curr.shape[0] == 0 or prev.shape[0] < 10:
        return curr

    prev_w = transform_points(prev, T_prev_to_curr)
    tree = cKDTree(prev_w)
    # Subsample curr for speed if huge
    idx = np.arange(curr.shape[0])
    if curr.shape[0] > 20000:
        idx = np.linspace(0, curr.shape[0] - 1, 20000, dtype=int)
    sample = curr[idx]
    dists, _ = tree.query(sample, k=1, workers=-1)
    ranges = np.linalg.norm(sample, axis=1)
    thresh = base_thresh_m + range_scale * ranges
    keep_sample = dists <= thresh

    if idx.shape[0] == curr.shape[0]:
        kept = curr[keep_sample]
    else:
        # Map sample keep back roughly: keep all if sample mostly static
        keep_ratio = float(np.mean(keep_sample))
        if keep_ratio < min_keep_ratio:
            return curr
        # Re-query full set (may be slow); prefer sample mask expansion
        dists_full, _ = tree.query(curr, k=1, workers=-1)
        ranges_full = np.linalg.norm(curr, axis=1)
        kept = curr[dists_full <= (base_thresh_m + range_scale * ranges_full)]

    if kept.shape[0] < min_keep_ratio * curr.shape[0]:
        return curr
    return kept


def filter_scan_pair(
    scan_curr: dict,
    scan_prev: dict,
    imu: IMUPreintegrator,
    imu_measurements: List[dict],
    **kwargs,
) -> np.ndarray:
    """Filter ``scan_curr['points']`` using previous scan + IMU motion."""
    t0 = float(scan_prev["timestamp"])
    t1 = float(scan_curr["timestamp"])
    T = relative_motion(imu, imu_measurements, t0, t1)
    return filter_dynamic_points(
        scan_curr["points"],
        scan_prev["points"],
        T,
        **kwargs,
    )


def apply_dynamic_filter_to_scans(
    lidar_scans: List[dict],
    scan_indices: np.ndarray,
    imu: Optional[IMUPreintegrator],
    imu_measurements: Optional[List[dict]],
    **kwargs,
) -> dict:
    """
    Return point_override dict with dynamically filtered clouds.

    First index in the window keeps original points (no previous).
    """
    out = {}
    if imu is None or not imu_measurements:
        return out
    indices = [int(i) for i in scan_indices]
    for k, i in enumerate(indices):
        if k == 0:
            out[i] = np.asarray(lidar_scans[i]["points"], dtype=np.float64)
            continue
        prev_i = indices[k - 1]
        out[i] = filter_scan_pair(
            lidar_scans[i],
            {"timestamp": lidar_scans[prev_i]["timestamp"], "points": out.get(prev_i, lidar_scans[prev_i]["points"])},
            imu,
            imu_measurements,
            **kwargs,
        )
    return out
