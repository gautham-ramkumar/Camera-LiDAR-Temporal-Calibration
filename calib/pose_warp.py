"""IMU pose warp Tv — paper-style motion between LiDAR and camera times."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .geometry import transform_points
from .imu import IMUPreintegrator


def relative_motion(
    imu: IMUPreintegrator,
    imu_measurements: List[dict],
    t_from: float,
    t_to: float,
) -> np.ndarray:
    """
    4x4 motion of the body from ``t_from`` to ``t_to``.

    If ``t_to < t_from``, returns the inverse of the forward preintegration.
    """
    if abs(t_to - t_from) < 1e-6:
        return np.eye(4)
    if t_to >= t_from:
        return imu.preintegrate(imu_measurements, t_from, t_to)
    T = imu.preintegrate(imu_measurements, t_to, t_from)
    return np.linalg.inv(T)


def warp_points_by_motion(points: np.ndarray, T_motion: np.ndarray) -> np.ndarray:
    """Apply SE(3) motion to Nx3 points."""
    return transform_points(np.asarray(points, dtype=np.float64), T_motion)


def warp_scan_to_camera_time(
    points_L: np.ndarray,
    t_lidar: float,
    tau: float,
    imu: IMUPreintegrator,
    imu_measurements: List[dict],
) -> np.ndarray:
    """
    Warp LiDAR points from ``t_lidar`` toward camera time ``t_L + tau``.

    Paper model: project adjacent-moment geometry with Tv over et = tau
    (when stamps already associate nearby frames, et is small).
    """
    t_cam = float(t_lidar) + float(tau)
    Tv = relative_motion(imu, imu_measurements, float(t_lidar), t_cam)
    return warp_points_by_motion(points_L, Tv)


def build_warped_point_override(
    lidar_scans: List[dict],
    scan_indices: np.ndarray,
    tau: float,
    imu: Optional[IMUPreintegrator],
    imu_measurements: Optional[List[dict]],
    max_points: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Map scan index → warped points for scoring at a candidate tau."""
    out: Dict[int, np.ndarray] = {}
    if imu is None or not imu_measurements:
        return out
    for idx in scan_indices:
        i = int(idx)
        scan = lidar_scans[i]
        pts = np.asarray(scan["points"], dtype=np.float64)
        if max_points is not None and pts.shape[0] > max_points:
            sel = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
            pts = pts[sel]
        t_L = float(scan["timestamp"])
        out[i] = warp_scan_to_camera_time(pts, t_L, tau, imu, imu_measurements)
    return out
