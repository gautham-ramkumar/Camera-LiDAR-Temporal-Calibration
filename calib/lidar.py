"""LiDAR range-image projection, edge extraction, and IMU densification."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .imu import IMUPreintegrator


def project_to_range_image(
    points: np.ndarray,
    height: int = 128,
    width: int = 1024,
    fov_up: float = 22.5,
    fov_down: float = -22.5,
) -> np.ndarray:
    """Project 3D points to a (height, width, 3) grid in LiDAR coordinates."""
    total = height * width
    if points.shape[0] == total:
        return points.reshape(height, width, 3)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    valid_mask = r > 0.001

    u = np.zeros_like(x, dtype=int)
    v = np.zeros_like(x, dtype=int)

    if np.any(valid_mask):
        x_v, y_v, z_v, r_v = x[valid_mask], y[valid_mask], z[valid_mask], r[valid_mask]
        yaw = np.arctan2(y_v, x_v)
        pitch = np.arcsin(np.clip(z_v / r_v, -1, 1))

        u_val = 0.5 * (yaw / np.pi + 1.0) * width
        fov_range = np.deg2rad(fov_up - fov_down)
        v_val = (1.0 - (pitch - np.deg2rad(fov_down)) / fov_range) * height

        u[valid_mask] = u_val.astype(int)
        v[valid_mask] = v_val.astype(int)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    img = np.zeros((height, width, 3), dtype=np.float32)
    valid = valid_mask & (v >= 0) & (v < height) & (u >= 0) & (u < width)
    img[v[valid], u[valid]] = points[valid]
    return img


def extract_edges(grid_points: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Return edge points where range jumps exceed ``threshold`` meters."""
    ranges = np.linalg.norm(grid_points, axis=2)
    diff = np.abs(np.diff(ranges, axis=1))
    diff = np.pad(diff, ((0, 0), (0, 1)), constant_values=0)
    return grid_points[diff > threshold]


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform ``T`` to Nx3 points."""
    if len(points) == 0:
        return points
    ones = np.ones((len(points), 1), dtype=points.dtype)
    pts_hom = np.hstack((points, ones))
    pts_out = (T @ pts_hom.T).T
    return pts_out[:, :3]


def densify_lidar_scans(
    lidar_scans: List[Dict],
    imu_measurements: List[Dict],
    imu_integrator: Optional[IMUPreintegrator] = None,
    height: int = 128,
    width: int = 1024,
    fov_up: float = 22.5,
    fov_down: float = -22.5,
    edge_threshold: float = 1.0,
) -> List[Dict]:
    """
    Build densified, motion-compensated LiDAR edge clouds.

    For each frame with a 3-frame window (prev, curr, next), edges from prev/next
    are warped into the current frame via IMU preintegration and stacked.

    Returns a list of ``{"timestamp": float, "points": (N,3) float32}``.
    """
    if imu_integrator is None:
        imu_integrator = IMUPreintegrator(gravity=np.array([0.0, 0.0, 9.81]))

    window: List[Dict] = []
    processed: List[Dict] = []

    for scan in lidar_scans:
        raw_points = scan["points"]
        timestamp = scan["timestamp_sec"] if "timestamp_sec" in scan else scan["timestamp"]

        grid = project_to_range_image(
            raw_points, height=height, width=width, fov_up=fov_up, fov_down=fov_down
        )
        edges = extract_edges(grid, threshold=edge_threshold)
        if edges.size == 0:
            continue

        window.append({"t": timestamp, "edges": edges})
        if len(window) > 3:
            window.pop(0)

        if len(window) < 3:
            continue

        f_prev, f_curr, f_next = window[0], window[1], window[2]

        T_p2c = imu_integrator.preintegrate(imu_measurements, f_prev["t"], f_curr["t"])
        edges_prev = apply_transform(f_prev["edges"], T_p2c)

        T_c2n = imu_integrator.preintegrate(imu_measurements, f_curr["t"], f_next["t"])
        edges_next = apply_transform(f_next["edges"], np.linalg.inv(T_c2n))

        dense = np.vstack([edges_prev, f_curr["edges"], edges_next]).astype(np.float32)
        processed.append({"timestamp": float(f_curr["t"]), "points": dense})

    return processed
