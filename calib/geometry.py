"""SE(3) helpers and LiDAR → camera projection."""

from __future__ import annotations

import numpy as np


def euler_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ Euler angles (radians) -> 3x3 rotation matrix (Rz @ Ry @ Rx)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) transform from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (T @ pts_h.T).T[:, :3]


def transform_lidar_to_cam(points_L: np.ndarray, T_CL: np.ndarray) -> np.ndarray:
    return transform_points(points_L, T_CL).astype(np.float32)


def project_cam_points_to_pixels(points_C: np.ndarray, K: np.ndarray) -> np.ndarray:
    if points_C.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    X, Y, Z = points_C[:, 0], points_C[:, 1], points_C[:, 2]
    mask = Z > 0
    X, Y, Z = X[mask], Y[mask], Z[mask]
    if X.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts_norm = np.vstack([X / Z, Y / Z, np.ones_like(X)])
    uv_h = K @ pts_norm
    return np.stack([uv_h[0, :], uv_h[1, :]], axis=1).astype(np.float32)


def lidar_points_to_pixels(points_L: np.ndarray, T_CL: np.ndarray, K: np.ndarray) -> np.ndarray:
    return project_cam_points_to_pixels(transform_lidar_to_cam(points_L, T_CL), K)


def closest_frame_id(cam_times: np.ndarray, frame_ids: np.ndarray, t_C: float) -> int:
    idx = int(np.argmin(np.abs(np.asarray(cam_times) - t_C)))
    return int(np.asarray(frame_ids)[idx])


def apply_extrinsic_delta(
    T_CL: np.ndarray,
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
) -> np.ndarray:
    """Right-multiply a small SE(3) delta onto T_CL."""
    return T_CL @ make_T(euler_to_R(rx, ry, rz), np.array([tx, ty, tz], dtype=float))
