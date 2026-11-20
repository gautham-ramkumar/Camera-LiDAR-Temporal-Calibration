"""Scan-to-local-map LiDAR odometry with per-point deskew.

The repo's IMU preintegrator starts every window from zero velocity and has
no IMU→LiDAR extrinsic, so it cannot give the LiDAR-frame translation that
per-point deskew needs (at 5 m/s, 50 ms is 25 cm). This module estimates the
LiDAR trajectory directly from the scans instead:

* each scan's key pose is placed at mid-sweep (``stamp + 0.05 s``);
* points are deskewed to that time with the current constant-velocity guess,
  using every point's own ``t``;
* the deskewed, voxel-downsampled scan is aligned to a local map of the last
  few scans with robust point-to-plane ICP;
* deskew + ICP is repeated once with the updated velocity.

Only local relative motion (≈ ±0.15 s) matters for deskew, so slow drift of
the odometry does not affect the calibration.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .trajectory import Trajectory


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def estimate_normals(points: np.ndarray, tree: cKDTree, k: int = 10):
    """PCA normals and planarity for every map point."""
    _, nn = tree.query(points, k=k)
    nb = points[nn]                                   # (N,k,3)
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    w, v = np.linalg.eigh(cov)                        # ascending
    normals = v[:, :, 0]
    planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-9)
    return normals, planarity


def icp_point_to_plane(
    src: np.ndarray,
    map_pts: np.ndarray,
    map_normals: np.ndarray,
    tree: cKDTree,
    R0: np.ndarray,
    p0: np.ndarray,
    max_dist: float = 1.0,
    iters: int = 20,
    huber: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align ``src`` (body frame) to the map; returns world pose ``(R, p)`` and inlier count."""
    R, p = R0.copy(), p0.copy()
    n_in = 0
    for it in range(iters):
        q = src @ R.T + p
        d, j = tree.query(q, k=1, distance_upper_bound=max_dist)
        ok = np.isfinite(d)
        n_in = int(ok.sum())
        if n_in < 30:
            break
        qs, m, n = q[ok], map_pts[j[ok]], map_normals[j[ok]]
        r = np.einsum("ni,ni->n", qs - m, n)
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        J = np.hstack([np.cross(qs, n), n])          # left perturbation on world pose
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r
        dx = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
        dR = Rotation.from_rotvec(dx[:3]).as_matrix()
        R = dR @ R
        p = dR @ p + dx[3:]
        if np.linalg.norm(dx[:3]) < 1e-5 and np.linalg.norm(dx[3:]) < 1e-4:
            break
        if it == 5:
            max_dist = max(0.5, max_dist * 0.5)
    return R, p, n_in


def _deskew_body(points: np.ndarray, dt: np.ndarray, w: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Constant body twist: point captured ``dt`` after the key time → key frame."""
    Rs = Rotation.from_rotvec(dt[:, None] * w[None, :]).as_matrix()
    return np.einsum("nij,nj->ni", Rs, points) + dt[:, None] * v_body[None, :]


def run_lidar_odometry(
    n_scans: int,
    load_scan: Callable[[int], tuple[np.ndarray, np.ndarray, float]],
    voxel: float = 0.6,
    map_voxel: float = 0.4,
    map_window: int = 6,
    min_range: float = 2.0,
    max_range: float = 80.0,
    sweep_mid: float = 0.05,
    verbose: bool = True,
) -> tuple[Trajectory, dict]:
    """Odometry over all scans.

    ``load_scan(k)`` returns ``(points (N,3) LiDAR frame, t_rel (N,) seconds
    after the header stamp, header stamp)``; invalid times must already be
    removed.
    """
    times, Rs, ps, inliers = [], [], [], []
    map_frames: deque = deque(maxlen=map_window)
    w_body = np.zeros(3)
    v_body = np.zeros(3)
    for k in range(n_scans):
        pts, t_rel, stamp = load_scan(k)
        r = np.linalg.norm(pts, axis=1)
        keep = (r > min_range) & (r < max_range)
        pts, t_rel = pts[keep].astype(np.float64), t_rel[keep]
        t_key = stamp + sweep_mid
        dt = (stamp + t_rel) - t_key

        if k == 0:
            R, p = np.eye(3), np.zeros(3)
            src = voxel_downsample(pts, voxel)
            n_in = src.shape[0]
        else:
            # constant-velocity prediction
            R_prev, p_prev = Rs[-1], ps[-1]
            h = t_key - times[-1]
            R_pred = R_prev @ Rotation.from_rotvec(w_body * h).as_matrix()
            p_pred = p_prev + R_prev @ (v_body * h)
            map_pts = np.vstack(map_frames)
            map_pts = voxel_downsample(map_pts, map_voxel)
            tree = cKDTree(map_pts)
            normals, planarity = estimate_normals(map_pts, tree)
            good = planarity > 0.3
            map_pts, normals = map_pts[good], normals[good]
            tree = cKDTree(map_pts)
            R, p = R_pred, p_pred
            for _pass in range(2):
                src = voxel_downsample(_deskew_body(pts, dt, w_body, v_body), voxel)
                R, p, n_in = icp_point_to_plane(src, map_pts, normals, tree, R, p)
                # update twist from previous key pose → this key pose
                rel_R = R_prev.T @ R
                w_body = Rotation.from_matrix(rel_R).as_rotvec() / h
                v_body = R_prev.T @ (p - p_prev) / h
        times.append(t_key)
        Rs.append(R)
        ps.append(p)
        inliers.append(n_in)
        full = voxel_downsample(_deskew_body(pts, dt, w_body, v_body), map_voxel)
        map_frames.append(full @ R.T + p)
        if verbose and k % 100 == 0:
            speed = np.linalg.norm(v_body)
            print(f"  odom {k}/{n_scans}  speed={speed:.2f} m/s  yaw_rate={np.degrees(np.linalg.norm(w_body)):.1f} deg/s  inliers={n_in}", flush=True)
    traj = Trajectory(np.asarray(times), np.asarray(Rs), np.asarray(ps))
    return traj, {"inliers": np.asarray(inliers)}
