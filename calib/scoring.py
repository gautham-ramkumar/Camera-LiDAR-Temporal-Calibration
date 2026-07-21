"""Edge-alignment scoring with time-blended distance transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .camera import CameraPipeline
from .geometry import lidar_points_to_pixels


@dataclass
class WindowCalibResult:
    window_idx: int
    frame_start: int
    frame_end: int
    tau: float
    T_CL: np.ndarray
    score: float


def blended_distances(
    cam_times: np.ndarray,
    cam_dts: Sequence[np.ndarray],
    t_target: float,
    ui: np.ndarray,
    vi: np.ndarray,
) -> np.ndarray:
    """Linearly blend DT samples between bracketing camera frames."""
    cam_times = np.asarray(cam_times, dtype=np.float64)
    if len(cam_times) == 0:
        return np.zeros(len(ui), dtype=np.float32)
    if len(cam_times) == 1:
        return cam_dts[0][vi, ui]

    idx = int(np.searchsorted(cam_times, t_target))
    idx_hi = min(len(cam_times) - 1, idx)
    idx_lo = max(0, idx_hi - 1)
    t_lo, t_hi = float(cam_times[idx_lo]), float(cam_times[idx_hi])
    denom = t_hi - t_lo
    if denom <= 1e-9:
        return cam_dts[idx_lo][vi, ui]
    alpha = float(np.clip((t_target - t_lo) / denom, 0.0, 1.0))
    return (1.0 - alpha) * cam_dts[idx_lo][vi, ui] + alpha * cam_dts[idx_hi][vi, ui]


def soft_alignment_from_distances(d_vals: np.ndarray, sigma_px: float) -> float:
    if d_vals.size == 0:
        return 0.0
    return float(np.sum(np.exp(-(d_vals ** 2) / (2.0 * sigma_px ** 2))))


def mean_alignment_score(
    pipeline: CameraPipeline,
    lidar_scans: List[dict],
    cam_times: np.ndarray,
    frame_ids: np.ndarray,
    K: np.ndarray,
    T_CL: np.ndarray,
    tau: float,
    scan_indices: Optional[np.ndarray] = None,
    max_points_per_scan: Optional[int] = 3000,
    point_override: Optional[dict] = None,
) -> float:
    """
    Mean soft edge-alignment per scan (higher is better).

    ``point_override`` maps scan index → points already warped into the
    evaluation frame (paper-style Tv); otherwise uses ``scan['points']``.
    """
    cam_times = np.asarray(cam_times, dtype=np.float64)
    frame_ids = np.asarray(frame_ids)
    if scan_indices is None:
        scan_indices = np.arange(len(lidar_scans))
    else:
        scan_indices = np.asarray(scan_indices)

    cam_dts = [pipeline.dist_transforms[int(fid)] for fid in frame_ids]
    sigma = float(pipeline.sigma_px)
    total_score = 0.0
    valid_count = 0

    for idx in scan_indices:
        i = int(idx)
        scan = lidar_scans[i]
        t_L = float(scan["timestamp"])
        if point_override is not None and i in point_override:
            pts_L = point_override[i]
        else:
            pts_L = scan["points"]

        num_pts = pts_L.shape[0]
        if max_points_per_scan is not None and num_pts > max_points_per_scan:
            sel = np.linspace(0, num_pts - 1, max_points_per_scan, dtype=int)
            pts_L_sub = pts_L[sel]
        else:
            pts_L_sub = pts_L

        uv = lidar_points_to_pixels(pts_L_sub, T_CL, K)
        if uv.shape[0] == 0:
            continue

        h, w = cam_dts[0].shape
        pts = uv.astype(np.int32)
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[mask]
        if pts.shape[0] == 0:
            continue

        d = blended_distances(cam_times, cam_dts, t_L + tau, pts[:, 0], pts[:, 1])
        total_score += soft_alignment_from_distances(d, sigma)
        valid_count += 1

    if valid_count == 0:
        return 0.0
    return total_score / valid_count


def tau_only_objective(
    tau_vec: Sequence[float],
    pipeline: CameraPipeline,
    lidar_scans: List[dict],
    cam_times: np.ndarray,
    frame_ids: np.ndarray,
    K: np.ndarray,
    T_CL: np.ndarray,
    scan_indices: Optional[np.ndarray] = None,
    max_points_per_scan: Optional[int] = 5000,
) -> float:
    """Negative mean alignment score (for minimize)."""
    tau = float(np.asarray(tau_vec).reshape(-1)[0])
    return -mean_alignment_score(
        pipeline,
        lidar_scans,
        cam_times,
        frame_ids,
        K,
        T_CL,
        tau,
        scan_indices=scan_indices,
        max_points_per_scan=max_points_per_scan,
    )


def seed_tau_grid(
    pipeline: CameraPipeline,
    lidar_scans: List[dict],
    cam_times: np.ndarray,
    frame_ids: np.ndarray,
    K: np.ndarray,
    T_CL: np.ndarray,
    scan_indices: np.ndarray,
    max_points_per_scan: int = 2000,
    tau_min: float = -0.05,
    tau_max: float = 0.05,
    num_samples: int = 41,
    point_override: Optional[dict] = None,
) -> float:
    """1-D grid search for tau maximizing alignment score."""
    taus = np.linspace(tau_min, tau_max, num_samples)
    best_tau, best_score = 0.0, -np.inf
    for t in taus:
        s = mean_alignment_score(
            pipeline,
            lidar_scans,
            cam_times,
            frame_ids,
            K,
            T_CL,
            float(t),
            scan_indices=scan_indices,
            max_points_per_scan=max_points_per_scan,
            point_override=point_override,
        )
        if s > best_score:
            best_score = s
            best_tau = float(t)
    return best_tau
