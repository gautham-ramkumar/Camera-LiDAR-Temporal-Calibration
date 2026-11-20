"""Build :class:`calib.joint_calib.JointFrame` objects from the bag cache."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .bag_cache import BagCache, valid_point_times
from .joint_calib import JointFrame, image_distance_transform, lidar_depth_edges, make_frame
from .trajectory import Trajectory, deskew_points


def scan_edge_points(cache: BagCache, k: int, min_range: float = 1.5, max_range: float = 60.0):
    """Foreground depth-edge points of scan ``k`` with their absolute capture times.

    Points with an out-of-sweep ``t`` (the bad scan near 53.7 s) are dropped.
    Returns ``(points (N,3), t_abs (N,), n_bad_time)``.
    """
    s = cache.scan(k)
    xyz, t_ns, rng = s["xyz"].astype(np.float64), s["t_ns"], s["range"].astype(np.float64)
    finite = np.isfinite(xyz).all(axis=2) & (rng > 0)
    t_ok = valid_point_times(t_ns)
    n_bad = int((finite & ~t_ok).sum())
    valid = finite & t_ok & (rng >= min_range) & (rng <= max_range)
    mask = lidar_depth_edges(rng, valid) & valid
    pts = xyz[mask]
    t_abs = float(cache.lidar_t[k]) + t_ns[mask].astype(np.float64) * 1e-9
    return pts, t_abs, n_bad


def prepare_frames(
    cache: BagCache,
    traj: Trajectory,
    scan_indices: Sequence[int],
    K: np.ndarray,
    T_init: np.ndarray,
    dt_init: float,
    blocks: Optional[Sequence[str]] = None,
    max_points: int = 1500,
    margin_px: float = -20.0,
    seed: int = 0,
) -> List[JointFrame]:
    """Pair each scan with the camera frame nearest (by stamp) to the time the
    camera sector was swept, and keep only edge points near the image.

    The pairing uses stamps only; it chooses *which* image, while the
    optimiser's Δt and the per-point deskew decide *where* points land.
    """
    rng = np.random.default_rng(seed)
    cam_t = cache.cam_t
    frames: List[JointFrame] = []
    for n, k in enumerate(scan_indices):
        pts, t_abs, _ = scan_edge_points(cache, int(k))
        if pts.shape[0] < 50:
            continue
        # Points roughly in front of the camera (initial guess) — their sweep time picks the image.
        pc = pts @ T_init[:3, :3].T + T_init[:3, 3]
        front = pc[:, 2] > 1.0
        if front.sum() < 50:
            continue
        t_fov = float(np.median(t_abs[front]))
        j = int(np.argmin(np.abs(cam_t - t_fov)))
        s_C = float(cam_t[j])
        # Image-region filter at the initial guess (deskewed to s_C - dt_init).
        pL = deskew_points(traj, pts, t_abs, s_C - dt_init)
        pc = pL @ T_init[:3, :3].T + T_init[:3, 3]
        z = pc[:, 2]
        zs = np.where(z > 1.0, z, 1.0)
        u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
        v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
        img = cache.image(j)
        h, w = img.shape
        keep = (z > 1.0) & (u > -margin_px) & (u < w + margin_px) & (v > -margin_px) & (v < h + margin_px)
        if keep.sum() < 50:
            continue
        sel = np.flatnonzero(keep)
        if sel.size > max_points:
            sel = np.sort(rng.choice(sel, max_points, replace=False))
        f = make_frame(traj, pts[sel], t_abs[sel], float(cache.lidar_t[k]), s_C,
                       image_distance_transform(img), int(k), j,
                       blocks[n] if blocks is not None else "train")
        frames.append(f)
    return frames
