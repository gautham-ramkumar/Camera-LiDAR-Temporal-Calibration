"""Slow extrinsic refine with stiff prior (upgrade over free paper Te search)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .camera import CameraPipeline
from .geometry import apply_extrinsic_delta
from .scoring import mean_alignment_score


def refine_extrinsics(
    pipeline: CameraPipeline,
    lidar_scans: List[dict],
    cam_times: np.ndarray,
    frame_ids: np.ndarray,
    K: np.ndarray,
    T_CL: np.ndarray,
    tau: float,
    scan_indices: np.ndarray,
    *,
    max_points_per_scan: int = 2000,
    rot_bound_deg: float = 2.0,
    trans_bound_m: float = 0.05,
    rot_steps: int = 3,
    trans_steps: int = 3,
    point_override: Optional[dict] = None,
) -> Tuple[np.ndarray, float]:
    """
    Coarse grid on small δR/δt around ``T_CL`` with ``tau`` fixed.

    Bounds keep Te stiff (paper free search can wander; we clip).
    Returns (T_CL_refined, best_score).
    """
    rb = np.deg2rad(rot_bound_deg)
    rots = np.linspace(-rb, rb, rot_steps)
    trans = np.linspace(-trans_bound_m, trans_bound_m, trans_steps)

    best_T = np.asarray(T_CL, dtype=np.float64).copy()
    best_score = mean_alignment_score(
        pipeline,
        lidar_scans,
        cam_times,
        frame_ids,
        K,
        best_T,
        tau,
        scan_indices=scan_indices,
        max_points_per_scan=max_points_per_scan,
        point_override=point_override,
    )

    # Coordinate-wise then joint light search (keep cheap)
    for axis, name in enumerate(["rx", "ry", "rz"]):
        for a in rots:
            kwargs = {"rx": 0.0, "ry": 0.0, "rz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}
            kwargs[name] = float(a)
            T = apply_extrinsic_delta(T_CL, **kwargs)
            s = mean_alignment_score(
                pipeline,
                lidar_scans,
                cam_times,
                frame_ids,
                K,
                T,
                tau,
                scan_indices=scan_indices,
                max_points_per_scan=max_points_per_scan,
                point_override=point_override,
            )
            if s > best_score:
                best_score = s
                best_T = T

    T_seed = best_T.copy()
    for axis, name in enumerate(["tx", "ty", "tz"]):
        for a in trans:
            kwargs = {"rx": 0.0, "ry": 0.0, "rz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}
            kwargs[name] = float(a)
            T = apply_extrinsic_delta(T_seed, **kwargs)
            s = mean_alignment_score(
                pipeline,
                lidar_scans,
                cam_times,
                frame_ids,
                K,
                T,
                tau,
                scan_indices=scan_indices,
                max_points_per_scan=max_points_per_scan,
                point_override=point_override,
            )
            if s > best_score:
                best_score = s
                best_T = T

    return best_T, float(best_score)
