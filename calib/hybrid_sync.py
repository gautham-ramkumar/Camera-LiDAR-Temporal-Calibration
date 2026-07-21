"""
Hybrid camera–LiDAR sync (stamp-primary) with paper-closer residual path.

Operational:  tau_sync = tau_stamp + tau_res
Diagnostic:   tau_unconst (unconstrained score) — never overrides sync

Paper-closer (optional flags):
  - IMU Tv warp of LiDAR edges to camera time
  - LSD/Canny line DTs
  - Adjacent-frame dynamic filter
  - Slow Te refine with stiff bounds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .camera import CameraPipeline
from .dynamic_filter import apply_dynamic_filter_to_scans
from .extrinsic_refine import refine_extrinsics
from .imu import IMUPreintegrator
from .lines import apply_line_dt_to_pipeline
from .pose_warp import build_warped_point_override
from .scoring import WindowCalibResult, mean_alignment_score, seed_tau_grid

DEFAULT_RESIDUAL_BAND = 0.05
DEFAULT_EMA_ALPHA = 0.25
DEFAULT_ALIAS_WARN_MS = 80.0


@dataclass
class StampSyncStats:
    tau_stamp: float
    median_abs_gap: float
    mean_gap: float
    std_gap: float
    n_pairs: int
    gaps: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class HybridWindowResult:
    window_idx: int
    frame_start: int
    frame_end: int
    tau_stamp: float
    tau_res: float
    tau_sync: float
    score_at_sync: float
    score_at_stamp: float
    tau_unconst: Optional[float] = None
    alias_suspect: bool = False


@dataclass
class HybridSyncResult:
    tau_sync: float
    tau_stamp: float
    tau_res: float
    T_CL: np.ndarray
    stamp: StampSyncStats
    window_results: List[HybridWindowResult] = field(default_factory=list)
    tau_unconst: float = 0.0
    alias_suspect: bool = False
    health_ok: bool = True
    health_messages: List[str] = field(default_factory=list)
    T_CL_init: Optional[np.ndarray] = None

    @property
    def tau(self) -> float:
        return self.tau_sync

    @property
    def taus(self) -> np.ndarray:
        return np.array([w.tau_sync for w in self.window_results], dtype=float)

    @property
    def scores(self) -> np.ndarray:
        return np.array([w.score_at_sync for w in self.window_results], dtype=float)

    @property
    def tau_seed(self) -> float:
        return self.tau_stamp


def stamp_sync_stats(
    lidar_scans: Sequence[dict],
    cam_times: np.ndarray,
    scan_indices: Optional[np.ndarray] = None,
) -> StampSyncStats:
    cam_times = np.asarray(cam_times, dtype=np.float64)
    if scan_indices is None:
        scan_indices = np.arange(len(lidar_scans))
    gaps = []
    for i in scan_indices:
        scan = lidar_scans[int(i)]
        t_L = float(scan["timestamp"] if "timestamp" in scan else scan["timestamp_sec"])
        j = int(np.argmin(np.abs(cam_times - t_L)))
        gaps.append(float(cam_times[j] - t_L))
    gaps_arr = np.asarray(gaps, dtype=np.float64)
    if gaps_arr.size == 0:
        return StampSyncStats(0.0, 0.0, 0.0, 0.0, 0, gaps_arr)
    return StampSyncStats(
        tau_stamp=float(np.median(gaps_arr)),
        median_abs_gap=float(np.median(np.abs(gaps_arr))),
        mean_gap=float(np.mean(gaps_arr)),
        std_gap=float(np.std(gaps_arr)),
        n_pairs=int(gaps_arr.size),
        gaps=gaps_arr,
    )


def associate_camera_time(t_lidar: float, tau_sync: float) -> float:
    return float(t_lidar) + float(tau_sync)


def _prepare_points(
    lidar_scans: List[dict],
    indices: np.ndarray,
    tau: float,
    imu: Optional[IMUPreintegrator],
    imu_measurements: Optional[List[dict]],
    use_warp: bool,
    use_dynamic_filter: bool,
    max_points: int,
) -> Optional[dict]:
    override = None
    if use_dynamic_filter and imu is not None and imu_measurements:
        override = apply_dynamic_filter_to_scans(
            lidar_scans, indices, imu, imu_measurements
        )
    if use_warp and imu is not None and imu_measurements:
        if override is None:
            warped = build_warped_point_override(
                lidar_scans, indices, tau, imu, imu_measurements, max_points=max_points
            )
        else:
            local_scans = [
                {
                    "timestamp": lidar_scans[int(i)]["timestamp"],
                    "points": override.get(int(i), lidar_scans[int(i)]["points"]),
                }
                for i in indices
            ]
            local_idx = np.arange(len(indices))
            local_warp = build_warped_point_override(
                local_scans, local_idx, tau, imu, imu_measurements, max_points=max_points
            )
            warped = {int(indices[k]): local_warp[k] for k in local_warp}
        override = warped if warped else override
    return override


def run_hybrid_sync(
    pipeline: CameraPipeline,
    lidar_scans: List[dict],
    cam_times: np.ndarray,
    frame_ids: np.ndarray,
    K: np.ndarray,
    T_CL_init: np.ndarray,
    window_size: int = 50,
    max_points_per_scan: int = 2000,
    scan_stride: int = 1,
    residual_band: float = DEFAULT_RESIDUAL_BAND,
    residual_samples: int = 41,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    unconst_half_width: float = 0.5,
    unconst_samples: int = 101,
    alias_warn_s: float = DEFAULT_ALIAS_WARN_MS / 1000.0,
    diagnose_every_window: bool = False,
    imu: Optional[IMUPreintegrator] = None,
    imu_measurements: Optional[List[dict]] = None,
    use_pose_warp: bool = True,
    use_dynamic_filter: bool = True,
    use_lsd: bool = False,
    refine_extrinsics_flag: bool = True,
    on_window: Optional[Callable[[HybridWindowResult], None]] = None,
) -> HybridSyncResult:
    n = len(lidar_scans)
    stride = max(1, int(scan_stride))
    T_CL = np.asarray(T_CL_init, dtype=np.float64).copy()
    T_CL_0 = T_CL.copy()
    cam_times = np.asarray(cam_times, dtype=np.float64)
    frame_ids = np.asarray(frame_ids)

    if use_lsd:
        for fid in frame_ids:
            apply_line_dt_to_pipeline(pipeline, int(fid), use_lsd=True)

    all_idx = np.arange(0, n, stride, dtype=int)
    stamp = stamp_sync_stats(lidar_scans, cam_times, all_idx)
    tau_stamp = stamp.tau_stamp
    tau_res = 0.0

    print(f"\n{'='*72}")
    print("  HYBRID SYNC  |  primary=stamps  residual=edge(+warp/filter)  diag=unconst")
    print(f"{'='*72}")
    print(
        f"  [Primary] tau_stamp = {tau_stamp*1000:+.2f} ms  "
        f"(n={stamp.n_pairs}, |gap|med={stamp.median_abs_gap*1000:.2f} ms)"
    )
    print(
        f"  [Flags]   warp={use_pose_warp}  dyn_filter={use_dynamic_filter}  "
        f"lsd={use_lsd}  refine_Te={refine_extrinsics_flag}"
    )
    print(f"  [Refine]  ±{residual_band*1000:.0f} ms around stamp, EMA α={ema_alpha}")
    print(f"{'='*72}")
    print(
        f"  {'Win':>4}  {'Frames':>11}  {'stamp':>8}  {'res':>8}  {'sync':>8}  "
        f"{'score':>8}  {'flag':>6}"
    )
    print(f"{'='*72}")

    window_results: List[HybridWindowResult] = []
    w_idx = 0

    for ws in range(0, n, window_size):
        we = min(ws + window_size, n)
        indices = np.arange(ws, we, stride, dtype=int)
        if len(indices) == 0:
            continue

        st_w = stamp_sync_stats(lidar_scans, cam_times, indices)
        tau_stamp_w = st_w.tau_stamp

        override = _prepare_points(
            lidar_scans, indices, tau_stamp_w, imu, imu_measurements,
            use_pose_warp, use_dynamic_filter, max_points_per_scan,
        )

        tau_abs_best = seed_tau_grid(
            pipeline, lidar_scans, cam_times, frame_ids, K, T_CL, indices,
            max_points_per_scan=max_points_per_scan,
            tau_min=tau_stamp_w - residual_band,
            tau_max=tau_stamp_w + residual_band,
            num_samples=residual_samples,
            point_override=override,
        )
        tau_res_obs = float(tau_abs_best - tau_stamp_w)
        tau_res = (1.0 - ema_alpha) * tau_res + ema_alpha * tau_res_obs
        tau_res = float(np.clip(tau_res, -residual_band, residual_band))
        tau_sync_w = float(tau_stamp_w + tau_res)

        override_sync = _prepare_points(
            lidar_scans, indices, tau_sync_w, imu, imu_measurements,
            use_pose_warp, use_dynamic_filter, max_points_per_scan,
        )

        score_sync = mean_alignment_score(
            pipeline, lidar_scans, cam_times, frame_ids, K, T_CL, tau_sync_w,
            scan_indices=indices, max_points_per_scan=max_points_per_scan,
            point_override=override_sync,
        )
        score_stamp = mean_alignment_score(
            pipeline, lidar_scans, cam_times, frame_ids, K, T_CL, tau_stamp_w,
            scan_indices=indices, max_points_per_scan=max_points_per_scan,
            point_override=override,
        )

        tau_u = None
        alias = False
        if diagnose_every_window:
            tau_u = seed_tau_grid(
                pipeline, lidar_scans, cam_times, frame_ids, K, T_CL, indices,
                max_points_per_scan=max_points_per_scan,
                tau_min=tau_stamp_w - unconst_half_width,
                tau_max=tau_stamp_w + unconst_half_width,
                num_samples=min(41, unconst_samples),
            )
            alias = abs(tau_u - tau_stamp_w) > alias_warn_s

        wr = HybridWindowResult(
            window_idx=w_idx,
            frame_start=ws,
            frame_end=we,
            tau_stamp=tau_stamp_w,
            tau_res=tau_res,
            tau_sync=tau_sync_w,
            score_at_sync=float(score_sync),
            score_at_stamp=float(score_stamp),
            tau_unconst=tau_u,
            alias_suspect=alias,
        )
        window_results.append(wr)
        if on_window is not None:
            on_window(wr)

        print(
            f"  {w_idx:>4}  {ws:>5}-{we:<5}  {tau_stamp_w*1000:>7.1f}  "
            f"{tau_res*1000:>7.1f}  {tau_sync_w*1000:>7.1f}  "
            f"{score_sync:>8.1f}  {('ALIAS' if alias else 'ok'):>6}"
        )
        w_idx += 1

    print(f"{'='*72}")

    override_f = _prepare_points(
        lidar_scans, all_idx, tau_stamp, imu, imu_measurements,
        use_pose_warp, use_dynamic_filter, max_points_per_scan,
    )
    tau_abs_f = seed_tau_grid(
        pipeline, lidar_scans, cam_times, frame_ids, K, T_CL, all_idx,
        max_points_per_scan=max_points_per_scan,
        tau_min=tau_stamp - residual_band,
        tau_max=tau_stamp + residual_band,
        num_samples=residual_samples,
        point_override=override_f,
    )
    tau_res = float(np.clip(tau_abs_f - tau_stamp, -residual_band, residual_band))
    tau_sync = float(tau_stamp + tau_res)

    if refine_extrinsics_flag:
        print("\n[Te] Slow extrinsic refine (stiff bounds, tau fixed) ...")
        mid = window_results[len(window_results) // 2] if window_results else None
        ref_idx = (
            np.arange(mid.frame_start, mid.frame_end, stride, dtype=int)
            if mid is not None
            else all_idx[: min(50, len(all_idx))]
        )
        ov = _prepare_points(
            lidar_scans, ref_idx, tau_sync, imu, imu_measurements,
            use_pose_warp, use_dynamic_filter, max_points_per_scan,
        )
        T_CL, te_score = refine_extrinsics(
            pipeline, lidar_scans, cam_times, frame_ids, K, T_CL, tau_sync, ref_idx,
            max_points_per_scan=max_points_per_scan, point_override=ov,
        )
        print(f"  Te refined score={te_score:.1f}")

    print(f"\n[Diagnostic] Unconstrained score ±{unconst_half_width*1000:.0f} ms ...")
    mid = window_results[len(window_results) // 2] if window_results else None
    diag_idx = (
        np.arange(mid.frame_start, mid.frame_end, stride, dtype=int)
        if mid is not None
        else all_idx[: max(1, min(50, len(all_idx)))]
    )
    tau_unconst = seed_tau_grid(
        pipeline, lidar_scans, cam_times, frame_ids, K, T_CL, diag_idx,
        max_points_per_scan=max_points_per_scan,
        tau_min=-unconst_half_width,
        tau_max=unconst_half_width,
        num_samples=unconst_samples,
    )
    alias_suspect = abs(tau_unconst - tau_stamp) > alias_warn_s

    health_messages: List[str] = []
    health_ok = True
    if stamp.median_abs_gap > 0.05:
        health_ok = False
        health_messages.append(
            f"Large stamp |gap| median={stamp.median_abs_gap*1000:.1f} ms"
        )
    if alias_suspect:
        health_messages.append(
            f"Edge score prefers {tau_unconst*1000:+.1f} ms vs stamp "
            f"{tau_stamp*1000:+.1f} ms — alias, not primary sync"
        )
    if not health_messages:
        health_messages.append("Stamp sync stable; residual in-band; no alias flag")

    print(f"\n  === OPERATIONAL ===")
    print(f"  tau_stamp = {tau_stamp*1000:+.2f} ms")
    print(f"  tau_res   = {tau_res*1000:+.2f} ms")
    print(f"  tau_sync  = {tau_sync*1000:+.2f} ms  ← use for association")
    print(f"\n  === DIAGNOSTIC ===")
    print(f"  tau_unconst   = {tau_unconst*1000:+.2f} ms")
    print(f"  alias_suspect = {alias_suspect}  health_ok={health_ok}")
    for m in health_messages:
        print(f"  • {m}")
    print()

    return HybridSyncResult(
        tau_sync=tau_sync,
        tau_stamp=tau_stamp,
        tau_res=tau_res,
        T_CL=T_CL,
        stamp=stamp,
        window_results=window_results,
        tau_unconst=float(tau_unconst),
        alias_suspect=alias_suspect,
        health_ok=health_ok,
        health_messages=health_messages,
        T_CL_init=T_CL_0,
    )


def hybrid_windows_as_calib(result: HybridSyncResult) -> List[WindowCalibResult]:
    return [
        WindowCalibResult(
            window_idx=w.window_idx,
            frame_start=w.frame_start,
            frame_end=w.frame_end,
            tau=w.tau_sync,
            T_CL=result.T_CL.copy(),
            score=w.score_at_sync,
        )
        for w in result.window_results
    ]


run_temporal_calibration = run_hybrid_sync
