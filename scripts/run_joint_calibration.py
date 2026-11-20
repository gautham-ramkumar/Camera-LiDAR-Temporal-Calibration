#!/usr/bin/env python3
"""Joint (R, t, Δt[, distortion]) camera–LiDAR calibration on huntington.mcap.

Replaces the two-stage "stamp prior → residual search → extrinsic refine"
pipeline with one optimiser over rotation, translation, time offset, and
optional Brown–Conrady lens distortion, with per-point LiDAR deskew.

    python scripts/run_joint_calibration.py --mcap huntington.mcap
    python scripts/run_joint_calibration.py --dist-model k1_k2 --out outputs/joint_distortion

Time convention (same sign as the old ``tau``): t_C = t_L + Δt, i.e. an image
stamped s_C was exposed at LiDAR time s_C − Δt.
"""

from __future__ import annotations

import os

# One BLAS / OpenCV thread per process; the script parallelises over starts with processes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calib import joint_calib as jc  # noqa: E402
from calib.bag_cache import load_cache, valid_point_times  # noqa: E402
from calib.joint_data import prepare_frames  # noqa: E402
from calib.lidar_odometry import run_lidar_odometry  # noqa: E402
from calib.trajectory import Trajectory  # noqa: E402

# Notebook intrinsics (the bag's camera_info is all zeros); distortion fitted or zero.
K = np.array([[1501.9374712879626, 0.0, 566.5690420612353],
              [0.0, 1498.8879775647906, 537.1294320963829],
              [0.0, 0.0, 1.0]])
T_CL_INIT = np.array([[0.99919851, 0.04002921, 0.0, 0.15],
                      [0.0, 0.0, -1.0, -0.2815789473684212],
                      [-0.04002921, 0.99919851, 0.0, -0.13157894736842124],
                      [0.0, 0.0, 0.0, 1.0]])
# Old two-stage result (run_hybrid_sync on main c3988b5: tau_sync = tau_stamp,
# stiff T_e refine with tau fixed). Values from that run's metrics.json.
TAU_STAMP = -0.01949334144592285
T_CL_TWO_STAGE = np.array([[0.9971928259827496, 0.07487635039170995, 0.0, 0.19985964129913747],
                           [0.0, 0.0, -1.0, -0.2815789473684212],
                           [-0.07487635039170995, 0.9971928259827495, 0.0, -0.13532276488800674],
                           [0.0, 0.0, 0.0, 1.0]])

G: dict = {}  # shared with forked workers


# ----------------------------------------------------------------------------- workers
def _solve_start(x0):
    return jc.solve_joint(G["P"], x0)


def _solve_boot(w):
    return jc.solve_joint(G["P"], G["x_hat"], scales=(5.0,), frame_weights=w, maxfev=3000).x


def _solve_fixed_dt(dt):
    P = G["P"]
    x0 = G["x_hat"].copy()
    x0[6] = dt
    lo, hi = jc.problem_bounds(P.dist_model)
    lo, hi = lo.copy(), hi.copy()
    lo[6], hi[6] = -0.5, 0.5
    fix = [False] * len(x0)
    fix[6] = True
    s = jc.solve_joint(P, x0, bounds=(lo, hi), scales=(5.0,), fix_mask=fix, maxfev=3000)
    return s.x, s.cost


def _solve_synth(x0):
    return jc.solve_joint(G["PS"], x0, maxfev=6000).x


def _solve_ablation_task(task):
    dt0, model, dist_prior = task
    P = jc.JointProblem(G["train"], G["traj"], K, T_CL_INIT, dist_model=model, dist_prior=dist_prior)
    x0 = jc.pose_to_params(T_CL_INIT, dt0, T_CL_INIT)
    s = jc.solve_joint(P, x0, maxfev=5000)
    # Report pure data cost (without prior) for fair model comparison
    P.dist_prior = None
    P.cauchy_px = 5.0
    return model, bool(dist_prior is not None), P.cost(s.x), s.x, s.at_bound


# ----------------------------------------------------------------------------- helpers
def make_dist_prior(dist_model: str, k2_sigma: float, tan_sigma: float):
    if dist_model == "none" or dist_model == "k1":
        return None
    if dist_model == "k1_k2":
        if k2_sigma <= 0:
            return None
        return ([0.0, 0.0, 0.0, 0.0], [None, k2_sigma, None, None])
    if dist_model == "k1_k2_p1_p2":
        s_k2 = k2_sigma if k2_sigma > 0 else None
        s_t = tan_sigma if tan_sigma > 0 else None
        return ([0.0, 0.0, 0.0, 0.0], [None, s_k2, s_t, s_t])
    return None


def state_summary(x, dist_model: str = "none", T0=T_CL_INIT):
    R, t, dt = jc.params_to_pose(x, T0)
    d4 = jc.params_to_dist4(x, dist_model)
    return {
        "rpy_deg": jc.rpy_deg(R).tolist(),
        "t_m": t.tolist(),
        "dt_ms": dt * 1e3,
        "T_CL": jc.make_T(R, t).tolist(),
        "dist_k1_k2_p1_p2": d4.tolist(),
    }


def odometry(cache, path: Path) -> Trajectory:
    if path.is_file():
        d = np.load(path)
        return Trajectory(d["times"], d["rotations"], d["positions"])

    def load(k):
        s = cache.scan(k)
        xyz, t, rng = s["xyz"].reshape(-1, 3), s["t_ns"].reshape(-1), s["range"].reshape(-1)
        ok = valid_point_times(t) & np.isfinite(xyz).all(1) & (rng > 0)
        return xyz[ok][::4], t[ok][::4] * 1e-9, float(cache.lidar_t[k])

    traj, info = run_lidar_odometry(len(cache.lidar_t), load)
    np.savez(path, times=traj.times, rotations=traj.rotations, positions=traj.positions,
             inliers=info["inliers"])
    return traj


def ego_motion_check(cache, traj):
    ix = cache.index
    tk = traj.times[:-1] + 1e-3
    w, v = traj.velocity_at(tk)
    sp = np.linalg.norm(v, axis=1)
    gt, gs = ix["gps_t"], ix["gps_speed"]
    m = (gt > traj.times[1]) & (gt < traj.times[-2])
    osp = np.interp(gt[m], tk, sp)
    return {
        "odom_speed_median_mps": float(np.median(sp)),
        "odom_speed_p95_mps": float(np.percentile(sp, 95)),
        "odom_yaw_rate_median_dps": float(np.degrees(np.median(np.linalg.norm(w, axis=1)))),
        "odom_yaw_rate_p95_dps": float(np.degrees(np.percentile(np.linalg.norm(w, axis=1), 95))),
        "gps_vs_odom_speed_corr": float(np.corrcoef(gs[m], osp)[0, 1]),
        "gps_vs_odom_speed_median_ratio": float(np.median(osp / np.maximum(gs[m], 0.1))),
    }, (gt[m], gs[m], osp)


def render_synthetic(frames, traj, x_true, T0, dist_model="none", radius=1):
    """Replace each frame's DT with one rendered from its own LiDAR edges at a known state."""
    R, t, dt = jc.params_to_pose(x_true, T0)
    d4 = jc.params_to_dist4(x_true, dist_model)
    P = jc.JointProblem(frames, traj, K, T0, dist_model=dist_model)
    out = []
    for f in frames:
        u, v, good = P.project(f, R, t, dt, dist=d4)
        h, w = f.dt_img.shape
        img = np.full((h, w), 255, np.uint8)
        for a, b in zip(u[good], v[good]):
            if 0 <= a < w and 0 <= b < h:
                cv2.circle(img, (int(round(a)), int(round(b))), radius, 0, -1)
        dtimg = np.minimum(cv2.distanceTransform(img, cv2.DIST_L2, 5), 40.0).astype(np.float32)
        out.append(jc.JointFrame(f.scan_idx, f.cam_idx, f.cam_stamp, f.lidar_stamp, f.q_point, f.q_scan,
                                 f.t_points, dtimg, f.speed, f.ang_rate, f.block))
    return out


def overlay(cache, P, f, x, label, dist_override=None):
    R, t, dt = jc.params_to_pose(x, P.T0)
    d4 = jc.params_to_dist4(x, P.dist_model, dist_override) if dist_override is None else np.asarray(dist_override)
    img = cv2.cvtColor(cache.image(f.cam_idx), cv2.COLOR_GRAY2BGR)
    img = (img * 0.75).astype(np.uint8)
    u, v, good = P.project(f, R, t, dt, dist=d4)
    d, inside = P.frame_distances(f, R, t, dt, dist=d4)
    for a, b, dd, ok in zip(u, v, d, inside):
        if ok:
            col = (0, 255, 0) if dd < 2 else ((0, 255, 255) if dd < 5 else (0, 0, 255))
            cv2.circle(img, (int(a), int(b)), 3, col, -1)
    med = float(np.median(d[inside])) if inside.any() else float("nan")
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, f"{label}  dt={dt*1e3:+.1f} ms  median {med:.2f} px", (8, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return img


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcap", default=str(ROOT / "huntington.mcap"))
    ap.add_argument("--cache", default=str(ROOT / ".calib_cache" / "joint_v1"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "joint"))
    ap.add_argument("--dist-model", default="none", choices=tuple(jc.DIST_MODELS))
    ap.add_argument("--k2-prior", type=float, default=0.3, help="Gaussian prior sigma on k2 (<=0 disables)")
    ap.add_argument("--tan-prior", type=float, default=0.01, help="Gaussian prior sigma on p1,p2 (<=0 disables)")
    ap.add_argument("--margin-px", type=float, default=None,
                    help="border margin in prepare_frames (defaults to 20 with distortion, -20 pinhole)")
    ap.add_argument("--stride", type=int, default=4, help="use every Nth LiDAR scan")
    ap.add_argument("--train-end", type=float, default=50.0, help="train block ends (s from bag start)")
    ap.add_argument("--test-start", type=float, default=52.0, help="held-out block starts (s)")
    ap.add_argument("--bootstrap", type=int, default=30)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    margin_px = args.margin_px if args.margin_px is not None else (20.0 if args.dist_model != "none" else -20.0)
    dist_prior = make_dist_prior(args.dist_model, args.k2_prior, args.tan_prior)

    cache = load_cache(args.mcap, args.cache)
    t0_bag = float(cache.lidar_t[0])
    traj = odometry(cache, Path(args.cache) / "odometry.npz")
    ego, (gps_t, gps_s, odo_s) = ego_motion_check(cache, traj)
    print("ego-motion:", json.dumps(ego, indent=1))

    bad_scans = {}
    for k in range(len(cache.lidar_t)):
        s = cache.scan(k)
        fin = np.isfinite(s["xyz"]).all(2) & (s["range"] > 0)
        nb = int((fin & ~valid_point_times(s["t_ns"])).sum())
        if nb:
            bad_scans[k] = {"t_s": float(cache.lidar_t[k] - t0_bag), "n_bad_points": nb,
                            "max_t_s": float(s["t_ns"][fin].max() * 1e-9)}
    print("bad per-point t:", bad_scans)

    idx = np.arange(2, len(cache.lidar_t) - 2, args.stride)
    rel = cache.lidar_t[idx] - t0_bag
    keep = (rel < args.train_end) | (rel >= args.test_start)
    idx = idx[keep]
    blocks = ["train" if cache.lidar_t[k] - t0_bag < args.train_end else "test" for k in idx]
    frames = prepare_frames(cache, traj, idx, K, T_CL_INIT, TAU_STAMP, blocks=blocks, margin_px=margin_px)
    train = [f for f in frames if f.block == "train"]
    test = [f for f in frames if f.block == "test"]
    test_rate = np.array([np.degrees(f.ang_rate) for f in test])
    fast = [i for i, r in enumerate(test_rate) if r >= 8]

    P = jc.JointProblem(train, traj, K, T_CL_INIT, dist_model=args.dist_model, dist_prior=dist_prior)
    Pt = jc.JointProblem(test, traj, K, T_CL_INIT, dist_model=args.dist_model)
    param_names = jc.problem_param_names(args.dist_model)
    d_names = jc.dist_param_names(args.dist_model)
    print(f"dist_model={args.dist_model} margin={margin_px:+.0f}px frames: train {len(train)} ({P.n_points} pts), test {len(test)} ({Pt.n_points} pts)", flush=True)
    G["P"] = P
    G["train"] = train
    G["traj"] = traj
    ctx = mp.get_context("fork")

    # ---------------------------------------------------------------- optional model ablation (when distortion enabled)
    ablation = None
    pinhole_x = None
    if args.dist_model != "none":
        t1 = time.time()
        specs = [
            ("none", None),
            ("k1", None),
            ("k1_k2", None),
            ("k1_k2", make_dist_prior("k1_k2", args.k2_prior, args.tan_prior)),
            ("k1_k2_p1_p2", make_dist_prior("k1_k2_p1_p2", args.k2_prior, args.tan_prior)),
        ]
        tasks = [(dt0, m, pr) for m, pr in specs for dt0 in (TAU_STAMP, 0.0, 0.025)]
        with ctx.Pool(args.workers) as pool:
            res_abl = pool.map(_solve_ablation_task, tasks)
        ablation = []
        for idx_s, (m, pr) in enumerate(specs):
            sub = res_abl[3 * idx_s:3 * (idx_s + 1)]
            _, has_pr, c5, x_m, ab_m = min(sub, key=lambda o: o[2])
            Pm = jc.JointProblem(train, traj, K, T_CL_INIT, dist_model=m)
            Ptm = jc.JointProblem(test, traj, K, T_CL_INIT, dist_model=m)
            if m == "none":
                pinhole_x = x_m
            ablation.append({
                "model": m,
                "regularised": has_pr,
                "data_cost": float(c5),
                **state_summary(x_m, m),
                "at_bound": [k for k, v in ab_m.items() if v],
                "train_median_px": Pm.median_distance(x_m),
                "test_median_px": Ptm.median_distance(x_m),
                "test_fast_frames_median_px": Ptm.median_distance(x_m, fast) if fast else None,
            })
        print(f"distortion model comparison ({len(specs)} models) in {time.time()-t1:.0f}s", flush=True)

    # ---------------------------------------------------------------- multi-start joint solve
    starts = []
    for dt0 in (-0.060, -0.040, TAU_STAMP, 0.0, 0.020, 0.040, 0.060):
        x = jc.pose_to_params(T_CL_INIT, dt0, T_CL_INIT)
        starts.append(np.r_[x, np.zeros(len(d_names))] if d_names else x)
    rs = np.random.default_rng(1)
    for dt0 in (TAU_STAMP, 0.0, 0.030):
        for _ in range(2):
            x = jc.pose_to_params(T_CL_INIT, dt0, T_CL_INIT)
            x[:3] += np.radians(rs.uniform(-1.5, 1.5, 3))
            x[3:6] += rs.uniform(-0.08, 0.08, 3)
            if d_names:
                d0 = np.zeros(len(d_names))
                d0[0] = rs.uniform(-0.35, 0.0)
                x = np.r_[x, d0]
            starts.append(x)
    t1 = time.time()
    with ctx.Pool(args.workers) as pool:
        sols = pool.map(_solve_start, starts)
    costs = np.array([s.cost for s in sols])
    best = sols[int(np.argmin(costs))]
    x_hat = best.x
    G["x_hat"] = x_hat
    print(f"multi-start ({args.dist_model}): {len(starts)} starts in {time.time()-t1:.0f}s, best cost {costs.min():.1f}", flush=True)
    R_hat, t_hat, dt_hat = jc.params_to_pose(x_hat, T_CL_INIT)
    d4_hat = jc.params_to_dist4(x_hat, args.dist_model)
    near = [s for s in sols if s.cost <= costs.min() * 1.005]
    rep = {
        "n_starts": len(sols),
        "n_within_0p5pct_of_best_cost": len(near),
        "max_rot_diff_deg": max(jc.rotation_angle_deg(R_hat, jc.params_to_pose(s.x, T_CL_INIT)[0]) for s in near),
        "max_trans_diff_mm": max(1e3 * np.linalg.norm(jc.params_to_pose(s.x, T_CL_INIT)[1] - t_hat) for s in near),
        "max_dt_diff_ms": max(1e3 * abs(s.x[6] - dt_hat) for s in near),
        "max_k1_diff": max(abs(jc.params_to_dist4(s.x, args.dist_model)[0] - d4_hat[0]) for s in near) if d_names else 0.0,
        "all_starts": [{"start_dt_ms": 1e3 * s.start[6], "cost": s.cost, **state_summary(s.x, args.dist_model),
                        "at_bound": [k for k, v in s.at_bound.items() if v]} for s in sols],
    }

    # ---------------------------------------------------------------- uncertainty
    cov = jc.gauss_newton_covariance(P, x_hat)
    corr = jc.correlation_from_cov(cov)
    rng = np.random.default_rng(7)
    weights = [np.bincount(rng.integers(0, len(train), len(train)), minlength=len(train)).astype(float)
               for _ in range(args.bootstrap)]
    t1 = time.time()
    with ctx.Pool(args.workers) as pool:
        boots = np.array(pool.map(_solve_boot, weights))
    print(f"bootstrap {args.bootstrap} in {time.time()-t1:.0f}s", flush=True)
    boot_rot = np.array([Rotation.from_matrix(jc.params_to_pose(b, T_CL_INIT)[0] @ R_hat.T).as_rotvec()
                         for b in boots])
    boot_vec = np.hstack([np.degrees(boot_rot), boots[:, 3:6] - x_hat[3:6], boots[:, 6:] - x_hat[6:]])
    boot_std = boot_vec.std(axis=0, ddof=1)
    boot_corr = np.corrcoef(boot_vec.T)
    gn_std = np.sqrt(np.diag(cov))

    # ---------------------------------------------------------------- Δt profiles
    grid_fixed = np.arange(-0.250, 0.2501, 0.005)
    prof_fixed = {}
    for name, prob in (("train", P), ("test", Pt)):
        prob.cauchy_px = 5.0
        old_pr = prob.dist_prior
        prob.dist_prior = None
        vals = []
        for d in grid_fixed:
            x = x_hat.copy()
            x[6] = d
            vals.append(prob.cost(x) / prob.n_points)
        prob.dist_prior = old_pr
        prof_fixed[name] = np.array(vals)
    x_old = jc.pose_to_params(T_CL_TWO_STAGE, 0.0, T_CL_INIT)
    P_pin = jc.JointProblem(train, traj, K, T_CL_INIT)
    prof_fixed["train_two_stage_pose"] = np.array(
        [P_pin.cost(np.r_[x_old[:6], d]) / P_pin.n_points for d in grid_fixed])

    grid_reopt = np.unique(np.r_[np.arange(-0.200, 0.2001, 0.020), -0.120, TAU_STAMP, dt_hat])
    t1 = time.time()
    with ctx.Pool(args.workers) as pool:
        reopt = pool.map(_solve_fixed_dt, list(grid_reopt))
    print(f"re-optimised Δt profile ({len(grid_reopt)} pts) in {time.time()-t1:.0f}s", flush=True)
    reopt_x = np.array([r[0] for r in reopt])
    reopt_cost = np.array([r[1] for r in reopt]) / P.n_points

    def summarize_reopt(i):
        R, t, d = jc.params_to_pose(reopt_x[i], T_CL_INIT)
        return {"dt_ms": d * 1e3, "cost_per_pt": float(reopt_cost[i]), "rpy_deg": jc.rpy_deg(R).tolist(),
                "t_m": t.tolist(), "dist": jc.params_to_dist4(reopt_x[i], args.dist_model).tolist(),
                "test_median_px": Pt.median_distance(reopt_x[i])}

    reopt_table = [summarize_reopt(i) for i in range(len(grid_reopt))]
    m = np.abs(grid_reopt - dt_hat) <= 0.0605
    tr_slope = np.polyfit(grid_reopt[m] * 1e3, reopt_x[m, 3:6] * 1e3, 1)[0]
    rot_slope = np.polyfit(grid_reopt[m] * 1e3, np.degrees(reopt_x[m, :3]), 1)[0]

    # ---------------------------------------------------------------- ablation: scan-level time (no per-point deskew)
    Pscan = jc.JointProblem(train, traj, K, T_CL_INIT, per_point=False,
                            dist_model=args.dist_model, dist_prior=dist_prior)
    G["P"] = Pscan
    with ctx.Pool(args.workers) as pool:
        sols_scan = pool.map(_solve_start, [starts[2], starts[3], starts[5]])
    G["P"] = P
    best_scan = min(sols_scan, key=lambda s: s.cost)
    corr_scan = jc.correlation_from_cov(jc.gauss_newton_covariance(Pscan, best_scan.x))
    Pt_scan = jc.JointProblem(test, traj, K, T_CL_INIT, per_point=False, dist_model=args.dist_model)

    # ---------------------------------------------------------------- sensitivity: anchor translation to the hand guess
    Pprior = jc.JointProblem(train, traj, K, T_CL_INIT, t_prior=(T_CL_INIT[:3, 3], 0.05),
                             dist_model=args.dist_model, dist_prior=dist_prior)
    G["P"] = Pprior
    with ctx.Pool(args.workers) as pool:
        sols_prior = pool.map(_solve_start, [starts[2], starts[3], starts[5]])
    G["P"] = P
    best_prior = min(sols_prior, key=lambda s: s.cost)

    # ---------------------------------------------------------------- per-window Δt (R,t,dist fixed at joint)
    allf = sorted(frames, key=lambda f: f.cam_stamp)
    ts = np.array([f.cam_stamp - t0_bag for f in allf])
    edges = np.linspace(ts.min(), ts.max() + 1e-6, 16)
    win = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [f for f, tt in zip(allf, ts) if a <= tt < b]
        if len(sel) < 3:
            continue
        Pw = jc.JointProblem(sel, traj, K, T_CL_INIT, dist_model=args.dist_model)
        Pw.cauchy_px = 5.0

        def win_cost(d):
            xx = x_hat.copy()
            xx[6] = d
            return Pw.cost(xx)

        r = minimize_scalar(win_cost, bounds=(-0.1, 0.1), method="bounded", options={"xatol": 2e-4})
        g = np.arange(-0.1, 0.1001, 0.005)
        c = np.array([win_cost(d) for d in g])
        dg = g[int(np.argmin(c))]
        rr = minimize_scalar(win_cost, bounds=(dg - 0.006, dg + 0.006), method="bounded", options={"xatol": 2e-4})
        dbest = rr.x if rr.fun <= r.fun else r.x
        win.append({"t0_s": float(a), "t1_s": float(b), "n_frames": len(sel), "dt_ms": float(dbest * 1e3),
                    "mean_yaw_rate_dps": float(np.degrees(np.mean([f.ang_rate for f in sel]))),
                    "mean_speed_mps": float(np.mean([f.speed for f in sel]))})
    wdt = np.array([w["dt_ms"] for w in win])
    moving = np.array([w["mean_yaw_rate_dps"] >= 5 or w["mean_speed_mps"] >= 1.0 for w in win])

    # ---------------------------------------------------------------- held-out evaluation
    # Pad 7-vectors with zero distortion when evaluating under Pt (which reads x[7:] if dist_model != none).
    def pad(x):
        return np.r_[x, np.zeros(len(d_names))] if (d_names and len(x) == 7) else x

    cands = {
        "initial guess (stamp dt)": pad(jc.pose_to_params(T_CL_INIT, TAU_STAMP, T_CL_INIT)),
        "old two-stage (stamp dt + Te refine)": pad(jc.pose_to_params(T_CL_TWO_STAGE, TAU_STAMP, T_CL_INIT)),
        "joint (R,t,dt)" if not d_names else f"joint (R,t,dt,{args.dist_model})": x_hat,
        "joint pose, dt forced to stamp": np.r_[x_hat[:6], TAU_STAMP, x_hat[7:]],
    }
    if pinhole_x is not None:
        cands["joint pinhole (no distortion)"] = pad(pinhole_x)
    evaluation = {}
    for name, x in cands.items():
        evaluation[name] = {"train_median_px": P.median_distance(x), "test_median_px": Pt.median_distance(x),
                            "test_median_px_scan_level_model": Pt_scan.median_distance(x),
                            "dt_ms": 1e3 * x[6]}
        evaluation[name]["test_fast_frames_median_px"] = Pt.median_distance(x, fast) if fast else None
    evaluation["_n_test_fast_frames(yaw>=8dps)"] = len(fast)

    # ---------------------------------------------------------------- semi-synthetic recovery
    x_true = x_hat.copy()
    x_true[:3] += np.radians([0.4, -0.3, 0.5])
    x_true[3:6] += np.array([0.03, -0.02, 0.04])
    x_true[6] = -0.025
    synth = render_synthetic(train, traj, x_true, T_CL_INIT, dist_model=args.dist_model)
    rs2 = np.random.default_rng(3)
    half = []
    for f in synth:
        sel = np.sort(rs2.choice(len(f.q_point), max(len(f.q_point) // 2, 1), replace=False))
        half.append(jc.JointFrame(f.scan_idx, f.cam_idx, f.cam_stamp, f.lidar_stamp, f.q_point[sel],
                                  f.q_scan[sel], f.t_points[sel], f.dt_img, f.speed, f.ang_rate))
    G["PS"] = jc.JointProblem(half, traj, K, T_CL_INIT, dist_model=args.dist_model, dist_prior=dist_prior)
    synth_starts = [pad(jc.pose_to_params(T_CL_INIT, d, T_CL_INIT)) for d in (TAU_STAMP, 0.0, 0.040)]
    with ctx.Pool(args.workers) as pool:
        synth_x = pool.map(_solve_synth, synth_starts)
    Rt, tt_, dtt = jc.params_to_pose(x_true, T_CL_INIT)
    dt_dist = jc.params_to_dist4(x_true, args.dist_model)
    synth_res = []
    for s0, xs in zip(synth_starts, synth_x):
        Rs, tsn, dts = jc.params_to_pose(xs, T_CL_INIT)
        de = jc.params_to_dist4(xs, args.dist_model)
        synth_res.append({"start_dt_ms": 1e3 * s0[6], "rot_err_deg": jc.rotation_angle_deg(Rt, Rs),
                          "trans_err_mm": 1e3 * float(np.linalg.norm(tsn - tt_)),
                          "trans_err_xyz_mm": (1e3 * (tsn - tt_)).tolist(),
                          "dt_err_ms": 1e3 * (dts - dtt),
                          "dist_err": (de - dt_dist).tolist()})

    # ---------------------------------------------------------------- write results
    R0 = T_CL_INIT[:3, :3]
    other_names = [n for n in param_names if n != "dt"]
    non_dt_idx = [i for i, n in enumerate(param_names) if n != "dt"]
    p_names_unc = ["rot_x_deg", "rot_y_deg", "rot_z_deg", "tx_m", "ty_m", "tz_m", "dt_s", *d_names]
    result = {
        "convention": "t_C = t_L + dt (same sign as the old tau); image stamped s_C is exposed at LiDAR time s_C - dt",
        "dist_model": args.dist_model,
        "dist_prior": {"k2_sigma": args.k2_prior, "tan_sigma": args.tan_prior} if dist_prior else None,
        "margin_px": margin_px,
        "joint": state_summary(x_hat, args.dist_model),
        "joint_at_bound": [k for k, v in best.at_bound.items() if v],
        "bounds": {"rot_deg": 5, "trans_m": 0.40, "dt_ms": [-60, 60],
                   **{n: list(jc.DIST_BOUNDS[n]) for n in d_names}},
        "change_from_initial": {
            "rotation_deg": jc.rotation_angle_deg(R0, R_hat),
            "translation_m": float(np.linalg.norm(t_hat - T_CL_INIT[:3, 3])),
            "translation_delta_m": (t_hat - T_CL_INIT[:3, 3]).tolist(),
            "dt_vs_stamp_ms": 1e3 * (dt_hat - TAU_STAMP),
        },
        "uncertainty_1sigma": {
            "params": p_names_unc,
            "gauss_newton": [*(np.degrees(gn_std[:3])).tolist(), *gn_std[3:].tolist()],
            "frame_bootstrap": boot_std.tolist(),
            "n_bootstrap": args.bootstrap,
        },
        "correlation_dt_vs": {
            "names": other_names,
            "gauss_newton_per_point_deskew": corr[6, non_dt_idx].tolist(),
            "bootstrap_per_point_deskew": boot_corr[6, non_dt_idx].tolist(),
            "gauss_newton_scan_level_time": corr_scan[6, non_dt_idx].tolist(),
        },
        "pose_shift_per_ms_of_dt": {"translation_mm_per_ms": tr_slope.tolist(),
                                    "rotation_deg_per_ms": rot_slope.tolist(),
                                    "fit_range_ms": [1e3 * (dt_hat - 0.06), 1e3 * (dt_hat + 0.06)]},
        "repeatability": rep,
        "dt_profile_fixed_pose": {
            "grid_ms": (grid_fixed * 1e3).tolist(),
            "train_cost_per_pt": prof_fixed["train"].tolist(),
            "test_cost_per_pt": prof_fixed["test"].tolist(),
            "argmin_train_ms": float(grid_fixed[np.argmin(prof_fixed["train"])] * 1e3),
            "argmin_test_ms": float(grid_fixed[np.argmin(prof_fixed["test"])] * 1e3),
            "cost_at_minus120_vs_min_train": float(prof_fixed["train"][np.argmin(np.abs(grid_fixed + 0.12))]
                                                   - prof_fixed["train"].min()),
        },
        "dt_profile_reoptimised_pose": reopt_table,
        "scan_level_ablation": {**state_summary(best_scan.x, args.dist_model), "cost": best_scan.cost,
                                "train_median_px": Pscan.median_distance(best_scan.x),
                                "test_median_px": Pt_scan.median_distance(best_scan.x)},
        "sensitivity_translation_prior_5cm": {**state_summary(best_prior.x, args.dist_model),
                                              "train_median_px": P.median_distance(best_prior.x),
                                              "test_median_px": Pt.median_distance(best_prior.x),
                                              "dt_of_each_start_ms": [1e3 * s.x[6] for s in sols_prior]},
        "per_window_dt": {"windows": win, "spread_all_ms": float(np.ptp(wdt)),
                          "std_all_ms": float(np.std(wdt)),
                          "spread_moving_ms": float(np.ptp(wdt[moving])) if moving.sum() > 1 else None,
                          "std_moving_ms": float(np.std(wdt[moving])) if moving.sum() > 1 else None,
                          "old_two_stage_window_spread_ms": 43.76},
        "evaluation_median_edge_distance_px": evaluation,
        "distortion_model_comparison": ablation,
        "semi_synthetic": {"truth": state_summary(x_true, args.dist_model), "recoveries": synth_res},
        "ego_motion": ego,
        "bad_point_times": bad_scans,
        "frames": {"train": len(train), "test": len(test), "train_points": P.n_points,
                   "test_points": Pt.n_points, "stride": args.stride,
                   "train_block_s": [0, args.train_end], "test_block_s": [args.test_start, float(ts.max())],
                   "train_speed_mps_p10_p50_p90": np.percentile([f.speed for f in train], [10, 50, 90]).tolist(),
                   "train_yaw_rate_dps_p10_p50_p90": np.degrees(np.percentile([f.ang_rate for f in train],
                                                                             [10, 50, 90])).tolist()},
        "runtime_s": time.time() - t_start,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    (out / "T_CL_joint.json").write_text(json.dumps({**state_summary(x_hat, args.dist_model),
                                                     "convention": result["convention"],
                                                     "dt_bootstrap_1sigma_ms": 1e3 * boot_std[6]}, indent=1))

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    ax[0].plot(grid_fixed * 1e3, prof_fixed["train"], label="train (pose fixed at joint)")
    ax[0].plot(grid_fixed * 1e3, prof_fixed["test"], label="held-out (pose fixed at joint)")
    ax[0].plot(grid_fixed * 1e3, prof_fixed["train_two_stage_pose"], ":", label="train (old two-stage pose)")
    ax[1].plot(grid_reopt * 1e3, reopt_cost, "o-", color="k", label="train, R,t re-optimised at each dt")
    for a in ax:
        a.axvline(dt_hat * 1e3, color="g", ls="--", label=f"joint {dt_hat*1e3:+.1f} ms")
        a.axvline(TAU_STAMP * 1e3, color="r", ls=":", label="stamp -19.5 ms")
        a.axvline(-120, color="m", ls="-.", label="old alias -120 ms")
        a.axvspan(-60, 60, color="orange", alpha=0.08)
        a.set_xlabel("dt [ms]  (t_C = t_L + dt)")
        a.set_ylabel("robust cost per point")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_title(f"Δt profile, pose fixed ({args.dist_model})")
    ax[1].set_title(f"Δt profile, R,t re-optimised ({args.dist_model})")
    fig.tight_layout()
    fig.savefig(out / "dt_profile.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, n in enumerate(["tx", "ty", "tz"]):
        ax[0].plot(grid_reopt * 1e3, reopt_x[:, 3 + i] * 100 + T_CL_INIT[i, 3] * 100, "o-", label=n)
    ax[0].set_ylabel("translation [cm]")
    for i, n in enumerate(["rot_x", "rot_y", "rot_z"]):
        ax[1].plot(grid_reopt * 1e3, np.degrees(reopt_x[:, i]), "o-", label=n)
    ax[1].set_ylabel("rotation change from initial [deg]")
    for a in ax[:2]:
        a.axvline(dt_hat * 1e3, color="g", ls="--")
        a.axvline(TAU_STAMP * 1e3, color="r", ls=":")
        a.set_xlabel("dt held fixed [ms]")
        a.legend()
        a.grid(alpha=0.3)
    ax[0].set_title("Translation the fit needs at each fixed dt")
    ax[1].set_title("Rotation the fit needs at each fixed dt")
    xx = np.arange(len(other_names))
    ax[2].bar(xx - 0.27, corr[6, non_dt_idx], 0.27, label="joint, per-point deskew (GN)")
    ax[2].bar(xx, boot_corr[6, non_dt_idx], 0.27, label="joint, per-point deskew (bootstrap)")
    ax[2].bar(xx + 0.27, corr_scan[6, non_dt_idx], 0.27, label="scan-level time (GN)")
    ax[2].set_xticks(xx, other_names)
    ax[2].set_ylim(-1, 1)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("corr(dt, ·)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "confound.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(1e3 * boots[:, 6], bins=15)
    ax[0].set_xlabel("dt [ms]")
    ax[0].set_title(f"bootstrap dt (n={args.bootstrap})")
    ax[1].scatter(1e3 * boots[:, 6], 100 * (boots[:, 5] + T_CL_INIT[2, 3]))
    ax[1].set_xlabel("dt [ms]")
    ax[1].set_ylabel("tz [cm]")
    ax[1].set_title("bootstrap: dt vs tz")
    if d_names:
        ax[2].scatter(1e3 * boots[:, 6], boots[:, 7])
        ax[2].set_ylabel("k1")
        ax[2].set_title("bootstrap: dt vs k1")
    else:
        ax[2].scatter(1e3 * boots[:, 6], boot_vec[:, 2])
        ax[2].set_ylabel("Δrot_z [deg]")
        ax[2].set_title("bootstrap: dt vs rot_z")
    ax[2].set_xlabel("dt [ms]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "bootstrap.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    mids = [(w["t0_s"] + w["t1_s"]) / 2 for w in win]
    ax[0].plot(mids, wdt, "o-", label="per-window dt (pose fixed at joint)")
    ax[0].plot(np.array(mids)[~moving], wdt[~moving], "x", color="r", ms=10, label="low-motion window")
    ax[0].axhline(dt_hat * 1e3, color="g", ls="--", label="joint")
    ax[0].axhline(TAU_STAMP * 1e3, color="r", ls=":", label="stamp")
    ax[0].set_xlabel("bag time [s]")
    ax[0].set_ylabel("dt [ms]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(gps_t - t0_bag, gps_s, label="u-blox GPS speed")
    ax[1].plot(gps_t - t0_bag, odo_s, label="LiDAR odometry speed")
    ax[1].set_xlabel("bag time [s]")
    ax[1].set_ylabel("m/s")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[1].set_title("ego-motion check used for deskew")
    fig.tight_layout()
    fig.savefig(out / "per_window_and_ego_motion.png", dpi=120)
    plt.close(fig)

    if d_names:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        r_px = np.linspace(0, 760, 200)
        r_norm = r_px / float(K[0, 0])
        r2 = r_norm ** 2
        dr = r_px * (d4_hat[0] * r2 + d4_hat[1] * (r2 ** 2))
        ax[0].plot(r_px, dr, lw=2, label=f"k1={d4_hat[0]:+.3f}, k2={d4_hat[1]:+.3f}")
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].set_xlabel("pinhole radius from principal point [px]")
        ax[0].set_ylabel("radial distortion shift Δr [px]")
        ax[0].set_title("Fitted radial distortion curve")
        ax[0].grid(alpha=0.3)
        ax[0].legend()
        if ablation:
            labels_m = [f"{a['model']}\n({'reg' if a['regularised'] else 'free'})" for a in ablation]
            x_m = np.arange(len(ablation))
            ax[1].bar(x_m - 0.18, [a["test_median_px"] for a in ablation], 0.36, label="held-out all")
            ax[1].bar(x_m + 0.18, [a["test_fast_frames_median_px"] for a in ablation], 0.36, label="held-out fast")
            ax[1].set_xticks(x_m, labels_m, fontsize=8)
            ax[1].set_ylabel("median edge distance [px]")
            ax[1].set_title("Distortion model comparison (held-out)")
            ax[1].legend(fontsize=8)
            ax[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "distortion_effect.png", dpi=120)
        plt.close(fig)

    # overlays: fastest-turning and a few spread frames
    allp = jc.JointProblem(frames, traj, K, T_CL_INIT, dist_model=args.dist_model)
    order = np.argsort([-f.ang_rate for f in frames])
    picks = list(order[:3]) + list(np.linspace(0, len(frames) - 1, 4).astype(int))
    panel_specs = [("initial guess (stamp dt)", "initial"), ("old two-stage (stamp dt + Te refine)", "two-stage")]
    if pinhole_x is not None:
        panel_specs.append(("joint pinhole (no distortion)", "joint pinhole"))
    panel_specs.append(("joint (R,t,dt)" if not d_names else f"joint (R,t,dt,{args.dist_model})",
                        f"joint ({args.dist_model})" if d_names else "joint"))
    paths = []
    for n, i in enumerate(dict.fromkeys(int(p) for p in picks)):
        f = frames[i]
        panels = [overlay(cache, allp, f, cands[k], lab) for k, lab in panel_specs]
        scale_ov = 0.45 if len(panels) >= 4 else 0.55
        im = np.hstack([cv2.resize(p, None, fx=scale_ov, fy=scale_ov) for p in panels])
        p = out / "overlays" / (f"overlay_{n:02d}_t{f.cam_stamp - t0_bag:05.1f}s_{f.block}"
                                f"_yaw{np.degrees(f.ang_rate):.0f}dps.png").replace(".", "p", 1)
        cv2.imwrite(str(p), im)
        paths.append(str(p))
    print("overlays:", *paths, sep="\n  ")
    print(json.dumps({k: result[k] for k in ("joint", "change_from_initial", "uncertainty_1sigma",
                                              "correlation_dt_vs", "evaluation_median_edge_distance_px")},
                     indent=1))
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
