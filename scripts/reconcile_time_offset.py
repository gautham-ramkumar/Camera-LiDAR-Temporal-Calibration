#!/usr/bin/env python3
"""Quantitative reconciliation between header-stamp phase (-19.49 ms) and joint Δt.

Verifies:
1. Ouster /ouster/points stamp semantics (header.stamp vs log_time vs per-point t).
2. Azimuth and capture-time window of the camera FOV inside each 100 ms LiDAR sweep.
3. 20 Hz camera vs 10 Hz LiDAR pulse-train aliasing (-19.49 ms modulo 50 ms).
4. Independent camera rotational optical flow vs VectorNav IMU and Ouster IMU lag.
5. Writes ``assets/time_offset_reconciliation.png`` and prints JSON summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mcap.reader import make_reader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calib.bag_cache import BagCache, valid_point_times  # noqa: E402

K = np.array([[1501.9374712879626, 0.0, 566.5690420612353],
              [0.0, 1498.8879775647906, 537.1294320963829],
              [0.0, 0.0, 1.0]])
T0 = np.array([[0.99919851, 0.04002921, 0.0, 0.15],
               [0.0, 0.0, -1.0, -0.2815789473684212],
               [-0.04002921, 0.99919851, 0.0, -0.13157894736842124],
               [0.0, 0.0, 0.0, 1.0]])


def main():
    cache = BagCache(ROOT / ".calib_cache" / "joint_v1")
    ix = cache.index
    cam_t, lid_t = cache.cam_t, cache.lidar_t

    # 1. Header stamp vs log_time from MCAP (first pass over lightweight / non-cloud topics + points header)
    mcap_path = ROOT / "huntington.mcap"
    log_minus_stamp = {t: [] for t in (
        "/ouster/points", "/cam_sync/cam0/image_raw/compressed",
        "/ouster/imu", "/vectornav/imu_uncompensated",
    )}
    ouster_imu_t, ouster_imu_g = [], []
    with mcap_path.open("rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _s, ch, m, msg in r.iter_decoded_messages(topics=list(log_minus_stamp)):
            tp = ch.topic
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            lt = m.log_time * 1e-9
            log_minus_stamp[tp].append((lt - st) * 1e3)
            if tp == "/ouster/imu":
                ouster_imu_t.append(st)
                ouster_imu_g.append((msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))
    ouster_imu_t = np.asarray(ouster_imu_t)
    ouster_imu_gnorm = np.linalg.norm(np.asarray(ouster_imu_g), axis=1)

    # 2. Per-point t semantics and camera-FOV window inside the sweep
    fov_t_ms, fov_az_deg, t_min_ms, t_max_ms = [], [], [], []
    for k in range(0, len(lid_t), 10):
        s = cache.scan(k)
        xyz, t_ns, rng = s["xyz"].reshape(-1, 3), s["t_ns"].reshape(-1), s["range"].reshape(-1)
        ok = valid_point_times(t_ns) & np.isfinite(xyz).all(1) & (rng > 1.5)
        t_ms = t_ns[ok] * 1e-6
        t_min_ms.append(float(t_ms.min()))
        t_max_ms.append(float(t_ms.max()))
        pc = xyz[ok] @ T0[:3, :3].T + T0[:3, 3]
        g = pc[:, 2] > 1.0
        u = K[0, 0] * pc[g, 0] / pc[g, 2] + K[0, 2]
        v = K[1, 1] * pc[g, 1] / pc[g, 2] + K[1, 2]
        in_img = (u >= 0) & (u < 1224) & (v >= 0) & (v < 1024)
        if in_img.any():
            pts_in = xyz[ok][g][in_img]
            fov_t_ms.append(float(np.median(t_ms[g][in_img])))
            fov_az_deg.append(float(np.median(np.degrees(np.arctan2(pts_in[:, 1], pts_in[:, 0])))))
    fov_t_med = float(np.median(fov_t_ms))

    # 3. Nearest camera stamp relative to candidate LiDAR reference times
    def nearest_gaps(offset_s):
        return np.array([cam_t[int(np.argmin(np.abs(cam_t - (tL + offset_s))))] - (tL + offset_s)
                         for tL in lid_t]) * 1e3

    gaps_start = nearest_gaps(0.0)
    gaps_mid = nearest_gaps(0.050)
    gaps_fov = nearest_gaps(fov_t_med * 1e-3)
    gaps_end = nearest_gaps(0.100)

    # 4. Independent camera rotational optical flow vs IMUs
    prev = cache.image(0)
    cam_w_norm = []
    for j in range(1, len(cam_t)):
        curr = cache.image(j)
        mask = np.zeros_like(prev)
        mask[:int(prev.shape[0] * 0.55), :] = 255
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=180, qualityLevel=0.01, minDistance=15, mask=mask)
        if p0 is not None and len(p0) >= 15:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            q0 = cv2.undistortPoints(p0[ok], K, None).reshape(-1, 2)
            q1 = cv2.undistortPoints(p1[ok], K, None).reshape(-1, 2)
            r0 = np.column_stack([q0, np.ones(len(q0))])
            r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
            r1 = np.column_stack([q1, np.ones(len(q1))])
            r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
            R = np.eye(3)
            for _ in range(3):
                U, _S, Vt = np.linalg.svd(r1.T @ r0)
                R = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
                err = np.linalg.norm(r1 - (r0 @ R.T), axis=1)
                keep = err < np.percentile(err, 70)
                if keep.sum() >= 10:
                    r0, r1 = r0[keep], r1[keep]
            cam_w_norm.append(float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()) / (cam_t[j] - cam_t[j - 1])))
        else:
            cam_w_norm.append(0.0)
        prev = curr
    cam_w_norm = np.asarray(cam_w_norm)

    vt, vg = ix["imu_t"], np.linalg.norm(ix["gyro"], axis=1)
    lags = np.arange(-0.060, 0.0601, 0.002)

    def flow_imu_curve(it, ig):
        cs = []
        for lag in lags:
            av = np.array([ig[(it >= a) & (it <= b)].mean() if ((it >= a) & (it <= b)).any() else np.nan
                           for a, b in zip(cam_t[:-1] - lag, cam_t[1:] - lag)])
            ok = np.isfinite(av)
            cs.append(float(np.corrcoef(cam_w_norm[ok], av[ok])[0, 1]))
        return np.asarray(cs)

    c_vn = flow_imu_curve(vt, vg)
    c_ou = flow_imu_curve(ouster_imu_t, ouster_imu_gnorm)

    summary = {
        "log_minus_stamp_ms_median": {k: float(np.median(v)) for k, v in log_minus_stamp.items()},
        "lidar_period_ms_median": float(np.median(np.diff(lid_t) * 1e3)),
        "cam0_period_ms_median": float(np.median(np.diff(cam_t) * 1e3)),
        "per_point_t_range_ms": [float(np.median(t_min_ms)), float(np.median(t_max_ms))],
        "fov_crossing_t_rel_ms": {"median": fov_t_med, "p10": float(np.percentile(fov_t_ms, 10)),
                                  "p90": float(np.percentile(fov_t_ms, 90)),
                                  "median_azimuth_deg": float(np.median(fov_az_deg))},
        "nearest_cam_minus_ref_ms": {
            "scan_start_0ms": float(np.median(gaps_start)),
            "mid_sweep_50ms": float(np.median(gaps_mid)),
            "fov_crossing_73p5ms": float(np.median(gaps_fov)),
            "scan_end_100ms": float(np.median(gaps_end)),
        },
        "cam_flow_vs_imu_dt_ms": {
            "vectornav_best_ms": float(lags[np.argmax(c_vn)] * 1e3),
            "vectornav_peak_corr": float(c_vn.max()),
            "ouster_imu_best_ms": float(lags[np.argmax(c_ou)] * 1e3),
            "ouster_imu_peak_corr": float(c_ou.max()),
        },
    }
    print(json.dumps(summary, indent=2))

    dt_joint_ms = 16.94
    mj = ROOT / "outputs" / "joint_distortion" / "metrics.json"
    if mj.exists():
        dt_joint_ms = float(json.loads(mj.read_text())["joint"]["dt_ms"])

    # Plot timeline + optical-flow cross-check
    fig, ax = plt.subplots(1, 2, figsize=(14.4, 4.8))
    # Left panel: one 100 ms Ouster sweep timeline relative to scan header stamp s_L
    ax[0].axvspan(0, 100, color="#ddeeff", alpha=0.7, label="Ouster 100 ms sweep (t_rel = 0..100 ms)")
    ax[0].axvspan(70, 77, color="#ffe5cc", alpha=0.9, label="Forward camera FOV swept (t_rel ≈ 73.5 ms)")
    ax[0].axvline(0, color="navy", lw=2, label="Ouster header.stamp (scan start)")
    ax[0].axvline(124.8, color="navy", ls=":", lw=1.5, label="Ouster log_time (+124.8 ms)")
    cam_ticks = np.array([-19.49, 30.51, 80.51])
    for i, ct in enumerate(cam_ticks):
        ax[0].plot([ct, ct], [0.10, 0.56], color="crimson", lw=2.5,
                   label="cam0 header.stamp (20 Hz, 50 ms lattice)" if i == 0 else None)
        ax[0].text(ct, 0.59, f"{ct:+.1f} ms", ha="center", va="bottom", fontsize=8.5, color="crimson")
    # Show inferred exposure time for the +80.5 ms frame
    ax[0].annotate("", xy=(80.51 - dt_joint_ms, 0.24), xytext=(80.51, 0.24),
                   arrowprops=dict(arrowstyle="->", color="green", lw=2.2))
    ax[0].text(72.0, 0.19, f"s_C − Δt\n(Δt = {dt_joint_ms:+.1f} ms)",
               ha="center", va="top", fontsize=8.5, color="green", fontweight="bold")
    ax[0].annotate("", xy=(-19.49, 0.36), xytext=(0, 0.36),
                   arrowprops=dict(arrowstyle="<->", color="purple", lw=1.6))
    ax[0].text(-9.7, 0.39, "tau_stamp\n−19.5 ms", ha="center", va="bottom", fontsize=8, color="purple")
    ax[0].set_xlim(-35, 138)
    ax[0].set_ylim(0, 1.15)
    ax[0].set_yticks([])
    ax[0].set_xlabel("time relative to Ouster scan-start header.stamp [ms]")
    ax[0].set_title("One LiDAR sweep vs 20 Hz camera timestamps")
    ax[0].legend(loc="upper center", ncol=2, fontsize=7.3, framealpha=0.92)
    ax[0].grid(alpha=0.25, axis="x")

    # Right panel: independent optical-flow vs IMU lag curve
    ax[1].plot(lags * 1e3, c_vn, "o-", ms=3, label=f"vs VectorNav IMU (peak {lags[np.argmax(c_vn)]*1e3:+.0f} ms)")
    ax[1].plot(lags * 1e3, c_ou, "s-", ms=3, label=f"vs Ouster IMU (peak {lags[np.argmax(c_ou)]*1e3:+.0f} ms)")
    ax[1].axvline(-19.49, color="purple", ls=":", label="stamp phase −19.5 ms")
    ax[1].axvline(dt_joint_ms, color="green", ls="--", label=f"joint + distortion {dt_joint_ms:+.1f} ms")
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set_xlabel("candidate Δt = t_C − t_IMU [ms]")
    ax[1].set_ylabel("angular speed correlation")
    ax[1].set_title("Independent check: camera optical flow vs IMU gyro")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out_png = ROOT / "assets" / "time_offset_reconciliation.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
