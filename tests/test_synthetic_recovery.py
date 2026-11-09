"""
End-to-end synthetic check of the hybrid sync (no bag needed).

Scene: two rectangles moving sinusoidally across the image at different
depths. LiDAR scan stamped ``t_L`` observes the scene at time ``t_L``; the
camera frame stamped ``t_C`` observes it at ``t_C - tau_true``, where

    tau_true = phase + bias

``phase`` is the stamp offset between the streams (recovered by
``stamp_sync_stats``) and ``bias`` is an extra latency the edge residual
must find. The camera runs at 200 Hz so a camera frame lands on
``t_L + tau_true`` exactly: blended-DT scoring peaks on camera frames, so
sub-frame residuals are only resolvable when frames are about one grid
step apart.
"""

import numpy as np
import pytest

from calib.camera import CameraPipeline
from calib.extrinsic_refine import refine_extrinsics
from calib.geometry import euler_to_R, make_T, transform_points
from calib.hybrid_sync import run_hybrid_sync

IMG_W, IMG_H = 320, 240
K = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])

LIDAR_DT = 0.100
CAM_DT = 0.005
N_SCANS = 12
PHASE = 0.002   # stamp offset camera vs LiDAR [s]
BIAS = 0.030    # latency residual to recover [s], < residual band
TAU_TRUE = PHASE + BIAS
MOTION_PERIOD = 0.8

RESIDUAL_BAND = 0.05
RESIDUAL_SAMPLES = 41
GRID_STEP = 2 * RESIDUAL_BAND / (RESIDUAL_SAMPLES - 1)

# LiDAR (x fwd, y left, z up) -> camera (x right, y down, z fwd), plus a small
# rotation and lever arm so T_CL is not trivial.
_R_BASE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
T_CL_TRUE = make_T(euler_to_R(0.01, -0.02, 0.015) @ _R_BASE, [0.05, -0.10, 0.02])


def _rects_cam(s):
    """Rectangles (center, half-size, depth) in camera frame at scene time s."""
    w = 2 * np.pi / MOTION_PERIOD
    return [
        # (cx, cy, Z, half_w, half_h)
        (1.2 * np.sin(w * s), -0.4, 4.0, 0.5, 0.3),
        (1.5 * np.cos(w * s), 0.8, 6.0, 0.6, 0.4),
    ]


def _corners(cx, cy, Z, hw, hh):
    return np.array(
        [[cx - hw, cy - hh, Z], [cx + hw, cy - hh, Z], [cx + hw, cy + hh, Z], [cx - hw, cy + hh, Z]]
    )


def _render(s):
    import cv2

    img = np.full((IMG_H, IMG_W), 30, dtype=np.uint8)
    for r in _rects_cam(s):
        c = _corners(*r)
        uv = (K @ (c / c[:, 2:3]).T).T[:, :2]
        cv2.fillPoly(img, [np.round(uv).astype(np.int32)], 200)
    return img


def _boundary_points_cam(s, n_per_side=80):
    pts = []
    for r in _rects_cam(s):
        c = _corners(*r)
        for a, b in zip(c, np.roll(c, -1, axis=0)):
            lam = np.linspace(0.0, 1.0, n_per_side, endpoint=False)[:, None]
            pts.append(a + lam * (b - a))
    return np.vstack(pts)


@pytest.fixture(scope="module")
def scene():
    t_lidar = np.arange(N_SCANS) * LIDAR_DT
    T_LC = np.linalg.inv(T_CL_TRUE)
    scans = [
        {"timestamp": float(t), "points": transform_points(_boundary_points_cam(t), T_LC)}
        for t in t_lidar
    ]

    n_cam = int(round((t_lidar[-1] + 0.12) / CAM_DT))
    cam_times = -0.06 + PHASE + np.arange(n_cam) * CAM_DT
    pipeline = CameraPipeline(canny_low=50, canny_high=150, sigma_px=2.0)
    for k, t_c in enumerate(cam_times):
        pipeline.process_frame(k, _render(t_c - TAU_TRUE))
    pipeline.images.clear()  # keep only DTs (memory)
    pipeline.edge_maps.clear()
    frame_ids = np.arange(n_cam)
    return pipeline, scans, cam_times, frame_ids


def test_hybrid_sync_recovers_phase_and_bias(scene):
    pipeline, scans, cam_times, frame_ids = scene
    res = run_hybrid_sync(
        pipeline, scans, cam_times, frame_ids, K, T_CL_TRUE,
        window_size=4,
        max_points_per_scan=2000,
        residual_band=RESIDUAL_BAND,
        residual_samples=RESIDUAL_SAMPLES,
        unconst_half_width=0.1,
        unconst_samples=21,
        imu=None,
        imu_measurements=None,
        refine_extrinsics_flag=False,
    )
    assert res.tau_stamp == pytest.approx(PHASE, abs=1e-6)
    assert abs(res.tau_res - BIAS) <= GRID_STEP + 1e-9
    assert abs(res.tau_sync - TAU_TRUE) <= GRID_STEP + 1e-9
    assert len(res.window_results) == N_SCANS // 4
    assert not res.alias_suspect


def test_refine_extrinsics_recovers_2deg_single_axis(scene):
    pipeline, scans, cam_times, frame_ids = scene
    delta = np.deg2rad(2.0)
    # Yaw about LiDAR z (up) shifts projections horizontally.
    T_init = T_CL_TRUE @ make_T(euler_to_R(0.0, 0.0, delta), np.zeros(3))
    T_ref, score = refine_extrinsics(
        pipeline, scans, cam_times, frame_ids, K, T_init, TAU_TRUE,
        np.arange(len(scans)),
        rot_bound_deg=2.0, rot_steps=5, trans_bound_m=0.05, trans_steps=3,
    )
    R_err = T_ref[:3, :3] @ T_CL_TRUE[:3, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
    init_err = T_init[:3, :3] @ T_CL_TRUE[:3, :3].T
    ang_init = np.degrees(np.arccos(np.clip((np.trace(init_err) - 1) / 2, -1, 1)))
    assert ang_init == pytest.approx(2.0, abs=1e-6)
    assert ang < 0.5
    assert np.linalg.norm(T_ref[:3, 3] - T_CL_TRUE[:3, 3]) < 0.02
    assert score > 0
