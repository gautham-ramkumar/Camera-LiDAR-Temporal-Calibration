"""Tests for the joint (R, t, Δt) calibration with per-point deskew. No bag needed."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calib import joint_calib as jc
from calib.bag_cache import valid_point_times
from calib.trajectory import Trajectory, deskew_points, points_to_world, world_to_lidar_at

K = np.array([[1500.0, 0, 612.0], [0, 1500.0, 512.0], [0, 0, 1]])
T_INIT = np.array([[0.99919851, 0.04002921, 0, 0.15],
                   [0, 0, -1, -0.2815789473684212],
                   [-0.04002921, 0.99919851, 0, -0.13157894736842124],
                   [0, 0, 0, 1]])
H, W = 1024, 1224


def motion(t):
    """Analytic ego-motion: forward along LiDAR +y at ~2 m/s, oscillating yaw (up to ~30 deg/s)."""
    yaw = 0.25 * np.sin(2.2 * t)
    R = Rotation.from_euler("z", yaw).as_matrix()
    p = np.array([0.3 * np.sin(1.1 * t), 2.0 * t + 0.2 * np.sin(1.7 * t), 0.0])
    return R, p


def make_traj(t0=-0.5, t1=3.0, step=0.01):
    ts = np.arange(t0, t1, step)
    Rs, ps = zip(*(motion(t) for t in ts))
    return Trajectory(ts, np.array(Rs), np.array(ps))


def world_structures(seed=0):
    """Vertical poles and horizontal bars in front of the vehicle (world = LiDAR frame at t=0)."""
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(40):
        x, y = rng.uniform(-9, 9), rng.uniform(8, 30)
        segs.append((np.array([x, y, -1.2]), np.array([x, y, rng.uniform(1.5, 4.0)])))
    for _ in range(14):
        x, y, z = rng.uniform(-8, 6), rng.uniform(10, 30), rng.uniform(-0.8, 3.0)
        segs.append((np.array([x, y, z]), np.array([x + rng.uniform(1.5, 4.0), y, z])))
    return segs


def sample(segs, step):
    pts = []
    for a, b in segs:
        n = max(int(np.linalg.norm(b - a) / step), 2)
        pts.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    return np.vstack(pts)


def lidar_capture(traj, Pw, stamp, period=0.1):
    """Ouster-like sweep: azimuth 0 at t=0, rotating clockwise; each point keeps its own time."""
    R, p = traj.pose_at(stamp)
    pl = (Pw - p) @ R
    az = np.arctan2(pl[:, 1], pl[:, 0])
    t_rel = np.mod(-az, 2 * np.pi) / (2 * np.pi) * period
    t_abs = stamp + t_rel
    Rt, pt = traj.pose_at(t_abs)
    pts = np.einsum("nji,nj->ni", Rt, Pw - pt)  # exact point in the LiDAR frame at its own capture time
    return pts, t_abs


def render_dt(traj, Pw_dense, T_CL, tau):
    pL = world_to_lidar_at(traj, Pw_dense, tau)
    pC = pL @ T_CL[:3, :3].T + T_CL[:3, 3]
    ok = pC[:, 2] > 1.0
    u = K[0, 0] * pC[ok, 0] / pC[ok, 2] + K[0, 2]
    v = K[1, 1] * pC[ok, 1] / pC[ok, 2] + K[1, 2]
    img = np.full((H, W), 255, np.uint8)
    for a, b in zip(u, v):
        if 0 <= a < W and 0 <= b < H:
            img[int(b), int(a)] = 0
    return np.minimum(cv2.distanceTransform(img, cv2.DIST_L2, 5), 40.0).astype(np.float32)


def synthetic_frames(traj, T_true, dt_true, n_frames=14, seed=0):
    segs = world_structures(seed)
    Pw_sparse = sample(segs, 0.12)
    Pw_dense = sample(segs, 0.01)
    frames = []
    for k in range(n_frames):
        stamp = 0.1 * k + 0.003
        pts, t_abs = lidar_capture(traj, Pw_sparse, stamp)
        s_C = stamp + 0.0805  # camera stamp near the time the camera sector is swept
        dtimg = render_dt(traj, Pw_dense, T_true, s_C - dt_true)
        frames.append(jc.make_frame(traj, pts, t_abs, stamp, s_C, dtimg, k, k))
    return frames


# ----------------------------------------------------------------------------- trajectory / deskew
def test_trajectory_interpolates_key_poses_and_constant_twist():
    traj = make_traj(step=0.05)
    R, p = traj.pose_at(traj.times[3])
    assert np.allclose(R, traj.rotations[3], atol=1e-12) and np.allclose(p, traj.positions[3])
    tm = 0.5 * (traj.times[3] + traj.times[4])
    Rm, _ = traj.pose_at(tm)
    half = Rotation.from_matrix(traj.rotations[3].T @ Rm).magnitude()
    full = Rotation.from_matrix(traj.rotations[3].T @ traj.rotations[4]).magnitude()
    assert half == pytest.approx(full / 2, rel=1e-6)


def test_per_point_deskew_maps_static_points_consistently():
    traj = make_traj()
    rng = np.random.default_rng(1)
    Pw = rng.uniform([-5, 5, -1], [5, 20, 3], (200, 3))
    t_pts = rng.uniform(0.40, 0.50, 200)
    R, p = traj.pose_at(t_pts)
    pts = np.einsum("nji,nj->ni", R, Pw - p)
    assert np.allclose(points_to_world(traj, pts, t_pts), Pw, atol=1e-9)
    tau = 0.47
    expected = world_to_lidar_at(traj, Pw, tau)
    assert np.allclose(deskew_points(traj, pts, t_pts, tau), expected, atol=1e-9)


def test_bad_point_times_are_rejected():
    t_ns = np.array([0, 50_000_000, 99_500_000, 4_290_000_000, 200_000_000], dtype=np.uint32)
    assert valid_point_times(t_ns).tolist() == [True, True, True, False, False]


# ----------------------------------------------------------------------------- features
def test_depth_edges_keep_foreground_and_skip_grazing_ground():
    rows, cols = 32, 128
    rng_img = np.full((rows, cols), 20.0)
    rng_img[8:20, 50:60] = 5.0                       # box in front of a wall
    # grazing ground in the bottom rows: range grows smoothly ring by ring
    for i, r in enumerate(range(24, 32)):
        rng_img[r, :] = 30.0 / (1.0 + 0.35 * i)
    valid = np.ones_like(rng_img, bool)
    m = jc.lidar_depth_edges(rng_img, valid)
    assert m[10:18, 50].all() and m[10:18, 59].all()  # box contour, foreground side
    assert not m[10:18, 49].any() and not m[10:18, 60].any()  # wall side not marked
    assert not m[25:31, :].any()                      # smooth ground steps are not edges


def test_out_of_image_points_get_neutral_cost():
    traj = make_traj()
    pts = np.array([[0.0, 10.0, 0.0], [0.5, 10.0, 0.5], [500.0, 10.0, 0.0]])  # last one far off-image
    f = jc.make_frame(traj, pts, np.full(3, 0.5), 0.5, 0.5, np.full((H, W), 3.0, np.float32))
    P = jc.JointProblem([f], traj, K, T_INIT)
    R, t, dt = jc.params_to_pose(np.zeros(7), T_INIT)
    rho = P.frame_rho(f, R, t, dt)
    assert rho[2] == pytest.approx(rho[:2].mean())


# ----------------------------------------------------------------------------- joint recovery
@pytest.fixture(scope="module")
def synthetic_setup():
    traj = make_traj()
    x_true = np.array([np.radians(0.6), np.radians(-0.4), np.radians(0.8), 0.04, -0.03, 0.05, 0.012])
    R_true, t_true, dt_true = jc.params_to_pose(x_true, T_INIT)
    frames = synthetic_frames(traj, jc.make_T(R_true, t_true), dt_true)
    return traj, frames, x_true


def test_joint_recovers_rotation_translation_and_time(synthetic_setup):
    traj, frames, x_true = synthetic_setup
    P = jc.JointProblem(frames, traj, K, T_INIT)
    x0 = jc.pose_to_params(T_INIT, -0.0195, T_INIT)   # start at the stamp value, pose at the hand guess
    sol = jc.solve_joint(P, x0)
    R_t, t_t, dt_t = jc.params_to_pose(x_true, T_INIT)
    R_e, t_e, dt_e = jc.params_to_pose(sol.x, T_INIT)
    assert abs(dt_e - dt_t) < 0.003
    assert jc.rotation_angle_deg(R_t, R_e) < 0.15
    assert np.linalg.norm(t_e - t_t) < 0.03
    assert not any(sol.at_bound.values())


def test_scan_level_time_cannot_explain_fast_motion(synthetic_setup):
    """At the true parameters, per-point deskew fits exactly; stamping the whole scan at t_L does not."""
    traj, frames, x_true = synthetic_setup
    pp = jc.JointProblem(frames, traj, K, T_INIT, per_point=True)
    sl = jc.JointProblem(frames, traj, K, T_INIT, per_point=False)
    assert pp.median_distance(x_true) < 1.0  # integer-pixel rendering only
    assert sl.median_distance(x_true) > 2 * pp.median_distance(x_true) + 1.0


def test_scan_level_time_leaks_into_the_pose(synthetic_setup):
    """The confound: without per-point times the sweep delay is absorbed by R, t."""
    traj, frames, x_true = synthetic_setup
    sl = jc.JointProblem(frames, traj, K, T_INIT, per_point=False)
    sol = jc.solve_joint(sl, jc.pose_to_params(T_INIT, -0.0195, T_INIT))
    R_t, t_t, _ = jc.params_to_pose(x_true, T_INIT)
    R_e, t_e, _ = jc.params_to_pose(sol.x, T_INIT)
    assert jc.rotation_angle_deg(R_t, R_e) > 0.5 or np.linalg.norm(t_e - t_t) > 0.03


# ----------------------------------------------------------------------------- sign convention & distortion
def test_time_offset_sign_convention_locked():
    """Lock ``t_C = t_L + dt`` across both positive and negative ground-truth offsets."""
    from calib.hybrid_sync import associate_camera_time

    traj = make_traj()
    segs = world_structures(seed=11)
    Pw_sparse = sample(segs, 0.12)
    Pw_dense = sample(segs, 0.01)

    for dt_true in (0.018, -0.0195):
        frames = []
        for k in range(14):
            stamp = 0.1 * k + 0.003
            pts, t_abs = lidar_capture(traj, Pw_sparse, stamp)
            t_fov = stamp + 0.075
            # Under t_C = t_L + dt, the camera stamp matching LiDAR time t_fov is t_fov + dt_true.
            s_C = associate_camera_time(t_fov, dt_true)
            dtimg = render_dt(traj, Pw_dense, T_INIT, s_C - dt_true)
            frames.append(jc.make_frame(traj, pts, t_abs, stamp, s_C, dtimg, k, k))
        P = jc.JointProblem(frames, traj, K, T_INIT)
        sol = jc.solve_joint(P, jc.pose_to_params(T_INIT, 0.0, T_INIT))
        _, _, dt_est = jc.params_to_pose(sol.x, T_INIT)
        assert dt_est == pytest.approx(dt_true, abs=0.002)


def test_distortion_projection_matches_opencv():
    rng = np.random.default_rng(5)
    pC = rng.uniform([-2.0, -1.5, 4.0], [2.0, 1.5, 25.0], (50, 3))
    dist4 = np.array([-0.32, 0.09, 0.002, -0.003])
    xn, yn = pC[:, 0] / pC[:, 2], pC[:, 1] / pC[:, 2]
    xd, yd, ok = jc.distort_normalized(xn, yn, dist4)
    assert ok.all()
    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    uv_cv, _ = cv2.projectPoints(pC, np.zeros(3), np.zeros(3), K, dist4)
    uv_cv = uv_cv.reshape(-1, 2)
    assert np.allclose(np.column_stack([u, v]), uv_cv, atol=1e-6)


def test_joint_recovers_pose_time_and_radial_distortion():
    traj = make_traj()
    segs = world_structures(seed=2)
    Pw_sparse = sample(segs, 0.12)
    Pw_dense = sample(segs, 0.01)
    x_true = np.array([np.radians(0.5), np.radians(-0.4), np.radians(0.6),
                       0.03, -0.02, -0.04, 0.016, -0.32, 0.08])
    R_t, t_t, dt_t = jc.params_to_pose(x_true[:7], T_INIT)
    d_t = jc.params_to_dist4(x_true, "k1_k2")

    frames = []
    for k in range(14):
        stamp = 0.1 * k + 0.003
        pts, t_abs = lidar_capture(traj, Pw_sparse, stamp)
        s_C = stamp + 0.0805
        pL = world_to_lidar_at(traj, Pw_dense, s_C - dt_t)
        pC = pL @ R_t.T + t_t
        ok = pC[:, 2] > 1.0
        xd, yd, rad_ok = jc.distort_normalized(pC[ok, 0] / pC[ok, 2], pC[ok, 1] / pC[ok, 2], d_t)
        u = K[0, 0] * xd[rad_ok] + K[0, 2]
        v = K[1, 1] * yd[rad_ok] + K[1, 2]
        img = np.full((H, W), 255, np.uint8)
        for a, b in zip(u, v):
            if 0 <= a < W and 0 <= b < H:
                img[int(b), int(a)] = 0
        dtimg = np.minimum(cv2.distanceTransform(img, cv2.DIST_L2, 5), 40.0).astype(np.float32)
        frames.append(jc.make_frame(traj, pts, t_abs, stamp, s_C, dtimg, k, k))

    P = jc.JointProblem(
        frames, traj, K, T_INIT,
        dist_model="k1_k2",
        dist_prior=([0.0, 0.0, 0.0, 0.0], [None, 0.5, None, None]),
    )
    x0 = np.r_[jc.pose_to_params(T_INIT, -0.0195, T_INIT), 0.0, 0.0]
    sol = jc.solve_joint(P, x0, maxfev=6000)
    R_e, t_e, dt_e = jc.params_to_pose(sol.x[:7], T_INIT)
    d_e = sol.distortion("k1_k2")
    assert abs(dt_e - dt_t) < 0.003
    assert jc.rotation_angle_deg(R_t, R_e) < 0.2
    assert np.linalg.norm(t_e - t_t) < 0.04
    assert abs(d_e[0] - d_t[0]) < 0.05
    assert not any(sol.at_bound.values())
