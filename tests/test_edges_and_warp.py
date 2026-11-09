"""Regression tests for validity-aware LiDAR edges and per-candidate pose warp."""

import numpy as np

from calib.hybrid_sync import _override_fn
from calib.lidar import extract_edges
from calib.pose_warp import relative_motion
from calib.imu import IMUPreintegrator
from calib.scoring import seed_tau_grid


def _grid_from_ranges(ranges):
    """Grid of points along +x with the given ranges (0 = empty cell)."""
    r = np.asarray(ranges, dtype=np.float32)
    g = np.zeros(r.shape + (3,), dtype=np.float32)
    g[..., 0] = r
    return g


def test_empty_cells_are_not_edges():
    grid = _grid_from_ranges([[5.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0]])
    assert extract_edges(grid, threshold=1.0).shape == (0, 3)


def test_real_jump_between_valid_cells_is_edge():
    grid = _grid_from_ranges([[5.0, 5.1, 12.0, 12.0, 0.0, 3.0]])
    edges = extract_edges(grid, threshold=1.0)
    np.testing.assert_allclose(edges[:, 0], [5.1])


def _imu_log(t0, t1, rate=200.0, gyro_z=0.5):
    ts = np.arange(t0, t1, 1.0 / rate)
    return [
        {
            "timestamp_sec": float(t),
            "linear_acceleration": np.array([0.0, 0.0, 9.81]),
            "angular_velocity": np.array([0.0, 0.0, gyro_z]),
        }
        for t in ts
    ]


def test_relative_motion_slice_matches_full_scan():
    imu = IMUPreintegrator(gravity=np.array([0.0, 0.0, 9.81]))
    meas = _imu_log(0.0, 5.0)
    for t0, t1 in [(1.0, 1.05), (2.3, 2.2), (4.9, 5.5)]:
        lo, hi = min(t0, t1), max(t0, t1)
        full = imu.preintegrate(meas, lo, hi)
        if t1 < t0:
            full = np.linalg.inv(full)
        np.testing.assert_allclose(relative_motion(imu, meas, t0, t1), full, atol=1e-12)


def test_override_fn_warps_per_candidate_tau():
    imu = IMUPreintegrator(gravity=np.array([0.0, 0.0, 9.81]))
    meas = _imu_log(0.0, 2.0, gyro_z=2.0)
    scans = [{"timestamp": 1.0, "points": np.array([[5.0, 0.0, 0.0]])}]
    idx = np.array([0])
    fn = _override_fn(scans, idx, None, imu, meas, True, 100)
    a = fn(0.0)[0]
    b = fn(0.04)[0]
    expected = (relative_motion(imu, meas, 1.0, 1.04)[:3, :3] @ [5.0, 0.0, 0.0])
    np.testing.assert_allclose(a, [[5.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(b[0], expected, atol=1e-9)
    assert not np.allclose(a, b)
    # No IMU / warp disabled -> no per-candidate builder.
    assert _override_fn(scans, idx, None, None, None, True, 100) is None
    assert _override_fn(scans, idx, None, imu, meas, False, 100) is None


def test_seed_tau_grid_uses_override_fn_for_each_candidate():
    class _Pipe:
        sigma_px = 1.0
        dist_transforms = {0: np.zeros((4, 4), dtype=np.float32)}

    seen = []

    def fn(tau):
        seen.append(tau)
        return None

    scans = [{"timestamp": 0.0, "points": np.array([[0.0, 0.0, 1.0]])}]
    seed_tau_grid(
        _Pipe(), scans, np.array([0.0]), np.array([0]), np.eye(3), np.eye(4),
        np.array([0]), tau_min=-0.02, tau_max=0.02, num_samples=5,
        point_override_fn=fn,
    )
    np.testing.assert_allclose(seen, np.linspace(-0.02, 0.02, 5))
