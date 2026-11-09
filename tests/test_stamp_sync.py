import numpy as np
import pytest

from calib.hybrid_sync import stamp_sync_stats

LIDAR_DT = 0.100
CAM_DT = 0.050


def _streams(phase, duration=10.0, cam_jitter=0.0, lidar_jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t_lidar = np.arange(0.0, duration, LIDAR_DT)
    t_cam = np.arange(-0.5, duration + 0.5, CAM_DT) + phase
    if lidar_jitter:
        t_lidar = t_lidar + rng.normal(0.0, lidar_jitter, t_lidar.shape)
    if cam_jitter:
        t_cam = np.sort(t_cam + rng.normal(0.0, cam_jitter, t_cam.shape))
    scans = [{"timestamp": float(t), "points": np.zeros((0, 3))} for t in t_lidar]
    return scans, t_cam


@pytest.mark.parametrize("phase", [0.012, -0.017, 0.0, 0.024])
def test_recovers_phase_exactly_without_jitter(phase):
    scans, t_cam = _streams(phase)
    st = stamp_sync_stats(scans, t_cam)
    assert st.n_pairs == len(scans)
    assert st.tau_stamp == pytest.approx(phase, abs=1e-9)
    assert st.median_abs_gap == pytest.approx(abs(phase), abs=1e-9)
    assert st.std_gap < 1e-9


@pytest.mark.parametrize("phase", [0.012, -0.017])
def test_median_robust_to_jitter(phase):
    scans, t_cam = _streams(phase, cam_jitter=0.003, lidar_jitter=0.001, seed=42)
    st = stamp_sync_stats(scans, t_cam)
    assert abs(st.tau_stamp - phase) < 0.0015
    assert st.std_gap > 0.0


def test_accepts_timestamp_sec_key_and_subset():
    phase = 0.008
    scans, t_cam = _streams(phase, duration=2.0)
    scans = [{"timestamp_sec": s["timestamp"]} for s in scans]
    st = stamp_sync_stats(scans, t_cam, scan_indices=np.array([0, 3, 5]))
    assert st.n_pairs == 3
    assert st.tau_stamp == pytest.approx(phase, abs=1e-9)
