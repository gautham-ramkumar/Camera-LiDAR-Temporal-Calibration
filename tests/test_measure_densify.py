"""Tests for scripts/measure_densify.py. They do not need huntington.mcap."""

import importlib.util
import os

import numpy as np
import pytest

from calib.lidar import extract_edges, project_to_range_image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "measure_densify", os.path.join(_ROOT, "scripts", "measure_densify.py")
)
md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md)

H, W = 128, 1024


def _step_scan(t, step_cols=(500,), near=5.0, far=10.0):
    """
    Organized H*W scan whose range jumps near->far at each column in
    ``step_cols``. That gives exactly one edge per row per step, so E = H * len(step_cols).
    """
    u = np.arange(W)
    yaw = (u + 0.5) / W * 2 * np.pi - np.pi
    r_row = np.full(W, near, dtype=np.float64)
    for i, c in enumerate(step_cols):
        r_row[c:] = far if i % 2 == 0 else near
    r = np.tile(r_row, (H, 1))
    yaw = np.tile(yaw, (H, 1))
    pts = np.stack([r * np.cos(yaw), r * np.sin(yaw), np.zeros_like(r)], axis=-1)
    return {"timestamp_sec": float(t), "points": pts.reshape(-1, 3).astype(np.float32)}


def _flat_scan(t):
    return _step_scan(t, step_cols=())


def test_step_scan_has_known_edge_count():
    scan = _step_scan(0.0)
    grid = project_to_range_image(scan["points"])
    assert extract_edges(grid).shape[0] == H


def test_three_identical_scans_give_ratio_three_with_no_imu():
    scans = [_step_scan(t) for t in (0.0, 0.1, 0.2)]
    # No IMU samples -> preintegrate returns identity, so the warp is a no-op.
    out = md.measure(scans, [])
    assert out["n_scans"] == 3
    assert out["n_dense_frames"] == 1
    assert out["n_pairs"] == 1
    assert out["mean_single_n"] == H
    assert out["mean_dense_n"] == 3 * H
    assert out["ratio_mean"] == pytest.approx(3.0)
    assert out["ratio_median"] == pytest.approx(3.0)
    assert out["ratio_std"] == pytest.approx(0.0)


def test_ratio_reflects_neighbor_edge_counts():
    # prev has 1 step (H edges), curr 2 steps (2H), next 3 steps (3H).
    scans = [
        _step_scan(0.0, step_cols=(300,)),
        _step_scan(0.1, step_cols=(300, 600)),
        _step_scan(0.2, step_cols=(200, 500, 800)),
    ]
    out = md.measure(scans, [])
    assert out["n_pairs"] == 1
    assert out["mean_single_n"] == 2 * H
    assert out["mean_dense_n"] == 6 * H
    assert out["ratio_mean"] == pytest.approx(3.0)


def test_sliding_window_over_five_scans():
    scans = [_step_scan(0.1 * i) for i in range(5)]
    out = md.measure(scans, [])
    # First and last scans lack a neighbour on one side.
    assert out["n_dense_frames"] == 3
    assert out["n_pairs"] == 3
    assert out["ratio_p10"] == pytest.approx(3.0)
    assert out["ratio_p90"] == pytest.approx(3.0)


def test_edgeless_scan_is_skipped_by_densify_and_matching():
    # densify drops the flat scan, so scans 0, 2, 3 form the only window (centered on 2).
    scans = [_step_scan(0.0), _flat_scan(0.1), _step_scan(0.2), _step_scan(0.3)]
    s_ts, s_n = md.single_edge_counts(scans)
    assert list(s_n) == [H, 0, H, H]
    out = md.measure(scans, [])
    assert out["n_dense_frames"] == 1
    assert out["n_pairs"] == 1
    assert out["ratio_mean"] == pytest.approx(3.0)


def test_densify_edge_ratios_pure():
    single_ts = np.array([0.0, 0.1, 0.2, 0.3])
    single_n = np.array([100, 0, 50, 200])
    dense_ts = np.array([0.1, 0.2, 0.3, 9.9])  # 0.1 has single_n==0, 9.9 has no match
    dense_n = np.array([999, 150, 400, 10])
    s, d, r = md.densify_edge_ratios(single_ts, single_n, dense_ts, dense_n)
    assert list(s) == [50, 200]
    assert list(d) == [150, 400]
    np.testing.assert_allclose(r, [3.0, 2.0])

    summ = md.summarize(s, d, r)
    assert summ["n_pairs"] == 2
    assert summ["ratio_mean"] == pytest.approx(2.5)
    assert summ["ratio_median"] == pytest.approx(2.5)
    assert summ["mean_single_n"] == pytest.approx(125.0)
    assert summ["mean_dense_n"] == pytest.approx(275.0)


def test_densify_edge_ratios_unsorted_single_timestamps():
    s, d, r = md.densify_edge_ratios(
        np.array([0.2, 0.0, 0.1]), np.array([20, 10, 5]), np.array([0.1]), np.array([15])
    )
    assert list(s) == [5] and list(d) == [15]
    np.testing.assert_allclose(r, [3.0])


def test_summarize_empty():
    e = np.empty(0)
    summ = md.summarize(e, e, e)
    assert summ["n_pairs"] == 0
    assert summ["ratio_mean"] is None


def test_main_missing_mcap_returns_error(tmp_path):
    assert md.main(["--mcap", str(tmp_path / "nope.mcap")]) == 2
