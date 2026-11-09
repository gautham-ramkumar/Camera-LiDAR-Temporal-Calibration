import numpy as np
import pytest

from calib.scoring import blended_distances

H, W = 4, 5


def _const_dts(values):
    return [np.full((H, W), v, dtype=np.float32) for v in values]


@pytest.fixture
def setup():
    cam_times = np.array([0.0, 1.0, 2.0])
    dts = _const_dts([2.0, 6.0, 10.0])
    ui = np.array([0, 1, 4, 2])
    vi = np.array([0, 3, 2, 1])
    return cam_times, dts, ui, vi


@pytest.mark.parametrize(
    "t, expected",
    [
        (0.0, 2.0),
        (0.25, 3.0),   # 0.75*2 + 0.25*6
        (0.5, 4.0),
        (1.0, 6.0),
        (1.5, 8.0),    # 0.5*6 + 0.5*10
        (1.9, 9.6),
        (2.0, 10.0),
    ],
)
def test_linear_interpolation_between_frames(setup, t, expected):
    cam_times, dts, ui, vi = setup
    d = blended_distances(cam_times, dts, t, ui, vi)
    assert d.shape == ui.shape
    np.testing.assert_allclose(d, expected, rtol=1e-6)


def test_clamps_outside_time_range(setup):
    cam_times, dts, ui, vi = setup
    np.testing.assert_allclose(blended_distances(cam_times, dts, -3.0, ui, vi), 2.0)
    np.testing.assert_allclose(blended_distances(cam_times, dts, 7.0, ui, vi), 10.0)


def test_single_frame_and_empty():
    ui = np.array([1, 2])
    vi = np.array([0, 1])
    np.testing.assert_allclose(
        blended_distances(np.array([0.3]), _const_dts([4.0]), 12.0, ui, vi), 4.0
    )
    out = blended_distances(np.array([]), [], 0.0, ui, vi)
    np.testing.assert_allclose(out, 0.0)
    assert out.shape == ui.shape
