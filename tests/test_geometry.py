import numpy as np
import pytest

from calib.geometry import euler_to_R, lidar_points_to_pixels, make_T


@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0, 0.0), (0.1, -0.2, 0.3), (1.2, 0.7, -2.5), (np.pi, -np.pi / 2, 0.4)],
)
def test_euler_to_R_is_orthonormal(angles):
    R = euler_to_R(*angles)
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_euler_to_R_single_axis():
    a = 0.3
    Rz = euler_to_R(0.0, 0.0, a)
    np.testing.assert_allclose(
        Rz,
        [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]],
        atol=1e-12,
    )


def _K():
    return np.array([[500.0, 0.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]])


def test_projection_matches_K_R_t_by_hand():
    K = _K()
    R = euler_to_R(0.05, -0.1, 0.2)
    t = np.array([0.1, -0.2, 0.3])
    T = make_T(R, t)
    np.testing.assert_allclose(T[:3, :3], R)
    np.testing.assert_allclose(T[:3, 3], t)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])

    pts_L = np.array(
        [[0.5, 0.2, 4.0], [-1.0, 0.3, 6.0], [0.0, 0.0, 2.0], [2.0, -1.5, 10.0]]
    )
    uv = lidar_points_to_pixels(pts_L, T, K)

    expected = []
    P = K @ np.hstack([R, t.reshape(3, 1)])
    for p in pts_L:
        h = P @ np.append(p, 1.0)
        expected.append(h[:2] / h[2])
    np.testing.assert_allclose(uv, np.array(expected), rtol=1e-4, atol=1e-3)


def test_points_with_nonpositive_depth_are_dropped():
    K = _K()
    T = np.eye(4)
    pts = np.array(
        [
            [0.0, 0.0, 5.0],   # kept
            [1.0, 1.0, 0.0],   # Z == 0 dropped
            [1.0, 1.0, -3.0],  # behind camera dropped
            [0.5, -0.5, 2.0],  # kept
        ]
    )
    uv = lidar_points_to_pixels(pts, T, K)
    assert uv.shape == (2, 2)
    np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-4)
    np.testing.assert_allclose(uv[1], [320.0 + 500 * 0.25, 240.0 - 480 * 0.25], atol=1e-3)


def test_all_points_behind_camera_returns_empty():
    uv = lidar_points_to_pixels(np.array([[0.0, 0.0, -1.0]]), np.eye(4), _K())
    assert uv.shape == (0, 2)
