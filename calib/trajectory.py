"""Continuous LiDAR trajectory (SE(3) interpolation) and per-point deskew.

A :class:`Trajectory` stores LiDAR-frame poses ``T_W_L(t_k)`` at key times and
interpolates between them with a constant body twist (slerp on rotation,
linear on position). Outside the key range it extrapolates with the first /
last segment's velocity.

Deskew moves a point captured at time ``t_i`` (in the LiDAR frame at that
instant) into the LiDAR frame at a target time ``tau``:

    p(tau) = T_W_L(tau)^-1 · T_W_L(t_i) · p_i

Because ``T_W_L(tau)`` is a smooth function of ``tau`` inside each segment,
the projected pixel moves continuously when the target time changes, which
is what lets the joint optimiser treat the time offset as a free variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Trajectory:
    times: np.ndarray        # (K,) sorted key times [s]
    rotations: np.ndarray    # (K, 3, 3) R_W_L
    positions: np.ndarray    # (K, 3)    p_W_L

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.rotations = np.asarray(self.rotations, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.times.size < 2:
            raise ValueError("Trajectory needs at least two key poses")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Trajectory key times must be strictly increasing")
        # Per-segment body rotation increments (rotation vectors).
        rel = np.einsum("kji,kjl->kil", self.rotations[:-1], self.rotations[1:])
        self._seg_rotvec = Rotation.from_matrix(rel).as_rotvec()

    def pose_at(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated ``(R, p)`` at time(s) ``t``; shapes (...,3,3), (...,3)."""
        t = np.asarray(t, dtype=np.float64)
        flat = t.reshape(-1)
        seg = np.searchsorted(self.times, flat, side="right") - 1
        seg = np.clip(seg, 0, self.times.size - 2)
        t0 = self.times[seg]
        dt = self.times[seg + 1] - t0
        a = (flat - t0) / dt  # may be <0 or >1 at the ends (extrapolation)
        dR = Rotation.from_rotvec(self._seg_rotvec[seg] * a[:, None]).as_matrix()
        R = np.einsum("kij,kjl->kil", self.rotations[seg], dR)
        p = self.positions[seg] + a[:, None] * (self.positions[seg + 1] - self.positions[seg])
        return R.reshape(t.shape + (3, 3)), p.reshape(t.shape + (3,))

    def velocity_at(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Body angular rate [rad/s] and world linear velocity [m/s] of the segment holding ``t``."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        seg = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, self.times.size - 2)
        dt = self.times[seg + 1] - self.times[seg]
        w = self._seg_rotvec[seg] / dt[:, None]
        v = (self.positions[seg + 1] - self.positions[seg]) / dt[:, None]
        return w, v


def points_to_world(traj: Trajectory, points_L: np.ndarray, t_points: np.ndarray) -> np.ndarray:
    """Map each LiDAR point (frame at its own capture time) into the world frame."""
    R, p = traj.pose_at(t_points)
    return np.einsum("nij,nj->ni", R, np.asarray(points_L, dtype=np.float64)) + p


def world_to_lidar_at(traj: Trajectory, q_world: np.ndarray, tau: float) -> np.ndarray:
    """Express world points in the LiDAR frame at a single time ``tau``."""
    R, p = traj.pose_at(float(tau))
    return (np.asarray(q_world, dtype=np.float64) - p) @ R


def deskew_points(traj: Trajectory, points_L: np.ndarray, t_points: np.ndarray, tau: float) -> np.ndarray:
    """Per-point deskew: move every point into the LiDAR frame at ``tau``."""
    return world_to_lidar_at(traj, points_to_world(traj, points_L, t_points), tau)
