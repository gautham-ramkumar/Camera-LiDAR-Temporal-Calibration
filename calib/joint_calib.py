"""Joint camera–LiDAR calibration: rotation R, translation t and time offset Δt
estimated together in one robust optimiser, with per-point LiDAR deskew.

Time convention (same sign as the old ``tau`` in :mod:`calib.hybrid_sync`)::

    t_C = t_L + Δt     →  an image stamped s_C was exposed at LiDAR time s_C − Δt

Every LiDAR edge point keeps its own capture time ``t_i = stamp + t_field``.
For a candidate (R, t, Δt) each point is moved with the LiDAR trajectory into
the LiDAR frame at the image's exposure time and projected::

    p_L(τ) = T_W_L(τ)^-1 · T_W_L(t_i) · p_i,     τ = s_C − Δt
    u      = π(K (R p_L(τ) + t))

and scored against the image edge distance transform (bilinear lookup, so the
residual changes smoothly with all seven parameters). A Cauchy loss makes the
fit robust to clutter, dynamic objects and unmatched edges.

The two-stage pipeline fixed Δt from stamps first and then fitted the pose, so
any timing error was absorbed into the translation/rotation (and vice versa).
Here they compete inside one cost, so motion-rich frames can separate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .trajectory import Trajectory, points_to_world

POSE_PARAM_NAMES = ("rx", "ry", "rz", "tx", "ty", "tz", "dt")
DIST_MODELS = {
    "none": (),
    "k1": ("k1",),
    "k1_k2": ("k1", "k2"),
    "k1_k2_p1_p2": ("k1", "k2", "p1", "p2"),
}
DIST_BOUNDS = {
    "k1": (-0.6, 0.6),
    "k2": (-1.0, 1.0),
    "p1": (-0.05, 0.05),
    "p2": (-0.05, 0.05),
}
DIST_SCALE = {"k1": 0.02, "k2": 0.05, "p1": 0.002, "p2": 0.002}
DIST_JAC_STEP = {"k1": 2e-3, "k2": 5e-3, "p1": 2e-4, "p2": 2e-4}
PARAM_NAMES = POSE_PARAM_NAMES  # backward-compatible alias


# --------------------------------------------------------------------------- features
def image_distance_transform(gray: np.ndarray, blur: int = 3, cap_px: float = 40.0) -> np.ndarray:
    """Canny edges (median-adaptive thresholds after a light blur) → Euclidean DT in pixels, capped."""
    work = cv2.GaussianBlur(gray, (blur, blur), 0) if blur and blur > 1 else gray
    med = float(np.median(work))
    edges = cv2.Canny(work, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5).astype(np.float32)
    return np.minimum(dt, np.float32(cap_px))


def _support(mask: np.ndarray, axis: int, half: int, wrap: bool) -> np.ndarray:
    """Count of edge pixels within ±half along ``axis`` (including self)."""
    acc = np.zeros(mask.shape, dtype=np.int16)
    for d in range(-half, half + 1):
        sh = np.roll(mask, d, axis=axis)
        if not wrap and d != 0:
            if axis == 0:
                if d > 0:
                    sh[:d, :] = False
                else:
                    sh[d:, :] = False
            else:
                if d > 0:
                    sh[:, :d] = False
                else:
                    sh[:, d:] = False
        acc += sh.astype(np.int16)
    return acc


def lidar_depth_edges(rng: np.ndarray, valid: np.ndarray, abs_thresh: float = 0.5,
                      rel_thresh: float = 0.08, min_support: int = 3, half: int = 2,
                      iso_ratio: float = 3.0) -> np.ndarray:
    """Foreground side of depth discontinuities on the organized (rows, cols) range image.

    Compares azimuth neighbours (with wrap-around) and elevation neighbours that
    both have a return, and keeps only the *nearer* point of each jump so the
    edge sits on the occluding contour. Azimuth jumps (vertical contours such as
    poles and building corners) must repeat on ``min_support`` of the nearby
    rings, and elevation jumps (horizontal contours) on ``min_support`` nearby
    columns; this drops most isolated speckle from foliage.
    """
    rng = np.where(valid, rng, np.nan).astype(np.float64)
    out = np.zeros(rng.shape, dtype=bool)
    for axis, wrap in ((1, True), (0, False)):
        nb = np.roll(rng, -1, axis=axis)
        if not wrap:
            nb[-1, :] = np.nan
        both = np.isfinite(rng) & np.isfinite(nb)
        diff = np.where(both, nb - rng, 0.0)
        thr = np.maximum(abs_thresh, rel_thresh * np.fmin(rng, nb))
        jump = both & (np.abs(diff) > thr)
        # Grazing surfaces (flat ground seen ring after ring, walls at a shallow
        # angle) produce large but *smooth* range steps. A true occlusion is an
        # isolated step: it must dominate the steps just before and after it.
        ad = np.where(both, np.abs(diff), np.nan)
        prev_d = np.roll(ad, 1, axis=axis)
        next_d = np.roll(ad, -1, axis=axis)
        if not wrap:
            prev_d[0, :] = np.nan
            next_d[-1, :] = np.nan
        iso = (np.nan_to_num(prev_d, nan=0.0) * iso_ratio < ad) & (np.nan_to_num(next_d, nan=0.0) * iso_ratio < ad)
        jump &= iso
        near_here = jump & (diff > 0)            # this point is in front
        near_next = jump & (diff < 0)            # neighbour is in front
        m = near_here.copy()
        if wrap:
            m |= np.roll(near_next, 1, axis=axis)
        else:
            m[1:, :] |= near_next[:-1, :]
        if min_support > 1:
            other = 0 if axis == 1 else 1
            m &= _support(m, other, half, wrap=(other == 1)) >= min_support
        out |= m
    return out


# --------------------------------------------------------------------------- data
@dataclass
class JointFrame:
    scan_idx: int
    cam_idx: int
    cam_stamp: float
    lidar_stamp: float
    q_point: np.ndarray        # (N,3) world points, each from its own capture time
    q_scan: np.ndarray         # (N,3) world points if the whole scan is stamped at t_L
    t_points: np.ndarray       # (N,) capture times
    dt_img: np.ndarray         # (H,W) float32 distance transform
    speed: float = 0.0         # m/s at exposure
    ang_rate: float = 0.0      # rad/s at exposure
    block: str = "train"


def make_frame(
    traj: Trajectory,
    points_L: np.ndarray,
    t_points: np.ndarray,
    lidar_stamp: float,
    cam_stamp: float,
    dt_img: np.ndarray,
    scan_idx: int = -1,
    cam_idx: int = -1,
    block: str = "train",
) -> JointFrame:
    q_point = points_to_world(traj, points_L, t_points)
    q_scan = points_to_world(traj, points_L, np.full(len(t_points), float(lidar_stamp)))
    w, v = traj.velocity_at(cam_stamp)
    return JointFrame(scan_idx, cam_idx, float(cam_stamp), float(lidar_stamp), q_point, q_scan,
                      np.asarray(t_points, float), dt_img, float(np.linalg.norm(v)),
                      float(np.linalg.norm(w)), block)


# --------------------------------------------------------------------------- model
def dist_param_names(dist_model: str = "none") -> tuple[str, ...]:
    if dist_model not in DIST_MODELS:
        raise ValueError(f"Unknown dist_model {dist_model!r}; expected one of {tuple(DIST_MODELS)}")
    return DIST_MODELS[dist_model]


def problem_param_names(dist_model: str = "none") -> tuple[str, ...]:
    return POSE_PARAM_NAMES + dist_param_names(dist_model)


def params_to_pose(x: Sequence[float], T0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """x[:7] = [rotvec (left increment on R0), Δtranslation, Δt] → (R, t, Δt)."""
    x = np.asarray(x, dtype=np.float64)
    R = Rotation.from_rotvec(x[:3]).as_matrix() @ T0[:3, :3]
    t = T0[:3, 3] + x[3:6]
    return R, t, float(x[6])


def params_to_dist4(
    x: Sequence[float],
    dist_model: str = "none",
    fixed_dist: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Extract (k1, k2, p1, p2) from ``x`` (or ``fixed_dist`` when ``x`` has length 7)."""
    d4 = np.zeros(4, dtype=np.float64)
    if fixed_dist is not None:
        fd = np.asarray(fixed_dist, dtype=np.float64).reshape(-1)
        d4[:min(len(fd), 4)] = fd[:4]
    names = dist_param_names(dist_model)
    x = np.asarray(x, dtype=np.float64)
    if names and len(x) >= 7 + len(names):
        for i, name in enumerate(names):
            idx = ("k1", "k2", "p1", "p2").index(name)
            d4[idx] = float(x[7 + i])
    return d4


def distort_normalized(xn: np.ndarray, yn: np.ndarray, dist4: Sequence[float]):
    """Brown–Conrady distortion on normalized image coords; returns (xd, yd, rad_ok)."""
    k1, k2, p1, p2 = (float(dist4[0]), float(dist4[1]), float(dist4[2]), float(dist4[3]))
    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
        return xn, yn, np.ones(xn.shape, dtype=bool)
    r2 = xn * xn + yn * yn
    rad = 1.0 + k1 * r2 + k2 * (r2 * r2)
    rad_ok = rad > 0.2  # guard against radial fold-back outside the FOV
    xd = xn * rad + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd, rad_ok


def pose_to_params(T: np.ndarray, dt: float, T0: np.ndarray) -> np.ndarray:
    dR = T[:3, :3] @ T0[:3, :3].T
    return np.concatenate([Rotation.from_matrix(dR).as_rotvec(), T[:3, 3] - T0[:3, 3], [dt]])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class JointProblem:
    """Seven-parameter robust edge-alignment problem.

    Residuals are ``sqrt`` of the Cauchy cost per point, so plain least squares
    on them minimises the robust cost and optional Gaussian priors stay
    quadratic (they are not flattened by the robust loss).
    """

    frames: List[JointFrame]
    traj: Trajectory
    K: np.ndarray
    T0: np.ndarray
    per_point: bool = True            # False = every point stamped at its scan header time
    cap_px: float = 40.0
    min_depth: float = 1.0
    cauchy_px: float = 5.0            # robust scale (same 5 px as the repo's Gaussian edge score)
    dt_prior: Optional[tuple] = None    # (mean_s, sigma_s)
    t_prior: Optional[tuple] = None     # (mean_t (3,), sigma_m)
    dist_model: str = "none"            # "none", "k1", "k1_k2", "k1_k2_p1_p2"
    fixed_dist: Optional[Sequence[float]] = None  # optional fixed (k1,k2,p1,p2) when x has length 7
    dist_prior: Optional[tuple] = None  # (mean_seq, sigma_seq); None/inf sigma = unregularised
    _offsets: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._offsets = [0]
        for f in self.frames:
            self._offsets.append(self._offsets[-1] + len(f.q_point))

    @property
    def n_points(self) -> int:
        return self._offsets[-1]

    @property
    def prior_weight(self) -> float:
        """A prior σ costs as much as one average frame being off by ~1 robust unit.

        Residuals inside a frame are strongly correlated, so the frame (not the
        point) is the independent unit; this keeps priors weak but not
        negligible against tens of thousands of points.
        """
        return float(np.sqrt(self.n_points / max(len(self.frames), 1)))

    def _dist_prior_terms(self, dist4: np.ndarray) -> List[float]:
        if not self.dist_prior:
            return []
        means, sigmas = self.dist_prior
        w = self.prior_weight
        terms = []
        for val, mu, sig in zip(dist4, means, sigmas):
            if sig is not None and np.isfinite(sig) and sig > 0:
                terms.append(w * (float(val) - float(mu)) / float(sig))
        return terms

    def project(self, f: JointFrame, R: np.ndarray, t: np.ndarray, dt: float,
                dist: Optional[Sequence[float]] = None):
        tau = f.cam_stamp - dt
        Rw, pw = self.traj.pose_at(tau)
        q = f.q_point if self.per_point else f.q_scan
        pL = (q - pw) @ Rw                       # world → LiDAR frame at τ
        pC = pL @ R.T + t
        z = pC[:, 2]
        good = z > self.min_depth
        zs = np.where(good, z, 1.0)
        xn = pC[:, 0] / zs
        yn = pC[:, 1] / zs
        if dist is None:
            dist = self.fixed_dist if self.fixed_dist is not None else (0.0, 0.0, 0.0, 0.0)
        xd, yd, rad_ok = distort_normalized(xn, yn, dist)
        good &= rad_ok
        u = self.K[0, 0] * xd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        return u, v, good

    def frame_distances(self, f: JointFrame, R, t, dt,
                        dist: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Edge distance per point (px) and an in-image mask."""
        u, v, good = self.project(f, R, t, dt, dist=dist)
        h, w = f.dt_img.shape
        inside = good & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        d = np.full(u.shape, self.cap_px, dtype=np.float64)
        if inside.any():
            mx = u[inside].astype(np.float32).reshape(1, -1)
            my = v[inside].astype(np.float32).reshape(1, -1)
            d[inside] = cv2.remap(f.dt_img, mx, my, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=self.cap_px).reshape(-1)
        return d, inside

    def frame_rho(self, f: JointFrame, R, t, dt,
                  dist: Optional[Sequence[float]] = None) -> np.ndarray:
        """Per-point Cauchy cost ``ρ = ln(1 + (d/c)^2)``.

        Points that leave the image get the frame's mean in-image cost, so
        pushing points out of (or into) the frame is never rewarded. Without
        this, a fixed out-of-image penalty biases the solution toward shrinking
        the projection (camera moved back, or deskew to an earlier time).
        """
        d, inside = self.frame_distances(f, R, t, dt, dist=dist)
        rho = np.log1p((d / self.cauchy_px) ** 2)
        if not inside.all():
            rho[~inside] = rho[inside].mean() if inside.any() else np.log1p((self.cap_px / self.cauchy_px) ** 2)
        return rho

    def residuals(self, x: Sequence[float], frame_weights: Optional[np.ndarray] = None) -> np.ndarray:
        R, t, dt = params_to_pose(x, self.T0)
        dist4 = params_to_dist4(x, self.dist_model, self.fixed_dist)
        d_terms = self._dist_prior_terms(dist4)
        n_prior = (1 if self.dt_prior else 0) + (3 if self.t_prior else 0) + len(d_terms)
        out = np.empty(self.n_points + n_prior)
        for i, f in enumerate(self.frames):
            r = np.sqrt(self.frame_rho(f, R, t, dt, dist=dist4))
            if frame_weights is not None:
                r = r * np.sqrt(frame_weights[i])
            out[self._offsets[i]:self._offsets[i + 1]] = r
        k = self.n_points
        w = self.prior_weight
        if self.dt_prior:
            mu, sig = self.dt_prior
            out[k] = w * (dt - mu) / sig
            k += 1
        if self.t_prior:
            mu, sig = self.t_prior
            out[k:k + 3] = w * (t - np.asarray(mu, float)) / sig
            k += 3
        for term in d_terms:
            out[k] = term
            k += 1
        return out

    def cost(self, x: Sequence[float], frame_weights: Optional[np.ndarray] = None) -> float:
        r = self.residuals(x, frame_weights)
        return 0.5 * float(r @ r)

    def median_distance(self, x: Sequence[float], frames: Optional[Sequence[int]] = None) -> float:
        R, t, dt = params_to_pose(x, self.T0)
        dist4 = params_to_dist4(x, self.dist_model, self.fixed_dist)
        idx = range(len(self.frames)) if frames is None else frames
        ds = []
        for i in idx:
            d, inside = self.frame_distances(self.frames[i], R, t, dt, dist=dist4)
            ds.append(d[inside])
        allv = np.concatenate(ds) if ds else np.array([np.nan])
        return float(np.median(allv))

    def per_frame_median(self, x: Sequence[float]) -> np.ndarray:
        R, t, dt = params_to_pose(x, self.T0)
        dist4 = params_to_dist4(x, self.dist_model, self.fixed_dist)
        out = []
        for f in self.frames:
            d, inside = self.frame_distances(f, R, t, dt, dist=dist4)
            out.append(float(np.median(d[inside])) if inside.any() else np.nan)
        return np.asarray(out)


# --------------------------------------------------------------------------- solver
#: rotation ±5°, translation ±0.40 m (the hand-set lever arm is a rough guess),
#: time offset ±60 ms (well inside one 100 ms LiDAR period, so the −120 ms alias
#: of the old free search cannot be reached by the main estimate).
DEFAULT_BOUNDS = (
    np.array([-np.radians(5)] * 3 + [-0.40] * 3 + [-0.060]),
    np.array([np.radians(5)] * 3 + [0.40] * 3 + [0.060]),
)
#: parameter scale used to normalise the search (≈ equal pixel effect)
X_SCALE = np.array([2e-3, 2e-3, 2e-3, 2e-2, 2e-2, 2e-2, 5e-3])
#: central-difference steps for the Jacobian / Hessian
JAC_STEP = np.array([2e-4, 2e-4, 2e-4, 5e-3, 5e-3, 5e-3, 1e-3])


@dataclass
class JointSolution:
    x: np.ndarray
    cost: float
    nfev: int
    at_bound: Dict[str, bool] = field(default_factory=dict)
    start: Optional[np.ndarray] = None

    def pose(self, T0):
        R, t, dt = params_to_pose(self.x, T0)
        return make_T(R, t), dt

    def distortion(self, dist_model: str = "none") -> np.ndarray:
        return params_to_dist4(self.x, dist_model)


def problem_bounds(dist_model: str = "none") -> tuple[np.ndarray, np.ndarray]:
    names = dist_param_names(dist_model)
    if not names:
        return DEFAULT_BOUNDS
    lo = np.r_[DEFAULT_BOUNDS[0], [DIST_BOUNDS[n][0] for n in names]]
    hi = np.r_[DEFAULT_BOUNDS[1], [DIST_BOUNDS[n][1] for n in names]]
    return lo, hi


def problem_scales(dist_model: str = "none") -> np.ndarray:
    names = dist_param_names(dist_model)
    if not names:
        return X_SCALE
    return np.r_[X_SCALE, [DIST_SCALE[n] for n in names]]


def problem_jac_steps(dist_model: str = "none") -> np.ndarray:
    names = dist_param_names(dist_model)
    if not names:
        return JAC_STEP
    return np.r_[JAC_STEP, [DIST_JAC_STEP[n] for n in names]]


def _at_bound(x, bounds, free, names) -> Dict[str, bool]:
    span = bounds[1] - bounds[0]
    return {n: bool(free[i] and min(x[i] - bounds[0][i], bounds[1][i] - x[i]) < 2e-3 * span[i])
            for i, n in enumerate(names)}


def solve_joint(
    problem: JointProblem,
    x0: Sequence[float],
    bounds=None,
    scales: Sequence[float] = (15.0, 5.0),
    frame_weights: Optional[np.ndarray] = None,
    fix_mask: Optional[Sequence[bool]] = None,
    maxfev: int = 5000,
) -> JointSolution:
    """One optimiser over (R, t, Δt) and any active distortion parameters at once.

    The edge-distance cost is piecewise smooth but rough at the pixel scale
    (bilinear DT, many nearby edges), and gradient steps from finite
    differences stall in it. A bounded Powell search on normalised parameters
    is used instead, with a coarse-to-fine robust scale (Cauchy 15 px, then
    5 px) as continuation. Both stages move all active parameters together;
    nothing is estimated in a separate stage.

    ``fix_mask`` freezes parameters and is used only for diagnostics (Δt
    profile, per-window Δt), never for the main joint estimate.
    """
    from scipy.optimize import minimize

    names = problem_param_names(problem.dist_model)
    ndim = len(names)
    if bounds is None:
        bounds = problem_bounds(problem.dist_model)
    lo, hi = bounds
    x0_arr = np.asarray(x0, float)
    if x0_arr.size == 7 and ndim > 7:
        x0_arr = np.r_[x0_arr, np.zeros(ndim - 7)]
    x0_arr = np.clip(x0_arr, lo, hi)
    xscale = problem_scales(problem.dist_model)
    free = np.ones(ndim, bool) if fix_mask is None else ~np.asarray(fix_mask, bool)
    x = x0_arr.copy()
    nfev = 0
    for c in scales:
        problem.cauchy_px = float(c)

        def f(z):
            xx = x.copy()
            xx[free] = np.clip(z * xscale[free], lo[free], hi[free])
            return problem.cost(xx, frame_weights)

        res = minimize(
            f, x[free] / xscale[free], method="Powell",
            bounds=list(zip(lo[free] / xscale[free], hi[free] / xscale[free])),
            options={"xtol": 1e-3, "ftol": 1e-7, "maxfev": maxfev},
        )
        x[free] = np.clip(res.x * xscale[free], lo[free], hi[free])
        nfev += int(res.nfev)
    problem.cauchy_px = float(scales[-1])
    return JointSolution(x, problem.cost(x, frame_weights), nfev, _at_bound(x, bounds, free, names), x0_arr)


def numerical_jacobian(problem: JointProblem, x: np.ndarray, steps: Optional[np.ndarray] = None) -> np.ndarray:
    """Central-difference Jacobian of the sqrt-Cauchy residual vector."""
    if steps is None:
        steps = problem_jac_steps(problem.dist_model)
    ndim = len(x)
    cols = []
    for i in range(ndim):
        e = np.zeros(ndim)
        e[i] = steps[i]
        cols.append((problem.residuals(x + e) - problem.residuals(x - e)) / (2 * steps[i]))
    return np.stack(cols, axis=1)


def gauss_newton_covariance(problem: JointProblem, x: np.ndarray) -> np.ndarray:
    """(JᵀJ)⁻¹·s² at the optimum, with s² = Σr²/(N−p).

    Treats residuals as independent, so it is optimistic (points within a
    frame are correlated); the frame bootstrap is the more honest spread.
    """
    J = numerical_jacobian(problem, x)
    r = problem.residuals(x)
    s2 = float(r @ r) / max(r.size - len(x), 1)
    return np.linalg.pinv(J.T @ J) * s2


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    return cov / np.outer(s, s)


def rpy_deg(R: np.ndarray) -> np.ndarray:
    """Intrinsic xyz Euler angles in degrees (same convention as the extrinsic report)."""
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))
