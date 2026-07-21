"""Visualization helpers for temporal / extrinsic calibration."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import CameraPipeline
from .geometry import lidar_points_to_pixels
from .scoring import blended_distances


def _resize_to_fit(img: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _draw_label_bar(img: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Dark top bar with white text (readable on edge maps)."""
    out = img.copy()
    bar_h = 22 + 18 * len(lines)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (8, 20 + 18 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return out


def _colorize_distances(d: np.ndarray, saturating_px: float = 15.0) -> np.ndarray:
    """
    Map distance (px) → BGR via Turbo: near edge = yellow/green, far = purple/red.
    """
    d = np.asarray(d, dtype=np.float32)
    norm = np.clip(d / max(saturating_px, 1e-6), 0.0, 1.0)
    # OpenCV COLORMAP_JET: 0 -> blue (far), 1 -> red (near) after invert
    u8 = (255.0 * (1.0 - norm)).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET).reshape(-1, 3)
    return bgr


def make_overlay_image(
    pipeline: CameraPipeline,
    frame_id: int,
    projected_pixels: Optional[np.ndarray],
    show_edges: bool = True,
    max_width: int = 600,
    max_height: int = 600,
    label_text: Optional[str] = None,
) -> np.ndarray:
    """Legacy uniform-green overlay (kept for compatibility)."""
    base = (
        pipeline.edge_maps[frame_id]
        if (show_edges and frame_id in pipeline.edge_maps)
        else pipeline.images[frame_id]
    )
    h, w = base.shape
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    if projected_pixels is not None and len(projected_pixels) > 0:
        pts = projected_pixels.astype(np.int32)
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[mask]
        for (u, v) in pts:
            cv2.circle(overlay, (int(u), int(v)), 2, (0, 255, 0), -1)

    overlay = _resize_to_fit(overlay, max_width, max_height)
    if label_text is not None:
        overlay = _draw_label_bar(overlay, [label_text])
    return overlay


def make_distance_overlay(
    pipeline: CameraPipeline,
    frame_id: int,
    projected_pixels: np.ndarray,
    distances_px: np.ndarray,
    *,
    show_edges: bool = True,
    max_width: int = 640,
    max_height: int = 640,
    saturating_px: float = 15.0,
    point_radius: int = 3,
    label_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Overlay LiDAR projections colored by distance-to-camera-edge (px).

    Near edge → warm/yellow (good alignment); far → cool/purple (bad).
    Background is the camera edge map (or grayscale image).
    """
    # Base image with scene context
    img_gray = pipeline.images[frame_id]
    h, w = img_gray.shape
    # Dim the original image so it acts as a nice background
    overlay = cv2.cvtColor((img_gray.astype(np.float32) * 0.35).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    if show_edges and frame_id in pipeline.edge_maps:
        edges = pipeline.edge_maps[frame_id]
        # Overlay edges as crisp white/light-gray lines
        edge_mask = edges > 0
        overlay[edge_mask] = [220, 220, 220]

    pts = np.asarray(projected_pixels, dtype=np.int32)
    d = np.asarray(distances_px, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        overlay = _resize_to_fit(overlay, max_width, max_height)
        if label_lines:
            overlay = _draw_label_bar(overlay, label_lines)
        return overlay

    mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[mask]
    d = d[mask] if d.shape[0] == mask.shape[0] else d[:0]
    if pts.shape[0] == 0:
        overlay = _resize_to_fit(overlay, max_width, max_height)
        if label_lines:
            overlay = _draw_label_bar(overlay, label_lines)
        return overlay

    colors = _colorize_distances(d, saturating_px=saturating_px)
    
    # Draw black borders first so they don't overwrite neighboring points' colors
    for (u, v) in pts:
        cv2.circle(overlay, (int(u), int(v)), point_radius + 1, (0, 0, 0), -1)
        
    # Draw colored points on top
    for (u, v), color in zip(pts, colors):
        cv2.circle(overlay, (int(u), int(v)), point_radius, tuple(int(c) for c in color), -1)

    # Colorbar strip on the right
    bar_w = 14
    bar = np.linspace(1.0, 0.0, overlay.shape[0], dtype=np.float32).reshape(-1, 1)
    bar_u8 = (255.0 * bar).astype(np.uint8)
    bar_bgr = cv2.applyColorMap(bar_u8, cv2.COLORMAP_JET)
    bar_bgr = np.repeat(bar_bgr, bar_w, axis=1)
    overlay = np.hstack([overlay, bar_bgr])

    overlay = _resize_to_fit(overlay, max_width, max_height)
    if label_lines:
        overlay = _draw_label_bar(overlay, list(label_lines))
    return overlay


def temporal_alignment_panel(
    pipeline: CameraPipeline,
    cam_times: np.ndarray,
    frame_ids: np.ndarray,
    points_L: np.ndarray,
    t_lidar: float,
    K: np.ndarray,
    T_CL: np.ndarray,
    tau: float,
    *,
    max_points: int = 8000,
    saturating_px: float = 15.0,
    max_width: int = 640,
    max_height: int = 640,
) -> Tuple[np.ndarray, dict]:
    """
    One temporal check panel: project at ``t_L + tau``, color by blended DT.

    Returns (BGR image, stats dict with mean_d, median_d, soft_score, n_pts).
    """
    pts = np.asarray(points_L, dtype=np.float64)
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
        pts = pts[idx]

    uv = lidar_points_to_pixels(pts, T_CL, K)
    cam_dts = [pipeline.dist_transforms[int(fid)] for fid in frame_ids]
    h, w = cam_dts[0].shape

    if uv.shape[0] == 0:
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        empty = _draw_label_bar(empty, [f"tau={tau*1000:.1f} ms", "no projections"])
        return _resize_to_fit(empty, max_width, max_height), {
            "mean_d": np.nan,
            "median_d": np.nan,
            "soft_score": 0.0,
            "n_pts": 0,
        }

    ui = uv[:, 0].astype(np.int32)
    vi = uv[:, 1].astype(np.int32)
    mask = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi = ui[mask], vi[mask]
    uv_valid = np.stack([ui, vi], axis=1)

    d = blended_distances(cam_times, cam_dts, float(t_lidar) + float(tau), ui, vi)
    sigma = float(pipeline.sigma_px)
    soft = float(np.sum(np.exp(-(d ** 2) / (2.0 * sigma ** 2)))) if d.size else 0.0
    mean_d = float(np.mean(d)) if d.size else np.nan
    median_d = float(np.median(d)) if d.size else np.nan

    # Nearest camera frame only for background context
    fid = int(frame_ids[int(np.argmin(np.abs(np.asarray(cam_times) - (t_lidar + tau))))])

    label = [
        f"tau = {tau * 1000:+.1f} ms",
        f"mean |d| = {mean_d:.1f} px   soft = {soft:.0f}",
    ]
    img = make_distance_overlay(
        pipeline,
        fid,
        uv_valid,
        d,
        saturating_px=saturating_px,
        max_width=max_width,
        max_height=max_height,
        label_lines=label,
    )
    stats = {
        "mean_d": mean_d,
        "median_d": median_d,
        "soft_score": soft,
        "n_pts": int(d.size),
        "tau": float(tau),
        "frame_id": fid,
        "distances": d,
    }
    return img, stats


def make_temporal_comparison(
    pipeline: CameraPipeline,
    cam_times: np.ndarray,
    frame_ids: np.ndarray,
    points_L: np.ndarray,
    t_lidar: float,
    K: np.ndarray,
    T_CL: np.ndarray,
    taus: Sequence[float],
    *,
    max_points: int = 8000,
    saturating_px: float = 15.0,
    panel_width: int = 520,
    panel_height: int = 520,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Side-by-side distance-colored panels for several ``tau`` values (same T_CL).

    Returns (RGB image for matplotlib, list of per-panel stats).
    """
    panels_bgr: List[np.ndarray] = []
    stats_list: List[dict] = []
    for tau in taus:
        img, st = temporal_alignment_panel(
            pipeline,
            cam_times,
            frame_ids,
            points_L,
            t_lidar,
            K,
            T_CL,
            float(tau),
            max_points=max_points,
            saturating_px=saturating_px,
            max_width=panel_width,
            max_height=panel_height,
        )
        panels_bgr.append(img)
        stats_list.append(st)

    combined = stack_overlays_horizontal(panels_bgr)
    return combined, stats_list


def stack_overlays_horizontal(overlays: list) -> np.ndarray:
    """Stack BGR overlays side-by-side (equalized height) and return RGB for matplotlib."""
    min_h = min(img.shape[0] for img in overlays)
    resized = [
        cv2.resize(img, (int(img.shape[1] * min_h / img.shape[0]), min_h))
        for img in overlays
    ]
    combined = np.hstack(resized)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


def plot_distance_histograms(
    stats_by_tau: Sequence[dict],
    distance_lists: Sequence[np.ndarray],
    labels: Sequence[str],
    ax=None,
):
    """
    Optional matplotlib helper: overlay histograms of edge distances for each tau.
    ``ax`` is a matplotlib Axes; if None, creates a new figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    for d, lab in zip(distance_lists, labels):
        d = np.asarray(d, dtype=float)
        if d.size == 0:
            continue
        ax.hist(d, bins=40, range=(0, 30), alpha=0.45, label=lab, density=True)
    ax.set_xlabel("distance to camera edge [px]")
    ax.set_ylabel("density")
    ax.set_title("Alignment quality vs tau (lower mass near 0 is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
