"""Camera edge detection and distance-transform scoring."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CameraPipeline:
    """Canny edges + raw-pixel distance transform for LiDAR–image alignment."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        sigma_px: float = 5.0,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.sigma_px = float(sigma_px)
        self.images = {}
        self.edge_maps = {}
        self.dist_transforms = {}

    def load_image(self, frame_id: int, path: str) -> np.ndarray:
        """Load an image from disk, convert to grayscale, store by frame_id."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.images[frame_id] = gray
        return gray

    def process_frame(self, frame_id: int, image: np.ndarray) -> np.ndarray:
        """
        Ingest a numpy image (BGR or gray), compute edges + distance transform.

        Returns the raw-pixel distance transform for ``frame_id``.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        self.images[frame_id] = gray
        self.compute_edges(frame_id)
        return self.compute_distance_transform(frame_id)

    def compute_edges(self, frame_id: int) -> np.ndarray:
        if frame_id not in self.images:
            raise KeyError(f"No image for frame_id={frame_id}")
        edges = cv2.Canny(self.images[frame_id], self.canny_low, self.canny_high)
        edges = (edges > 0).astype(np.uint8) * 255
        self.edge_maps[frame_id] = edges
        return edges

    def compute_distance_transform(self, frame_id: int) -> np.ndarray:
        """
        Raw pixel distance transform from the edge map.

        Edge pixels are 0; other pixels store Euclidean distance in pixels.
        (Not normalized — keeps the calibration score sharp in pixel units.)
        """
        if frame_id not in self.edge_maps:
            raise KeyError(f"No edge map for frame_id={frame_id}")
        edges = self.edge_maps[frame_id]
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        self.dist_transforms[frame_id] = dist.astype(np.float16)
        return self.dist_transforms[frame_id]

    def score_projection(
        self,
        frame_id: int,
        projected_pixels: np.ndarray,
        sigma: Optional[float] = None,
    ) -> float:
        """
        Sharp soft edge-alignment score. Higher is better.

        Each in-bounds projected point contributes ``exp(-d² / (2 σ²))`` where
        ``d`` is distance to the nearest image edge in **pixels**.
        """
        if frame_id not in self.dist_transforms:
            raise KeyError(f"No distance transform for frame_id={frame_id}")
        sigma = self.sigma_px if sigma is None else float(sigma)
        dist = self.dist_transforms[frame_id]
        h, w = dist.shape
        if len(projected_pixels) == 0:
            return 0.0

        pts = projected_pixels.astype(np.int32)
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[mask]
        if pts.shape[0] == 0:
            return 0.0

        d_vals = dist[pts[:, 1], pts[:, 0]]
        scores = np.exp(-(d_vals ** 2) / (2.0 * sigma ** 2))
        return float(scores.sum())

    def debug_show(
        self,
        frame_id: int,
        max_width: int = 1280,
        max_height: int = 720,
    ) -> Optional[np.ndarray]:
        """Return a side-by-side [image | edges | dist] visualization (BGR)."""
        if frame_id not in self.images:
            raise KeyError(f"No image for frame_id={frame_id}")
        if frame_id not in self.edge_maps:
            raise KeyError(f"No edge map for frame_id={frame_id}")
        if frame_id not in self.dist_transforms:
            raise KeyError(f"No dist transform for frame_id={frame_id}")

        img = self.images[frame_id]
        edges = self.edge_maps[frame_id]
        dist = self.dist_transforms[frame_id]

        # Cap display so large distances don't wash out near-edge structure
        dist_cap = np.clip(dist, 0, 50.0)
        dist_disp = (dist_cap / 50.0 * 255.0).astype(np.uint8)
        vis = np.hstack([
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
            cv2.applyColorMap(dist_disp, cv2.COLORMAP_JET),
        ])
        h, w = vis.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return vis

    def debug_overlay(
        self,
        frame_id: int,
        projected_pixels: np.ndarray,
        show_edges: bool = False,
        max_width: int = 1280,
        max_height: int = 720,
    ) -> np.ndarray:
        """Return projected LiDAR points overlaid on image or edge map (BGR)."""
        if frame_id not in self.images:
            raise KeyError(f"No image for frame_id={frame_id}")

        base = (
            self.edge_maps[frame_id]
            if (show_edges and frame_id in self.edge_maps)
            else self.images[frame_id]
        )
        h, w = base.shape
        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        if projected_pixels is not None and len(projected_pixels) > 0:
            pts = projected_pixels.astype(np.int32)
            mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            pts = pts[mask]
            for (u, v) in pts:
                cv2.circle(overlay, (int(u), int(v)), 2, (0, 255, 0), -1)

        oh, ow = overlay.shape[:2]
        scale = min(max_width / ow, max_height / oh, 1.0)
        if scale < 1.0:
            overlay = cv2.resize(
                overlay, (int(ow * scale), int(oh * scale)), interpolation=cv2.INTER_AREA
            )
        return overlay
