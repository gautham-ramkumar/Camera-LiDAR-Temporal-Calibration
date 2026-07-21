"""Camera–LiDAR hybrid temporal sync (stamp-primary + paper-closer residual)."""

from .camera import CameraPipeline
from .dynamic_filter import apply_dynamic_filter_to_scans, filter_dynamic_points
from .extract import McapTimeWindowExtractor
from .extrinsic_refine import refine_extrinsics
from .geometry import (
    closest_frame_id,
    euler_to_R,
    lidar_points_to_pixels,
    make_T,
    project_cam_points_to_pixels,
    transform_lidar_to_cam,
)
from .hybrid_sync import (
    HybridSyncResult,
    HybridWindowResult,
    StampSyncStats,
    associate_camera_time,
    hybrid_windows_as_calib,
    run_hybrid_sync,
    stamp_sync_stats,
)
from .imu import IMUPreintegrator
from .lidar import densify_lidar_scans, extract_edges, project_to_range_image
from .lines import apply_line_dt_to_pipeline, image_line_mask
from .pose_warp import relative_motion, warp_scan_to_camera_time
from .scoring import (
    WindowCalibResult,
    blended_distances,
    mean_alignment_score,
    seed_tau_grid,
    tau_only_objective,
)
from .viz import (
    make_distance_overlay,
    make_overlay_image,
    make_temporal_comparison,
    plot_distance_histograms,
    stack_overlays_horizontal,
    temporal_alignment_panel,
)

__all__ = [
    "McapTimeWindowExtractor",
    "IMUPreintegrator",
    "CameraPipeline",
    "project_to_range_image",
    "extract_edges",
    "densify_lidar_scans",
    "euler_to_R",
    "make_T",
    "transform_lidar_to_cam",
    "project_cam_points_to_pixels",
    "lidar_points_to_pixels",
    "closest_frame_id",
    "blended_distances",
    "mean_alignment_score",
    "tau_only_objective",
    "seed_tau_grid",
    "WindowCalibResult",
    "run_hybrid_sync",
    "stamp_sync_stats",
    "associate_camera_time",
    "hybrid_windows_as_calib",
    "HybridSyncResult",
    "HybridWindowResult",
    "StampSyncStats",
    "relative_motion",
    "warp_scan_to_camera_time",
    "image_line_mask",
    "apply_line_dt_to_pipeline",
    "filter_dynamic_points",
    "apply_dynamic_filter_to_scans",
    "refine_extrinsics",
    "make_overlay_image",
    "make_distance_overlay",
    "make_temporal_comparison",
    "temporal_alignment_panel",
    "plot_distance_histograms",
    "stack_overlays_horizontal",
]
