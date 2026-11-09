#!/usr/bin/env python3
"""Measure how many edge points LiDAR densification adds per frame.

For each LiDAR scan we count the edge points of that scan alone
(``extract_edges(project_to_range_image(points))``). We then run
``densify_lidar_scans`` once and take the size of each densified frame.
``densify_lidar_scans`` already returns the stacked (prev + curr + next) edge
points, so the dense count is just ``dense["points"].shape[0]``; edges are not
extracted a second time.

Frames are matched by timestamp, and the per-frame ratio dense_n / single_n
is summarized. Note that the ratio counts points only. Warping prev/next
edges into the current frame does not remove duplicates, so the ratio does
not measure new scene coverage.

Usage:
    python scripts/measure_densify.py --mcap huntington.mcap
    python scripts/measure_densify.py --mcap huntington.mcap --max-scans 50
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calib.imu import IMUPreintegrator  # noqa: E402
from calib.lidar import densify_lidar_scans, extract_edges, project_to_range_image  # noqa: E402

DEFAULT_LIDAR_TOPIC = "/ouster/points"
DEFAULT_IMU_TOPIC = "/vectornav/imu_uncompensated"


def _scan_time(scan: Dict) -> float:
    return float(scan["timestamp_sec"] if "timestamp_sec" in scan else scan["timestamp"])


def single_edge_counts(
    lidar_scans: Sequence[Dict],
    height: int = 128,
    width: int = 1024,
    fov_up: float = 22.5,
    fov_down: float = -22.5,
    edge_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, edge counts) for each scan taken on its own."""
    ts = np.empty(len(lidar_scans), dtype=np.float64)
    counts = np.empty(len(lidar_scans), dtype=np.int64)
    for i, scan in enumerate(lidar_scans):
        grid = project_to_range_image(
            scan["points"], height=height, width=width, fov_up=fov_up, fov_down=fov_down
        )
        ts[i] = _scan_time(scan)
        counts[i] = extract_edges(grid, threshold=edge_threshold).shape[0]
    return ts, counts


def dense_point_counts(dense_scans: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, point counts) for densify_lidar_scans output."""
    ts = np.array([float(d["timestamp"]) for d in dense_scans], dtype=np.float64)
    counts = np.array([int(d["points"].shape[0]) for d in dense_scans], dtype=np.int64)
    return ts, counts


def densify_edge_ratios(
    single_ts: np.ndarray,
    single_counts: np.ndarray,
    dense_ts: np.ndarray,
    dense_counts: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match dense frames to single scans by timestamp.

    Returns (single_n, dense_n, ratio) for the matched pairs. Dense frames with
    no single scan within ``tol`` seconds are dropped, and so are pairs with
    single_n == 0.
    """
    single_ts = np.asarray(single_ts, dtype=np.float64)
    single_counts = np.asarray(single_counts, dtype=np.int64)
    dense_ts = np.asarray(dense_ts, dtype=np.float64)
    dense_counts = np.asarray(dense_counts, dtype=np.int64)

    s_out: List[int] = []
    d_out: List[int] = []
    if single_ts.size and dense_ts.size:
        order = np.argsort(single_ts)
        st = single_ts[order]
        sc = single_counts[order]
        for t, dn in zip(dense_ts, dense_counts):
            j = int(np.searchsorted(st, t))
            best = None
            for k in (j - 1, j):
                if 0 <= k < st.size and abs(st[k] - t) <= tol:
                    if best is None or abs(st[k] - t) < abs(st[best] - t):
                        best = k
            if best is None or sc[best] == 0:
                continue
            s_out.append(int(sc[best]))
            d_out.append(int(dn))

    s_arr = np.array(s_out, dtype=np.int64)
    d_arr = np.array(d_out, dtype=np.int64)
    ratio = d_arr / s_arr if s_arr.size else np.empty(0, dtype=np.float64)
    return s_arr, d_arr, ratio


def summarize(single_n: np.ndarray, dense_n: np.ndarray, ratio: np.ndarray) -> Dict:
    """Summary statistics for the matched pairs."""
    n = int(ratio.size)
    if n == 0:
        return {
            "n_pairs": 0,
            "ratio_mean": None,
            "ratio_median": None,
            "ratio_std": None,
            "ratio_p10": None,
            "ratio_p90": None,
            "mean_single_n": None,
            "mean_dense_n": None,
        }
    return {
        "n_pairs": n,
        "ratio_mean": float(np.mean(ratio)),
        "ratio_median": float(np.median(ratio)),
        "ratio_std": float(np.std(ratio)),
        "ratio_p10": float(np.percentile(ratio, 10)),
        "ratio_p90": float(np.percentile(ratio, 90)),
        "mean_single_n": float(np.mean(single_n)),
        "mean_dense_n": float(np.mean(dense_n)),
    }


def measure(
    lidar_scans: Sequence[Dict],
    imu_measurements: Sequence[Dict],
    imu_integrator: Optional[IMUPreintegrator] = None,
) -> Dict:
    """Run the full measurement in memory (no MCAP needed)."""
    s_ts, s_n = single_edge_counts(lidar_scans)
    dense = densify_lidar_scans(list(lidar_scans), list(imu_measurements), imu_integrator=imu_integrator)
    d_ts, d_n = dense_point_counts(dense)
    del dense
    single_n, dense_n, ratio = densify_edge_ratios(s_ts, s_n, d_ts, d_n)
    out = summarize(single_n, dense_n, ratio)
    out["n_scans"] = int(len(lidar_scans))
    out["n_dense_frames"] = int(d_ts.size)
    return out


class _StopExtraction(Exception):
    pass


def load_scans(
    mcap_path: str,
    lidar_topic: str,
    imu_topic: str,
    max_scans: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Load LiDAR + IMU only (no camera). Stops early once ``max_scans`` is reached."""
    from calib.extract import McapTimeWindowExtractor

    class _Extractor(McapTimeWindowExtractor):
        def _process_lidar(self, msg, timestamp_sec, timestamp_ns):
            if max_scans is not None and len(self.lidar_scans) >= max_scans:
                raise _StopExtraction
            super()._process_lidar(msg, timestamp_sec, timestamp_ns)

    ex = _Extractor(mcap_path=mcap_path, camera_topic=None, lidar_topic=lidar_topic, imu_topic=imu_topic)
    try:
        # The extractor prints progress to stdout; send it to stderr so stdout is only JSON.
        with contextlib.redirect_stdout(sys.stderr):
            ex.extract()
    except _StopExtraction:
        print(f"Stopped after {len(ex.lidar_scans)} LiDAR scans (--max-scans).", file=sys.stderr)
    # Drop fields the measurement does not use.
    scans = [{"timestamp_sec": s["timestamp_sec"], "points": s["points"]} for s in ex.lidar_scans]
    imu = ex.imu_measurements
    imu.sort(key=lambda m: m["timestamp_sec"])
    scans.sort(key=lambda s: s["timestamp_sec"])
    return scans, imu


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcap", default=os.path.join(ROOT, "huntington.mcap"))
    p.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    p.add_argument("--imu-topic", default=DEFAULT_IMU_TOPIC)
    p.add_argument("--max-scans", type=int, default=None, help="Only use the first N LiDAR scans (for smoke tests).")
    p.add_argument("--json-out", default=None, help="Also write the JSON summary to this file.")
    args = p.parse_args(argv)

    if not os.path.exists(args.mcap):
        print(f"MCAP not found: {args.mcap}", file=sys.stderr)
        return 2

    scans, imu = load_scans(args.mcap, args.lidar_topic, args.imu_topic, args.max_scans)
    summary = measure(scans, imu, IMUPreintegrator(gravity=np.array([0.0, 0.0, 9.81])))
    summary["mcap"] = os.path.basename(args.mcap)
    summary["max_scans"] = args.max_scans

    print("\nDensification edge-count ratio (dense_n / single_n)", file=sys.stderr)
    print(f"  scans={summary['n_scans']}  dense frames={summary['n_dense_frames']}  pairs={summary['n_pairs']}", file=sys.stderr)
    if summary["n_pairs"]:
        print(
            f"  mean={summary['ratio_mean']:.3f}  median={summary['ratio_median']:.3f}  "
            f"std={summary['ratio_std']:.3f}  p10={summary['ratio_p10']:.3f}  p90={summary['ratio_p90']:.3f}",
            file=sys.stderr,
        )
        print(
            f"  mean single_n={summary['mean_single_n']:.1f}  mean dense_n={summary['mean_dense_n']:.1f}",
            file=sys.stderr,
        )

    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
