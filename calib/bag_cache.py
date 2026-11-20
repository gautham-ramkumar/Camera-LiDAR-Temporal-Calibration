"""One-pass MCAP cache for the joint calibration.

Unlike :class:`calib.extract.McapTimeWindowExtractor`, this keeps the Ouster
organized layout and the per-point ``t`` field (ns after the scan header
stamp), which per-point deskew needs. Everything is written under
``cache_dir`` so later runs never re-read the 4.5 GB bag:

* ``index.npz``           camera / LiDAR header stamps, IMU, GPS speed, exposure
* ``scans/<k>.npz``       organized xyz (H, W, 3) float32 + t (ns) + range (m)
* ``images/<j>.png``      grayscale cam0 frame (Bayer RGGB demosaiced)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

CAM_TOPIC = "/cam_sync/cam0/image_raw/compressed"
CAM_META_TOPIC = "/cam_sync/cam0/meta"
LIDAR_TOPIC = "/ouster/points"
IMU_TOPIC = "/vectornav/imu_uncompensated"
GPS_VEL_TOPIC = "/ublox_gps_node/fix_velocity"

CACHE_VERSION = 1

_PC2 = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def _stamp(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def parse_organized_cloud(msg) -> Dict[str, np.ndarray]:
    """Organized Ouster cloud -> xyz (H,W,3), t_ns (H,W) uint32, range_m (H,W)."""
    h, w, step = int(msg.height), int(msg.width), int(msg.point_step)
    n = h * w
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    fields = {f.name: f for f in msg.fields}

    def col(name: str) -> np.ndarray:
        f = fields[name]
        return np.ndarray((n,), dtype=_PC2[int(f.datatype)], buffer=data,
                          offset=int(f.offset), strides=(step,)).copy()

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32).reshape(h, w, 3)
    t_ns = col("t").astype(np.uint32).reshape(h, w)
    if "range" in fields:
        rng = col("range").astype(np.float64).reshape(h, w)
        pos = rng[rng > 0]
        if pos.size and np.median(pos) > 200:  # driver reports millimetres
            rng = rng * 1e-3
    else:
        rng = np.linalg.norm(xyz, axis=2).astype(np.float64)
    return {"xyz": xyz, "t_ns": t_ns, "range": rng.astype(np.float32)}


def decode_gray(msg) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode camera frame")
    if img.ndim == 2 and "bayer_rggb" in str(getattr(msg, "format", "")):
        img = cv2.cvtColor(img, cv2.COLOR_BayerRGGB2BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def build_cache(mcap_path: str | Path, cache_dir: str | Path, verbose: bool = True) -> Path:
    """Stream the bag once and write the cache. Returns the cache directory."""
    cache_dir = Path(cache_dir)
    mcap_path = Path(mcap_path)
    meta_path = cache_dir / "meta.json"
    st = mcap_path.stat()
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("version") == CACHE_VERSION and meta.get("size") == st.st_size:
            return cache_dir
    (cache_dir / "scans").mkdir(parents=True, exist_ok=True)
    (cache_dir / "images").mkdir(parents=True, exist_ok=True)

    cam_t, lid_t = [], []
    imu_t, gyro, acc = [], [], []
    gps_t, gps_v = [], []
    meta_t, exposure_us = [], []
    with mcap_path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        topics = [CAM_TOPIC, CAM_META_TOPIC, LIDAR_TOPIC, IMU_TOPIC, GPS_VEL_TOPIC]
        for _s, ch, _m, msg in reader.iter_decoded_messages(topics=topics):
            tp = ch.topic
            if tp == CAM_TOPIC:
                j = len(cam_t)
                cv2.imwrite(str(cache_dir / "images" / f"{j:05d}.png"), decode_gray(msg))
                cam_t.append(_stamp(msg))
            elif tp == LIDAR_TOPIC:
                k = len(lid_t)
                c = parse_organized_cloud(msg)
                np.savez(cache_dir / "scans" / f"{k:05d}.npz", **c)
                lid_t.append(_stamp(msg))
                if verbose and k % 100 == 0:
                    print(f"  scan {k}  cam {len(cam_t)}", flush=True)
            elif tp == IMU_TOPIC:
                imu_t.append(_stamp(msg))
                g, a = msg.angular_velocity, msg.linear_acceleration
                gyro.append((g.x, g.y, g.z))
                acc.append((a.x, a.y, a.z))
            elif tp == GPS_VEL_TOPIC:
                gps_t.append(_stamp(msg))
                v = msg.twist.twist.linear
                gps_v.append(float(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)))
            elif tp == CAM_META_TOPIC:
                meta_t.append(_stamp(msg))
                exposure_us.append(float(msg.exposure_time))

    np.savez(
        cache_dir / "index.npz",
        cam_t=np.asarray(cam_t), lidar_t=np.asarray(lid_t),
        imu_t=np.asarray(imu_t), gyro=np.asarray(gyro), acc=np.asarray(acc),
        gps_t=np.asarray(gps_t), gps_speed=np.asarray(gps_v),
        meta_t=np.asarray(meta_t), exposure_us=np.asarray(exposure_us),
    )
    meta_path.write_text(json.dumps({"version": CACHE_VERSION, "size": st.st_size,
                                     "n_cam": len(cam_t), "n_lidar": len(lid_t)}))
    return cache_dir


class BagCache:
    """Lazy reader for a cache written by :func:`build_cache`."""

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        idx = np.load(self.dir / "index.npz")
        self.index = {k: idx[k] for k in idx.files}
        self.cam_t = self.index["cam_t"]
        self.lidar_t = self.index["lidar_t"]

    def scan(self, k: int) -> Dict[str, np.ndarray]:
        d = np.load(self.dir / "scans" / f"{int(k):05d}.npz")
        return {name: d[name] for name in d.files}

    def image(self, j: int) -> np.ndarray:
        img = cv2.imread(str(self.dir / "images" / f"{int(j):05d}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"cached image {j} missing")
        return img


def valid_point_times(t_ns: np.ndarray, scan_period_s: float = 0.1, margin_s: float = 0.005) -> np.ndarray:
    """Mask of per-point times that fall inside one sweep.

    One scan in huntington.mcap (≈53.7 s) has 56 points whose ``t`` reaches
    4.29 s; those are rejected here rather than clipped, since clipping
    would assign them a wrong capture time.
    """
    t = np.asarray(t_ns, dtype=np.float64) * 1e-9
    return (t >= 0.0) & (t <= scan_period_s + margin_s)


def load_cache(mcap_path: str | Path, cache_dir: Optional[str | Path] = None) -> BagCache:
    mcap_path = Path(mcap_path)
    if cache_dir is None:
        cache_dir = mcap_path.parent / ".calib_cache" / "joint_v1"
    build_cache(mcap_path, cache_dir)
    return BagCache(cache_dir)
