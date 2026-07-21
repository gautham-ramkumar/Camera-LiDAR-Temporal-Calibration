# Camera–LiDAR Hybrid Temporal Sync

Stamp-primary temporal sync with paper-closer residual path (Wang et al. [arXiv:2207.10454](papers/2207.10454.pdf)), plus upgrades that avoid the paper’s soft-sync / free-search failure modes.

## Model

| Layer | Role |
|-------|------|
| `tau_stamp` | **Primary sync** — median nearest camera−LiDAR stamp |
| `tau_res` | Edge residual in ±50 ms around stamps (EMA), optional IMU warp + dynamic filter |
| `tau_sync` | **Operational** — \(t_C \approx t_L + \tau_{sync}\) |
| `tau_unconst` | **Diagnostic only** — unconstrained score peak / alias flag |
| `T_CL` | Stiff prior + optional slow refine (±2°, ±5 cm) |

## Example outputs

Notebook figures below follow the visual conventions of Wang et al. (arXiv:2207.10454) — grayscale image / Canny edges / distance transform triptychs, and uniform-green LiDAR points projected onto the raw image, rather than a distance-colored heatmap.

**Camera edges + distance transform** (§4 — `image | Canny edges | distance transform`, cf. Fig. 1b):

![Camera edges and distance transform](assets/cam.png)

**Projected LiDAR points across candidate offsets** (§8 — bad / stamp-primary / operational-sync comparison, cf. Fig. 1c / Fig. 8):

![Paper-style LiDAR projection comparison](assets/output.png)

**Per-window score / tau / residual convergence** (§6 — 15 sliding windows over 750 LiDAR scans, cf. Fig. 6):

![Hybrid sync score, tau, and residual evaluation](assets/calibration_eval.png)

## Layout

```
Calibration/
├── assets/                # README figures (edge/DT triptych, projection comparison, score/tau plots)
├── calib/
│   ├── extract.py, imu.py, lidar.py, camera.py
│   ├── geometry.py, scoring.py
│   ├── pose_warp.py, lines.py, dynamic_filter.py, extrinsic_refine.py
│   ├── hybrid_sync.py, viz.py
├── papers/2207.10454.pdf, REFERENCES.md
├── Camera_Lidar_Hybrid_Sync.ipynb
├── huntington.mcap
└── requirements.txt
```

## Setup

```bash
cd ~/Desktop/AFR/Calibration
pip install -r requirements.txt
jupyter notebook Camera_Lidar_Hybrid_Sync.ipynb
```

## Notebook flags

- `USE_POSE_WARP` — IMU \(T_v\) warp (paper-style)
- `USE_DYNAMIC_FILTER` — adjacent-frame NN dynamic removal
- `USE_LSD` — LSD lines (else Canny DT)
- `REFINE_EXTRINSICS` — slow stiff \(T_e\) refine
- `RESIDUAL_BAND` — default ±0.05 s around stamps
