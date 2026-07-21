# References for hybrid sync (Wang et al. 2207.10454 and related)

Paper PDF: [2207.10454.pdf](2207.10454.pdf)

## Adopted in this codebase

| Ref | Paper | Our use |
|-----|-------|---------|
| [4] Wu soft time sync | Software multi-rate association | `tau_stamp` nearest-neighbor stamps |
| [7] Castorena edge alignment | LiDAR↔camera edges | DT soft score residual |
| [19] Zhang line-based extrinsic | Line features + search | `lines.py`, stiff `extrinsic_refine.py` |
| [20] LSD | Image line segments | `lines.image_line_mask` (Canny fallback) |
| 2207.10454 itself | Pose warp Tv, dynamic NN filter | `pose_warp.py`, `dynamic_filter.py` |

## Upgrades over 2207.10454

- **Stamp-primary** sync — residual only near stamps (avoids ±100 ms score aliases)
- **Operational vs diagnostic** τ (`tau_sync` vs `tau_unconst`)
- **Stiff Te** (±2°, ±5 cm) instead of free multi-dim search
- Distance-colored overlays + alias health flags

## Optional later

| Ref | Idea |
|-----|------|
| [16][18] Taylor / Ishikawa | Motion-based τ / hand-eye style residual |
| [21] NDT | Registration health metric |
| [8][10][11][12] | Hardware sync (not software) |
