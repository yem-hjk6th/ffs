# FFS — Fast-FoundationStereo Wrapper Scripts

## Original Repository
- **Repo**: https://github.com/NVlabs/Fast-FoundationStereo
- **Authors**: Bowen Wen, Shaurya Dewan, Stan Birchfield (NVIDIA)
- **Paper**: CVPR 2026, https://arxiv.org/abs/2512.11130
- **License**: [CC-BY-NC-SA-4.0](https://github.com/NVlabs/Fast-FoundationStereo/blob/main/LICENSE)

## What This Repo Contains
Wrapper scripts for offline 3D reconstruction of KUKA RSI-printed parts using ZED 2i stereo camera + FFS depth estimation. Requires the original `Fast-FoundationStereo/` repo cloned alongside this folder.

## Custom Scripts
| Script | Purpose |
|--------|---------|
| `reconstruct_svo.py` | SVO2 + RSI CSV + FFS → world-coord point cloud → Poisson mesh |
| `extract_depth_snapshots.py` | Depth map snapshots at video time percentiles |
| `run_depth_images.py` | Stereo PNG pairs → FFS disparity/depth |
| `run_depth_video.py` | Stereo video → FFS depth |
| `run_depth_svo.py` | SVO2 → FFS depth (direct) |
| `svo2_to_stereo.py` | SVO2 → left/right PNG extraction |
| `recon_config.json` | Unified JSON config for reconstruct_svo.py |
| `environment.yaml` | Conda environment spec (Python 3.12, PyTorch nightly cu128) |

## Key Modifications vs Original
- All scripts are **new additions** (not modifications of upstream code)
- Integration with ZED SDK (`pyzed`) for SVO2 stereo file reading
- Integration with KUKA RSI CSV for robot pose synchronization
- Hand-eye extrinsic transform (camera → gripper → world)
- JSON config system for batch reconstruction parameter control
- Depth output modes: tri-panel / depth-only / separate stereo+depth
