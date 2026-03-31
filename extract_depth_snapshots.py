"""
extract_depth_snapshots.py — 在视频指定时间分位点输出 FFS 深度图

用法:
  python extract_depth_snapshots.py ^
      --svo ../../rsi_printing/recorded_data/20260324_174844/recording_20260324_174844.svo2 ^
      --out_dir ../../rsi_printing/recorded_data/20260324_174844/depth_snapshots
"""

import os, sys, argparse
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed 未安装。")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent / "Fast-FoundationStereo"
sys.path.insert(0, str(REPO_ROOT))

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity

WEIGHT_DIR = REPO_ROOT / "weights"


def load_ffs_model(model_name, valid_iters, max_disp):
    model_path = WEIGHT_DIR / model_name / "model_best_bp2_serialize.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"FFS 模型不存在: {model_path}")
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model


def ffs_disparity(model, img_left, img_right, valid_iters, padder_cache):
    H, W = img_left.shape[:2]
    t0 = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
    if padder_cache.get("padder") is None or padder_cache.get("shape") != t0.shape:
        padder_cache["padder"] = InputPadder(t0.shape, divis_by=32, force_square=False)
        padder_cache["shape"] = t0.shape
    t0p, t1p = padder_cache["padder"].pad(t0, t1)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(t0p, t1p, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")
    disp = padder_cache["padder"].unpad(disp.float())
    return disp.data.cpu().numpy().reshape(H, W).clip(0, None)


def depth_to_colormap(depth, depth_min=0.1, depth_max=2.0):
    """深度图 → 彩色可视化 (TURBO colormap)."""
    d = np.clip(depth, depth_min, depth_max)
    d_norm = ((d - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    # 无效区域设为 0
    invalid = ~np.isfinite(depth) | (depth < depth_min) | (depth > depth_max)
    d_norm[invalid] = 0
    colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    colored[invalid] = 0
    return colored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--svo", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="23-36-37")
    p.add_argument("--valid_iters", type=int, default=8)
    p.add_argument("--max_disp", type=int, default=192)
    p.add_argument("--depth_min", type=float, default=0.1)
    p.add_argument("--depth_max", type=float, default=2.0)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[1/6, 1/3, 1/2, 2/3, 5/6],
                   help="时间分位点 (默认 1/6 1/3 1/2 2/3 5/6)")
    args = p.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    svo_path = Path(args.svo).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 打开 SVO
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    input_type = sl.InputType()
    input_type.set_from_svo_file(str(svo_path))
    init_params.input = input_type

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"无法打开 SVO: {status}")
        sys.exit(1)

    cam_info = zed.get_camera_information()
    cal_params = cam_info.camera_configuration.calibration_parameters
    fx = cal_params.left_cam.fx
    baseline_m = abs(cal_params.get_camera_baseline()) / 1000.0
    total_frames = zed.get_svo_number_of_frames()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height

    print(f"SVO: {total_frames} frames, {width}x{height}, baseline={baseline_m:.4f}m")

    # 计算目标帧
    target_frames = []
    for frac in args.fractions:
        idx = int(round(frac * (total_frames - 1)))
        idx = max(0, min(idx, total_frames - 1))
        target_frames.append((frac, idx))
    print(f"目标帧: {[(f'{f:.3f}', i) for f, i in target_frames]}")

    # 加载模型
    print(f"加载 FFS 模型: {args.model} ...")
    model = load_ffs_model(args.model, args.valid_iters, args.max_disp)
    print("模型加载完成")

    left_mat, right_mat = sl.Mat(), sl.Mat()
    runtime_params = sl.RuntimeParameters()
    padder_cache = {}

    for frac, frame_idx in target_frames:
        print(f"\n--- 处理帧 {frame_idx} ({frac*100:.0f}% of video) ---")
        zed.set_svo_position(frame_idx)

        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            print(f"  无法读取帧 {frame_idx}, 跳过")
            continue

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
        img_left = cv2.cvtColor(left_mat.get_data()[:, :, :3], cv2.COLOR_BGRA2RGB)
        img_right = cv2.cvtColor(right_mat.get_data()[:, :, :3], cv2.COLOR_BGRA2RGB)

        # FFS 推理
        disp = ffs_disparity(model, img_left, img_right, args.valid_iters, padder_cache)
        depth = fx * baseline_m / np.clip(disp, 1e-6, None)

        # 保存 disparity 可视化
        disp_vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
        disp_vis_bgr = cv2.cvtColor(disp_vis, cv2.COLOR_RGB2BGR)

        # 保存 depth 可视化
        depth_vis = depth_to_colormap(depth, args.depth_min, args.depth_max)

        # 保存左图
        img_left_bgr = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)

        # 拼接: 左图 | 深度 | disparity
        # 先确保高度一致
        h = img_left_bgr.shape[0]
        combo = np.hstack([img_left_bgr, depth_vis, disp_vis_bgr])

        tag = f"{int(frac*6)}of6"  # e.g. "1of6", "2of6"
        pct = f"{int(frac*100):02d}pct"

        cv2.imwrite(str(out_dir / f"frame{frame_idx:04d}_{pct}_left.png"), img_left_bgr)
        cv2.imwrite(str(out_dir / f"frame{frame_idx:04d}_{pct}_depth.png"), depth_vis)
        cv2.imwrite(str(out_dir / f"frame{frame_idx:04d}_{pct}_disp.png"), disp_vis_bgr)
        cv2.imwrite(str(out_dir / f"frame{frame_idx:04d}_{pct}_combo.png"), combo)

        print(f"  已保存: frame{frame_idx:04d}_{pct}_*.png")
        print(f"  Disparity: min={disp.min():.1f}, max={disp.max():.1f}, median={np.median(disp):.1f}")
        print(f"  Depth: min={depth[depth>args.depth_min].min():.3f}m, "
              f"max={depth[depth<args.depth_max].max():.3f}m, "
              f"median={np.median(depth[np.isfinite(depth)]):.3f}m")

    zed.close()
    print(f"\n完成! 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
