"""
run_depth_images.py  —  对一组左右立体图像对运行 Fast-FoundationStereo 深度估计

用法:
  # 单对图像
  python run_depth_images.py --left img_left.png --right img_right.png

  # 图像序列文件夹 (left/ 和 right/ 下同名文件)
  python run_depth_images.py --left_dir frames/left --right_dir frames/right

  # 带内参 → 输出真实深度 (米)
  python run_depth_images.py --left_dir left/ --right_dir right/ --intrinsic_file K.txt

  # 快速模式
  python run_depth_images.py --left img_l.png --right img_r.png --scale 0.5 --valid_iters 4
"""

import os, sys, argparse, logging, yaml
import cv2
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# ── 定位 Fast-FoundationStereo ──
REPO_ROOT = Path(__file__).resolve().parent.parent / "Fast-FoundationStereo"
sys.path.insert(0, str(REPO_ROOT))

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity


DEFAULT_WEIGHTS = str(REPO_ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth")


def load_model(model_path, valid_iters, max_disp):
    cfg_path = os.path.join(os.path.dirname(model_path), "cfg.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model, cfg


def run_stereo(model, img_left, img_right, valid_iters, scale=1.0):
    """输入 RGB numpy HWC, 返回 disparity (H,W) 和可视化 (H,W,3)"""
    if scale != 1.0:
        img_left = cv2.resize(img_left, fx=scale, fy=scale, dsize=None)
        img_right = cv2.resize(img_right, dsize=(img_left.shape[1], img_left.shape[0]))

    H, W = img_left.shape[:2]

    t0 = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)

    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
    vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    return disp, vis


def main():
    parser = argparse.ArgumentParser(description="FFS 立体深度估计 (图像)")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--left", type=str, help="单张左图路径")
    g.add_argument("--left_dir", type=str, help="左图文件夹")
    parser.add_argument("--right", type=str, help="单张右图路径")
    parser.add_argument("--right_dir", type=str, help="右图文件夹")
    parser.add_argument("--intrinsic_file", type=str, default=None, help="K.txt 路径")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out_dir", type=str, default="output_depth")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--save_npy", action="store_true", help="保存 depth_meter.npy")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集图像对
    pairs = []
    if args.left:
        if not args.right:
            parser.error("使用 --left 时必须同时提供 --right")
        pairs.append((args.left, args.right))
    else:
        if not args.right_dir:
            parser.error("使用 --left_dir 时必须同时提供 --right_dir")
        left_dir = Path(args.left_dir)
        right_dir = Path(args.right_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        left_files = sorted([f for f in left_dir.iterdir() if f.suffix.lower() in exts])
        for lf in left_files:
            rf = right_dir / lf.name
            if rf.exists():
                pairs.append((str(lf), str(rf)))
        logging.info(f"找到 {len(pairs)} 个图像对")

    if not pairs:
        logging.error("没有找到图像对!"); sys.exit(1)

    # 加载内参
    K = None
    baseline = None
    if args.intrinsic_file:
        with open(args.intrinsic_file, "r") as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].strip().split()))).astype(np.float32).reshape(3, 3)
            baseline = float(lines[1])
        K[:2] *= args.scale
        logging.info(f"内参已加载, baseline={baseline:.4f}m")

    # 加载模型
    logging.info("加载模型...")
    model, cfg = load_model(args.model_dir, args.valid_iters, args.max_disp)
    logging.info("模型加载完成")

    for i, (lp, rp) in enumerate(pairs):
        name = Path(lp).stem
        logging.info(f"[{i+1}/{len(pairs)}] {name}")

        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            logging.warning(f"  跳过: 无法读取 {lp} 或 {rp}")
            continue

        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        disp, vis = run_stereo(model, img_l, img_r, args.valid_iters, args.scale)

        # 保存可视化
        combined = np.concatenate([img_l, vis], axis=1)
        cv2.imwrite(str(out_dir / f"{name}_disp.png"), combined[:, :, ::-1])

        # 保存深度
        if K is not None and baseline is not None and args.save_npy:
            depth = K[0, 0] * baseline / np.clip(disp, 1e-6, None)
            np.save(str(out_dir / f"{name}_depth.npy"), depth.astype(np.float32))

    logging.info(f"完成! 输出: {out_dir}")


if __name__ == "__main__":
    main()
