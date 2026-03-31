"""
run_depth_video.py  —  对立体视频运行 Fast-FoundationStereo 深度估计

支持三种输入模式:
  1. 并排 (SBS) 立体视频 —— 一个文件，左右画面水平拼接
  2. 独立左右视频文件
  3. 左右图像序列文件夹 (当作视频逐帧处理)

用法:
  # 并排立体视频
  python run_depth_video.py --sbs_video path/to/sbs.mp4

  # 独立左右视频
  python run_depth_video.py --left_video left.mp4 --right_video right.mp4

  # 图像序列 (left/ right/ 文件夹)
  python run_depth_video.py --left_dir frames/left --right_dir frames/right

  # 带内参 + 快速模式
  python run_depth_video.py --sbs_video sbs.mp4 --intrinsic_file K.txt --scale 0.5 --valid_iters 4

  # 不显示预览窗口 (无头模式)
  python run_depth_video.py --sbs_video sbs.mp4 --no_display
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
    return model, OmegaConf.create(cfg)


def process_frame(model, img0, img1, valid_iters, padder_cache):
    H, W = img0.shape[:2]
    t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    if padder_cache.get("padder") is None or padder_cache.get("shape") != t0.shape:
        padder_cache["padder"] = InputPadder(t0.shape, divis_by=32, force_square=False)
        padder_cache["shape"] = t0.shape
    padder = padder_cache["padder"]

    t0, t1 = padder.pad(t0, t1)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
    vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    return disp, vis


def main():
    parser = argparse.ArgumentParser(description="FFS 立体深度估计 (视频)")

    # 输入 (三选一)
    parser.add_argument("--sbs_video", type=str, help="并排立体视频路径")
    parser.add_argument("--left_video", type=str, help="左视频路径")
    parser.add_argument("--right_video", type=str, help="右视频路径")
    parser.add_argument("--left_dir", type=str, help="左图序列文件夹")
    parser.add_argument("--right_dir", type=str, help="右图序列文件夹")

    # 模型
    parser.add_argument("--model_dir", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)

    # 处理
    parser.add_argument("--scale", type=float, default=1.0, help="缩放因子 (0.5=降一半)")
    parser.add_argument("--skip_frames", type=int, default=1, help="每N帧取一帧")
    parser.add_argument("--max_frames", type=int, default=0, help="最多处理帧数 (0=全部)")

    # 输出
    parser.add_argument("--out_dir", type=str, default="output_depth_video")
    parser.add_argument("--intrinsic_file", type=str, default=None, help="K.txt 路径 (可选)")
    parser.add_argument("--save_depth_npy", action="store_true", help="保存每帧 depth npy")
    parser.add_argument("--no_display", action="store_true", help="不显示预览窗口")

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # ── 确定输入模式 ──
    if args.sbs_video:
        mode = "sbs"
    elif args.left_video and args.right_video:
        mode = "separate_video"
    elif args.left_dir and args.right_dir:
        mode = "image_seq"
    else:
        parser.error("请提供: --sbs_video, 或 --left_video + --right_video, 或 --left_dir + --right_dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型 ──
    logging.info("加载模型...")
    model, cfg = load_model(args.model_dir, args.valid_iters, args.max_disp)
    logging.info("模型加载完成")

    # ── 加载内参 (可选) ──
    K = None
    baseline = None
    if args.intrinsic_file:
        with open(args.intrinsic_file, "r") as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].strip().split()))).astype(np.float32).reshape(3, 3)
            baseline = float(lines[1])
        K[:2] *= args.scale

    # ── 准备帧迭代器 ──
    if mode == "image_seq":
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        left_files = sorted([f for f in Path(args.left_dir).iterdir() if f.suffix.lower() in exts])
        right_dir = Path(args.right_dir)
        frame_pairs = []
        for lf in left_files:
            rf = right_dir / lf.name
            if rf.exists():
                frame_pairs.append((lf, rf))
        total_frames = len(frame_pairs)
        fps = 30.0
        logging.info(f"图像序列模式: {total_frames} 帧")
    else:
        if mode == "sbs":
            cap_l = cv2.VideoCapture(args.sbs_video)
            cap_r = None
        else:
            cap_l = cv2.VideoCapture(args.left_video)
            cap_r = cv2.VideoCapture(args.right_video)

        if not cap_l.isOpened():
            logging.error("无法打开视频"); sys.exit(1)
        fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"视频: FPS={fps}, 总帧数={total_frames}")

    writer = None
    padder_cache = {}
    frame_idx = 0
    processed = 0

    logging.info("开始推理... (第一帧较慢，需要编译)")

    try:
        while True:
            # 读取帧
            if mode == "image_seq":
                if frame_idx >= len(frame_pairs):
                    break
                lf, rf = frame_pairs[frame_idx]
                img_left = cv2.imread(str(lf))
                img_right = cv2.imread(str(rf))
                if img_left is None or img_right is None:
                    frame_idx += 1
                    continue
                frame_idx += 1
            else:
                ret_l, frame_l = cap_l.read()
                if not ret_l:
                    break
                frame_idx += 1

                if mode == "sbs":
                    h, w = frame_l.shape[:2]
                    img_left = frame_l[:, :w // 2]
                    img_right = frame_l[:, w // 2:]
                else:
                    ret_r, frame_r = cap_r.read()
                    if not ret_r:
                        break
                    img_left = frame_l
                    img_right = frame_r

            # 跳帧
            if frame_idx % args.skip_frames != 0:
                continue

            # BGR → RGB
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

            # 缩放
            if args.scale != 1.0:
                img_left = cv2.resize(img_left, fx=args.scale, fy=args.scale, dsize=None)
                img_right = cv2.resize(img_right, dsize=(img_left.shape[1], img_left.shape[0]))

            H, W = img_left.shape[:2]

            # 推理
            disp, vis = process_frame(model, img_left, img_right, args.valid_iters, padder_cache)
            processed += 1

            # 合成输出帧
            output_frame = np.concatenate([img_left, vis], axis=1)

            # 初始化 writer
            if writer is None:
                out_path = str(out_dir / "depth_video.mp4")
                out_fps = fps / args.skip_frames
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, out_fps,
                                         (output_frame.shape[1], output_frame.shape[0]))
                logging.info(f"输出视频: {out_path}, {output_frame.shape[1]}x{output_frame.shape[0]}, {out_fps:.1f}fps")

            writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

            # 保存深度
            if args.save_depth_npy and K is not None and baseline is not None:
                depth = K[0, 0] * baseline / np.clip(disp, 1e-6, None)
                np.save(str(out_dir / f"depth_{frame_idx:06d}.npy"), depth.astype(np.float32))

            # 预览
            if not args.no_display:
                s = min(1280 / output_frame.shape[1], 720 / output_frame.shape[0])
                preview = cv2.resize(output_frame, (int(output_frame.shape[1] * s), int(output_frame.shape[0] * s)))
                cv2.imshow("FFS Depth", preview[:, :, ::-1])
                if cv2.waitKey(1) == 27:
                    logging.info("ESC 退出")
                    break

            logging.info(f"帧 {frame_idx}/{total_frames} (已处理 {processed})")

            if args.max_frames > 0 and processed >= args.max_frames:
                break

    finally:
        if mode != "image_seq":
            cap_l.release()
            if cap_r is not None:
                cap_r.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    logging.info(f"完成! 处理了 {processed} 帧, 输出: {out_dir}")


if __name__ == "__main__":
    main()
