"""
run_depth_svo.py  —  一键从 SVO2 + Fast-FoundationStereo 生成深度视频

把「SVO2 提取」和「FFS 推理」合为一个脚本，无需先手动提取。
直接读 SVO2 → 逐帧取左右图 → FFS 深度估计 → 输出深度视频 + 可选 depth npy

用法:
  # 基本用法
  python run_depth_svo.py --svo path/to/recording.svo2

  # 指定帧范围, 快速模式
  python run_depth_svo.py --svo recording.svo2 --start 100 --end 500 --scale 0.5 --valid_iters 4

  # 保存每帧深度 + 不显示窗口
  python run_depth_svo.py --svo recording.svo2 --save_depth_npy --no_display

  # 使用最快的模型
  python run_depth_svo.py --svo recording.svo2 --model 20-30-48 --valid_iters 4
"""

import os, sys, argparse, logging, yaml
import cv2
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed 未安装。请安装 ZED SDK 后再运行此脚本。")
    print("如果你只有提取好的图像/视频，请使用 run_depth_video.py 或 run_depth_images.py")
    sys.exit(1)

# ── 定位 Fast-FoundationStereo ──
REPO_ROOT = Path(__file__).resolve().parent.parent / "Fast-FoundationStereo"
sys.path.insert(0, str(REPO_ROOT))

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity


WEIGHT_DIR = REPO_ROOT / "weights"


def load_model(model_name, valid_iters, max_disp):
    model_path = WEIGHT_DIR / model_name / "model_best_bp2_serialize.pth"
    cfg_path = WEIGHT_DIR / model_name / "cfg.yaml"
    if not model_path.exists():
        logging.error(f"模型不存在: {model_path}")
        sys.exit(1)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model


def process_frame(model, img0, img1, valid_iters, padder_cache):
    H, W = img0.shape[:2]
    t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

    if padder_cache.get("padder") is None or padder_cache.get("shape") != t0.shape:
        padder_cache["padder"] = InputPadder(t0.shape, divis_by=32, force_square=False)
        padder_cache["shape"] = t0.shape

    t0, t1 = padder_cache["padder"].pad(t0, t1)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True,
                             optimize_build_volume="pytorch1")
    disp = padder_cache["padder"].unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
    vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    return disp, vis


def main():
    parser = argparse.ArgumentParser(description="SVO2 → FFS 深度估计 (一键)")
    parser.add_argument("--svo", required=True, help="SVO2 文件路径")
    parser.add_argument("--model", type=str, default="23-36-37",
                        choices=["23-36-37", "20-26-39", "20-30-48"],
                        help="模型名称 (23-36-37=最精确, 20-30-48=最快)")
    parser.add_argument("--valid_iters", type=int, default=8, help="迭代次数 (4=快, 8=精)")
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--scale", type=float, default=1.0, help="缩放因子")
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=-1, help="结束帧 (-1=全部)")
    parser.add_argument("--skip_frames", type=int, default=1, help="每N帧取一帧")
    parser.add_argument("--max_frames", type=int, default=0, help="最多处理帧数 (0=无限)")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--save_depth_npy", action="store_true")
    parser.add_argument("--no_display", action="store_true")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    svo_path = Path(args.svo).resolve()
    if not svo_path.exists():
        logging.error(f"SVO 不存在: {svo_path}"); sys.exit(1)

    # 输出目录
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = svo_path.parent / f"ffs_depth_{svo_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 打开 SVO ──
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    input_type = sl.InputType()
    input_type.set_from_svo_file(str(svo_path))
    init_params.input = input_type

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logging.error(f"无法打开 SVO: {status}"); sys.exit(1)

    cam_info = zed.get_camera_information()
    cam_config = cam_info.camera_configuration
    cal_params = cam_config.calibration_parameters
    left_cam = cal_params.left_cam

    width = cam_config.resolution.width
    height = cam_config.resolution.height
    fps = float(cam_config.fps) if cam_config.fps > 0 else 30.0
    total_frames = zed.get_svo_number_of_frames()

    # 内参
    fx, fy, cx, cy = left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy
    baseline_m = abs(cal_params.get_camera_baseline()) / 1000.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    K_scaled = K.copy()
    K_scaled[:2] *= args.scale

    # 保存 K.txt
    k_path = out_dir / "K.txt"
    with open(k_path, "w") as f:
        f.write(f"{fx} 0.0 {cx} 0.0 {fy} {cy} 0.0 0.0 1.0\n")
        f.write(f"{baseline_m}\n")

    logging.info(f"SVO: {svo_path.name}")
    logging.info(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    logging.info(f"内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, baseline={baseline_m:.4f}m")
    logging.info(f"K.txt 已保存: {k_path}")

    # ── 加载模型 ──
    logging.info(f"加载模型: {args.model} ...")
    model = load_model(args.model, args.valid_iters, args.max_disp)
    logging.info("模型加载完成")

    # 帧范围
    start = max(0, args.start)
    end = args.end if args.end >= 0 else total_frames - 1
    end = min(end, total_frames - 1)

    runtime_params = sl.RuntimeParameters()
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    writer = None
    padder_cache = {}
    processed = 0

    zed.set_svo_position(start)
    logging.info(f"帧范围: [{start}, {end}], skip={args.skip_frames}")
    logging.info("开始推理... (第一帧较慢)")

    try:
        while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            frame_idx = zed.get_svo_position()
            if frame_idx > end:
                break
            if (frame_idx - start) % args.skip_frames != 0:
                continue

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

            img_left = left_mat.get_data()[:, :, :3]  # BGRA → BGR → RGB
            img_right = right_mat.get_data()[:, :, :3]
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGRA2RGB)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2RGB)

            if args.scale != 1.0:
                img_left = cv2.resize(img_left, fx=args.scale, fy=args.scale, dsize=None)
                img_right = cv2.resize(img_right, dsize=(img_left.shape[1], img_left.shape[0]))

            H, W = img_left.shape[:2]
            disp, vis = process_frame(model, img_left, img_right, args.valid_iters, padder_cache)
            processed += 1

            output_frame = np.concatenate([img_left, vis], axis=1)

            if writer is None:
                out_path = str(out_dir / "depth_video.mp4")
                out_fps = fps / args.skip_frames
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, out_fps,
                                         (output_frame.shape[1], output_frame.shape[0]))

            writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

            if args.save_depth_npy:
                depth = K_scaled[0, 0] * baseline_m / np.clip(disp, 1e-6, None)
                np.save(str(out_dir / f"depth_{frame_idx:06d}.npy"), depth.astype(np.float32))

            if not args.no_display:
                s = min(1280 / output_frame.shape[1], 720 / output_frame.shape[0])
                preview = cv2.resize(output_frame,
                                     (int(output_frame.shape[1] * s), int(output_frame.shape[0] * s)))
                cv2.imshow("FFS Depth (SVO)", preview[:, :, ::-1])
                if cv2.waitKey(1) == 27:
                    logging.info("ESC 退出")
                    break

            logging.info(f"帧 {frame_idx}/{end} (已处理 {processed})")

            if args.max_frames > 0 and processed >= args.max_frames:
                break

    finally:
        zed.close()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    logging.info(f"完成! 处理了 {processed} 帧, 输出: {out_dir}")


if __name__ == "__main__":
    main()
