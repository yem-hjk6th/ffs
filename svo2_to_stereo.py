"""
svo2_to_stereo.py  —  从 ZED SVO2 文件提取左右立体图像帧 (PNG 序列 或 视频)

用法:
  # 提取为 PNG 序列 (默认)
  python svo2_to_stereo.py --svo path/to/recording.svo2

  # 提取为左右视频
  python svo2_to_stereo.py --svo path/to/recording.svo2 --mode video

  # 指定帧范围 + 输出目录
  python svo2_to_stereo.py --svo path/to/recording.svo2 --start 100 --end 500 --out_dir my_output

  # 同时导出 K.txt (相机内参)
  python svo2_to_stereo.py --svo path/to/recording.svo2 --export_K
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed (ZED SDK Python API) 未安装。")
    print("请先安装 ZED SDK: https://www.stereolabs.com/developers/release")
    print("然后运行: python -m pip install pyzed (或从 ZED SDK 安装目录安装)")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="从 SVO2 提取立体图像对")
    parser.add_argument("--svo", required=True, help="SVO2 文件路径")
    parser.add_argument("--out_dir", default=None, help="输出目录 (默认: svo文件同级目录/stereo_extract)")
    parser.add_argument("--mode", choices=["frames", "video", "sbs_video"], default="frames",
                        help="frames=PNG序列, video=左右MP4, sbs_video=并排MP4")
    parser.add_argument("--start", type=int, default=0, help="起始帧 (0-based)")
    parser.add_argument("--end", type=int, default=-1, help="结束帧 (-1=全部)")
    parser.add_argument("--skip", type=int, default=1, help="每N帧取一帧")
    parser.add_argument("--export_K", action="store_true", help="导出相机内参文件 K.txt")
    parser.add_argument("--resolution", choices=["native", "hd720", "vga"], default="native",
                        help="输出分辨率")
    return parser.parse_args()


def main():
    args = parse_args()

    svo_path = Path(args.svo).resolve()
    if not svo_path.exists():
        print(f"ERROR: SVO 文件不存在: {svo_path}")
        sys.exit(1)

    # 输出目录
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = svo_path.parent / "stereo_extract"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 打开 SVO
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # 不需要 ZED 的深度，我们用 FFS
    input_type = sl.InputType()
    input_type.set_from_svo_file(str(svo_path))
    init_params.input = input_type

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: 无法打开 SVO: {status}")
        sys.exit(1)

    cam_info = zed.get_camera_information()
    cam_config = cam_info.camera_configuration
    cal_params = cam_config.calibration_parameters

    width = cam_config.resolution.width
    height = cam_config.resolution.height
    fps = float(cam_config.fps) if cam_config.fps > 0 else 30.0
    total_frames = zed.get_svo_number_of_frames()

    print(f"SVO: {svo_path.name}")
    print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

    # 导出 K.txt
    if args.export_K:
        left_cam = cal_params.left_cam
        fx = left_cam.fx
        fy = left_cam.fy
        cx = left_cam.cx
        cy = left_cam.cy
        baseline = abs(cal_params.get_camera_baseline())  # 单位: mm → 转为 m
        baseline_m = baseline / 1000.0

        K_str = f"{fx} 0.0 {cx} 0.0 {fy} {cy} 0.0 0.0 1.0"
        k_path = out_dir / "K.txt"
        with open(k_path, "w") as f:
            f.write(K_str + "\n")
            f.write(f"{baseline_m}\n")
        print(f"内参已保存: {k_path}")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"  baseline={baseline_m:.6f} m")

    # 帧范围
    start = max(0, args.start)
    end = args.end if args.end >= 0 else total_frames - 1
    end = min(end, total_frames - 1)
    print(f"提取帧范围: [{start}, {end}], skip={args.skip}")

    runtime_params = sl.RuntimeParameters()
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    # 根据模式准备
    left_writer = None
    right_writer = None
    sbs_writer = None

    if args.mode == "video":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / args.skip
        left_writer = cv2.VideoWriter(str(out_dir / "left.mp4"), fourcc, out_fps, (width, height))
        right_writer = cv2.VideoWriter(str(out_dir / "right.mp4"), fourcc, out_fps, (width, height))
        print(f"输出: left.mp4 + right.mp4 -> {out_dir}")
    elif args.mode == "sbs_video":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / args.skip
        sbs_writer = cv2.VideoWriter(str(out_dir / "sbs_stereo.mp4"), fourcc, out_fps, (width * 2, height))
        print(f"输出: sbs_stereo.mp4 -> {out_dir}")
    else:
        left_dir = out_dir / "left"
        right_dir = out_dir / "right"
        left_dir.mkdir(exist_ok=True)
        right_dir.mkdir(exist_ok=True)
        print(f"输出: left/ + right/ PNG 序列 -> {out_dir}")

    zed.set_svo_position(start)
    extracted = 0
    frame_idx = start

    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        frame_idx = zed.get_svo_position()
        if frame_idx > end:
            break

        if (frame_idx - start) % args.skip != 0:
            continue

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

        left_img = left_mat.get_data()[:, :, :3]
        right_img = right_mat.get_data()[:, :, :3]

        # ZED 返回 BGRA, 取前3通道就是 BGR
        left_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
        right_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGBA2BGR)

        if args.mode == "frames":
            cv2.imwrite(str(left_dir / f"{frame_idx:06d}.png"), left_bgr)
            cv2.imwrite(str(right_dir / f"{frame_idx:06d}.png"), right_bgr)
        elif args.mode == "video":
            left_writer.write(left_bgr)
            right_writer.write(right_bgr)
        elif args.mode == "sbs_video":
            sbs = np.concatenate([left_bgr, right_bgr], axis=1)
            sbs_writer.write(sbs)

        extracted += 1
        if extracted % 100 == 0:
            print(f"  已提取 {extracted} 帧 (当前帧 {frame_idx}/{end})")

    # 清理
    if left_writer:
        left_writer.release()
    if right_writer:
        right_writer.release()
    if sbs_writer:
        sbs_writer.release()
    zed.close()

    print(f"\n完成! 共提取 {extracted} 帧")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
