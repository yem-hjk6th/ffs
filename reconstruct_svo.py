"""
reconstruct_svo.py — 离线后印重建: SVO2 + RSI CSV + FFS → 世界坐标点云 → mesh

完整流程:
  1. 读 SVO2 → 逐帧取左右图
  2. 加载 RSI CSV → 按时间戳匹配每帧的机器人位姿
  3. FFS 推理 → disparity → depth
  4. 反投影到相机坐标 → 乘外参T_cam2gripper → 乘RSI位姿T_gripper2base → 世界坐标
  5. 多帧点云融合 → voxel downsample → Poisson→ mesh.ply

用法:
  # 最简 (使用旧外参测试流程)
  python reconstruct_svo.py ^
      --svo  ../../rsi_printing/recorded_data/20260324_174844/recording_20260324_174844.svo2 ^
      --rsi  ../../rsi_printing/rsi_data/rsi_data_20260324_174844.csv ^
      --extrinsics ../../rsi_printing/build/Preparation/calibration/aut_cal/extrinsics_res/20260226_150931/extrinsics.txt

  # 指定采样率和帧范围
  python reconstruct_svo.py --svo ... --rsi ... --extrinsics ... ^
      --skip_frames 30 --start 100 --end 5000 --voxel_mm 3.0

  # 使用快速模型 + ROI 深度过滤
  python reconstruct_svo.py --svo ... --rsi ... --extrinsics ... ^
      --model 20-30-48 --valid_iters 4 --depth_min 0.1 --depth_max 1.5
"""

import os, sys, argparse, csv, json, logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed 未安装。请安装 ZED SDK。")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d 未安装。pip install open3d")
    sys.exit(1)

# ── 定位 Fast-FoundationStereo ──
REPO_ROOT = Path(__file__).resolve().parent.parent / "Fast-FoundationStereo"
sys.path.insert(0, str(REPO_ROOT))

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity

WEIGHT_DIR = REPO_ROOT / "weights"


# ═══════════════════════════════════════════════════════════
#  KUKA 位姿工具
# ═══════════════════════════════════════════════════════════

def euler_to_R(a_deg, b_deg, c_deg):
    """KUKA ZYX 欧拉角 (A,B,C) → 3x3 旋转矩阵 R_gripper2base.

    Convention: R = Rz(A) @ Ry(B) @ Rx(C)
    Transforms vectors FROM gripper frame TO base frame.
    """
    a, b, c = np.radians([a_deg, b_deg, c_deg])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


def pose_to_T(x_mm, y_mm, z_mm, a_deg, b_deg, c_deg):
    """RSI 位姿 → 4x4 T_gripper2base (齐次矩阵, 单位: 米)."""
    T = np.eye(4)
    T[:3, :3] = euler_to_R(a_deg, b_deg, c_deg)
    T[:3, 3] = np.array([x_mm, y_mm, z_mm]) / 1000.0
    return T


# ═══════════════════════════════════════════════════════════
#  RSI CSV 加载
# ═══════════════════════════════════════════════════════════

def load_rsi_csv(csv_path):
    """加载 RSI CSV → (timestamps, poses) numpy 数组.

    Returns:
        ts:    shape (N,), float64, Unix epoch 时间戳
        poses: shape (N, 6), float64, [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg]
    """
    ts_list, pose_list = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_list.append(float(row["timestamp"]))
            pose_list.append([
                float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]),
                float(row["a_deg"]), float(row["b_deg"]), float(row["c_deg"]),
            ])
    return np.array(ts_list, dtype=np.float64), np.array(pose_list, dtype=np.float64)


def interp_pose(ts_rsi, poses_rsi, t_query):
    """在 RSI 时序中线性插值得到 t_query 时刻的位姿.

    对于位置 (x, y, z): 直接线性插值
    对于角度 (a, b, c): 直接线性插值 (短时间内角度变化小，线性近似足够)

    Returns:
        (x_mm, y_mm, z_mm, a_deg, b_deg, c_deg) or None if out of range
    """
    if t_query < ts_rsi[0] or t_query > ts_rsi[-1]:
        return None
    idx = np.searchsorted(ts_rsi, t_query) - 1
    idx = max(0, min(idx, len(ts_rsi) - 2))
    t0, t1 = ts_rsi[idx], ts_rsi[idx + 1]
    dt = t1 - t0
    if dt < 1e-9:
        return poses_rsi[idx]
    alpha = (t_query - t0) / dt
    return poses_rsi[idx] * (1 - alpha) + poses_rsi[idx + 1] * alpha


# ═══════════════════════════════════════════════════════════
#  外参加载
# ═══════════════════════════════════════════════════════════

def load_extrinsics(path):
    """加载 T_cam2gripper 4x4 齐次矩阵 (来自 extrinsics.txt 或 .npz).

    支持两种格式:
      1. extrinsics.txt: 注释行以 # 开头, 4行x4列数字
      2. calibration_result.npz: 包含 T_cam2gripper 字段
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data["T_cam2gripper"]

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        raise ValueError(f"外参文件格式错误: 需要 4x4 矩阵, 得到 {len(rows)} 行")
    return np.array(rows, dtype=np.float64)


# ═══════════════════════════════════════════════════════════
#  FFS 推理
# ═══════════════════════════════════════════════════════════

def load_ffs_model(model_name, valid_iters, max_disp):
    model_path = WEIGHT_DIR / model_name / "model_best_bp2_serialize.pth"
    cfg_path = WEIGHT_DIR / model_name / "cfg.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"FFS 模型不存在: {model_path}")
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    return model


def ffs_disparity(model, img_left, img_right, valid_iters, padder_cache):
    """FFS 推理 → disparity (H, W) numpy array."""
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


# ═══════════════════════════════════════════════════════════
#  反投影: depth → 相机坐标 3D 点
# ═══════════════════════════════════════════════════════════

def backproject(depth, K, stride=1, depth_min=0.05, depth_max=3.0, color_img=None):
    """depth (H,W) + K (3x3) → points_cam (N,4 齐次) + colors (N,3) or None.

    Args:
        depth: 深度图 (米)
        K: 3x3 相机内参矩阵
        stride: 每隔 stride 个像素取一个点 (降采样)
        depth_min/max: 深度范围过滤 (米)
        color_img: RGB 图像 (H,W,3) uint8, 可选

    Returns:
        pts: (N, 4) float64, 齐次坐标 [X,Y,Z,1] in camera frame (meters)
        colors: (N, 3) float64 in [0,1], or None
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u = np.arange(0, W, stride, dtype=np.float32)
    v = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    d = depth[::stride, ::stride]
    valid = (d > depth_min) & (d < depth_max) & np.isfinite(d)
    d = d[valid]
    uu = uu[valid]
    vv = vv[valid]

    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d
    pts = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)  # (N, 4)

    colors = None
    if color_img is not None:
        c = color_img[::stride, ::stride]
        colors = c[valid].astype(np.float64) / 255.0

    return pts, colors


# ═══════════════════════════════════════════════════════════
#  ROI: 深度差分检测打印区域 (替代 SAM2)
# ═══════════════════════════════════════════════════════════

def compute_roi_mask(depth_ref, depth_cur, grow_threshold_m=0.003):
    """通过深度减少检测打印 ROI.

    打印体使深度减少 (物体越来越近), 阈值 3mm.

    Returns:
        mask: (H, W) bool, True = 打印区域
    """
    if depth_ref is None:
        return np.ones(depth_cur.shape, dtype=bool)
    diff = depth_ref - depth_cur  # 正值 = 物体长高
    valid = np.isfinite(diff)
    mask = valid & (diff > grow_threshold_m)
    # 形态学膨胀，填补空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=2)
    return mask_u8 > 0


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="离线后印重建: SVO2 + RSI + FFS → mesh")

    # 配置文件 (JSON, 字段名与命令行参数一致, 命令行参数优先级更高)
    p.add_argument("--config", type=str, default=None,
                   help="JSON 配置文件路径 (覆盖默认值, 命令行参数优先)")

    # 输入
    p.add_argument("--svo", required=True, help="SVO2 文件路径")
    p.add_argument("--rsi", required=True, help="RSI CSV 文件路径")
    p.add_argument("--extrinsics", default=None, help="T_cam2gripper 外参文件 (.txt 或 .npz)")

    # FFS 参数
    p.add_argument("--model", default="23-36-37",
                   choices=["23-36-37", "20-26-39", "20-30-48"])
    p.add_argument("--valid_iters", type=int, default=8)
    p.add_argument("--max_disp", type=int, default=192)

    # 帧选择
    p.add_argument("--start", type=int, default=0, help="起始帧索引")
    p.add_argument("--end", type=int, default=-1, help="结束帧索引 (-1=全部)")
    p.add_argument("--skip_frames", type=int, default=30, help="每N帧取一帧 (30=1fps)")
    p.add_argument("--max_frames", type=int, default=0, help="最多处理帧数 (0=无限)")

    # 时间同步
    p.add_argument("--time_offset", type=float, default=0.0,
                   help="SVO→RSI 时间偏移 (秒). "
                        "正值 = SVO 时间落后于 RSI. "
                        "可通过层边界对齐获得.")

    # 深度 / 点云
    p.add_argument("--depth_min", type=float, default=0.1, help="最小深度 (米)")
    p.add_argument("--depth_max", type=float, default=2.0, help="最大深度 (米)")
    p.add_argument("--point_stride", type=int, default=2, help="像素采样步长")
    p.add_argument("--voxel_mm", type=float, default=3.0, help="体素降采样尺寸 (mm)")

    # ROI — 世界坐标 bounding box (推荐，自动从 RSI 轨迹推算)
    p.add_argument("--use_bbox", action="store_true",
                   help="在世界坐标中用 bounding box 裁剪 (自动从 RSI 轨迹推算)")
    p.add_argument("--bbox_pad_mm", type=float, default=80.0,
                   help="RSI 轨迹外扩 padding (mm, 覆盖打印件+喷嘴偏移)")
    p.add_argument("--bbox_z_below_mm", type=float, default=30.0,
                   help="bbox 底部扩展 (mm, 覆盖基板以下)")
    p.add_argument("--bbox_min", type=float, nargs=3, default=None,
                   help="手动指定 bbox 最小角 (x_mm y_mm z_mm) 世界坐标")
    p.add_argument("--bbox_max", type=float, nargs=3, default=None,
                   help="手动指定 bbox 最大角 (x_mm y_mm z_mm) 世界坐标")

    # ROI — 旧方案: 深度差分 (仅适用于固定相机)
    p.add_argument("--use_roi", action="store_true",
                   help="[旧] 使用参考帧深度差分检测打印 ROI (相机固定时有效)")
    p.add_argument("--ref_frame", type=int, default=5,
                   help="参考帧索引 (打印前)")
    p.add_argument("--roi_threshold_mm", type=float, default=3.0,
                   help="ROI 深度变化阈值 (mm)")

    # 输出
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--poisson_depth", type=int, default=9)
    p.add_argument("--save_frames", action="store_true", help="保存每帧点云 PLY")
    p.add_argument("--save_depth_npy", action="store_true", help="保存每帧 depth npy")
    p.add_argument("--no_display", action="store_true")
    p.add_argument("--no_color", action="store_true", help="不保存颜色 (节省内存)")

    # 深度图 ROI 裁剪
    p.add_argument("--depth_roi_enabled", action="store_true", default=False,
                   help="是否启用深度图 ROI 裁剪 (默认关闭)")
    p.add_argument("--depth_roi", type=int, nargs=4, default=None,
                   help="深度图 ROI 裁剪 [x1 y1 x2 y2] 像素坐标")
    # 深度图输出模式: tri=左右目+深度三联, depth_only=仅深度图, sep_rgb_depth=左右目双拼+深度图分开保存
    p.add_argument("--depth_output_mode", type=str, default="depth_only",
                   choices=["tri", "depth_only", "sep_rgb_depth"],
                   help="深度图输出模式")

    # 先做一次解析拿 --config 路径, 再用 JSON 覆盖默认值后重新解析
    args_pre, _ = p.parse_known_args()
    if args_pre.config:
        cfg_path = Path(args_pre.config)
        if not cfg_path.exists():
            p.error(f"配置文件不存在: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 将 JSON 中的值设为 argparse 默认值 (命令行显式传参仍优先)
        cfg.pop("_comment", None)
        p.set_defaults(**cfg)
        logging.info(f"已加载配置: {cfg_path}")

    args = p.parse_args()
    if not args.extrinsics:
        p.error("--extrinsics 必须通过命令行或 config JSON 提供")
    return args


def main():
    args = parse_args()
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # ── 输入验证 ──
    svo_path = Path(args.svo).resolve()
    rsi_path = Path(args.rsi).resolve()
    ext_path = Path(args.extrinsics).resolve()
    for p, name in [(svo_path, "SVO"), (rsi_path, "RSI CSV"), (ext_path, "Extrinsics")]:
        if not p.exists():
            logging.error(f"{name} 不存在: {p}")
            sys.exit(1)

    # ── 输出目录 ──
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = svo_path.parent / f"reconstruction_{svo_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 加载外参 ──
    T_c2g = load_extrinsics(ext_path)
    logging.info(f"T_cam2gripper:\n{np.array2string(T_c2g, precision=4)}")

    # ── 加载 RSI ──
    logging.info(f"加载 RSI: {rsi_path.name} ...")
    ts_rsi, poses_rsi = load_rsi_csv(rsi_path)
    logging.info(f"RSI: {len(ts_rsi)} 行, 时间范围 [{ts_rsi[0]:.3f}, {ts_rsi[-1]:.3f}], "
                 f"持续 {ts_rsi[-1] - ts_rsi[0]:.1f}s")

    # ── 打开 SVO ──
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # 使用 FFS 而非 ZED 深度
    input_type = sl.InputType()
    input_type.set_from_svo_file(str(svo_path))
    init_params.input = input_type

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logging.error(f"无法打开 SVO: {status}")
        sys.exit(1)

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
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    logging.info(f"SVO: {svo_path.name}, {width}x{height} @ {fps}fps, {total_frames} frames")
    logging.info(f"K: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, baseline={baseline_m:.4f}m")

    # ── SVO 起始 epoch 推算 ──
    # 优先 1: session_meta.json (精确 epoch)
    # 优先 2: RSI CSV 中的 zed_timestamp_ns 列 (新版录制器)
    # 优先 3: 文件夹名 (秒级)
    meta_path = svo_path.parent / "session_meta.json"
    rsi_has_zed_ts = False
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        t0_svo = meta["recording_start_epoch"]
        logging.info(f"SVO 起始 epoch (从 session_meta.json): {t0_svo:.6f}")
    else:
        svo_folder_name = svo_path.parent.name
        try:
            t0_svo = datetime.strptime(svo_folder_name, "%Y%m%d_%H%M%S").timestamp()
            logging.info(f"SVO 起始 epoch (从文件夹名, 秒级): {t0_svo:.3f}")
        except ValueError:
            logging.warning(f"无法推断时间, 使用 RSI 首条时间戳")
            t0_svo = ts_rsi[0]

    t0_svo += args.time_offset
    logging.info(f"时间偏移: {args.time_offset:.3f}s → 校正后 t0_svo = {t0_svo:.3f}")

    # 检查 RSI CSV 是否包含 ZED 帧级时间戳 (新版录制器)
    with open(rsi_path, "r") as f:
        header = f.readline().strip().split(",")
    if "zed_timestamp_ns" in header and "svo_frame_idx" in header:
        rsi_has_zed_ts = True
        logging.info("RSI CSV 包含 ZED 帧级时间戳 — 使用精确帧同步")

    # ── 加载 FFS 模型 ──
    logging.info(f"加载 FFS 模型: {args.model} (iters={args.valid_iters}) ...")
    model = load_ffs_model(args.model, args.valid_iters, args.max_disp)
    logging.info("模型加载完成")

    # ── 帧范围 ──
    start = max(0, args.start)
    end = args.end if args.end >= 0 else total_frames - 1
    end = min(end, total_frames - 1)

    # ── 世界坐标 bounding box (从 RSI 轨迹自动推算) ──
    bbox_min_m, bbox_max_m = None, None
    if args.use_bbox:
        if args.bbox_min and args.bbox_max:
            # 手动指定 (mm)
            bbox_min_m = np.array(args.bbox_min) / 1000.0
            bbox_max_m = np.array(args.bbox_max) / 1000.0
        else:
            # 从 RSI 轨迹自动推算: 喷嘴活动范围 + padding
            xyz_mm = poses_rsi[:, :3]  # (N, 3) x,y,z in mm
            pad = args.bbox_pad_mm / 1000.0
            z_below = args.bbox_z_below_mm / 1000.0
            bbox_min_m = xyz_mm.min(axis=0) / 1000.0 - np.array([pad, pad, z_below])
            bbox_max_m = xyz_mm.max(axis=0) / 1000.0 + np.array([pad, pad, pad])
        logging.info(f"世界坐标 bbox (m): min={bbox_min_m} max={bbox_max_m}")
        logging.info(f"  尺寸: {(bbox_max_m - bbox_min_m)*1000} mm")

    # ── 参考帧深度 (用于 ROI) ──
    depth_ref = None
    if args.use_roi:
        logging.info(f"获取参考帧 {args.ref_frame} 的深度 (用于 ROI) ...")
        zed.set_svo_position(args.ref_frame)
        runtime_params = sl.RuntimeParameters()
        left_mat, right_mat = sl.Mat(), sl.Mat()
        padder_cache = {}
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            img_l = cv2.cvtColor(left_mat.get_data()[:, :, :3], cv2.COLOR_BGRA2RGB)
            img_r = cv2.cvtColor(right_mat.get_data()[:, :, :3], cv2.COLOR_BGRA2RGB)
            disp_ref = ffs_disparity(model, img_l, img_r, args.valid_iters, padder_cache)
            depth_ref = fx * baseline_m / np.clip(disp_ref, 1e-6, None)
            logging.info("参考帧深度已获取")
        else:
            logging.warning("无法读取参考帧, ROI 将被禁用")
            args.use_roi = False

    # ── 主循环 ──
    runtime_params = sl.RuntimeParameters()
    left_mat, right_mat = sl.Mat(), sl.Mat()
    padder_cache = {}

    pcd_all = o3d.geometry.PointCloud()
    voxel_m = args.voxel_mm / 1000.0
    processed = 0
    skipped_no_pose = 0
    frames_dir = out_dir / f"frames_{time_tag}" if args.save_frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    zed.set_svo_position(start)
    logging.info(f"帧范围: [{start}, {end}], skip={args.skip_frames}, max={args.max_frames or '∞'}")
    logging.info("开始重建推理... (第一帧较慢)")

    try:
        while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            frame_idx = zed.get_svo_position()
            if frame_idx > end:
                break
            if (frame_idx - start) % args.skip_frames != 0:
                continue

            # ── 时间戳 → RSI 位姿 ──
            t_frame = t0_svo + frame_idx / fps
            pose_6dof = interp_pose(ts_rsi, poses_rsi, t_frame)
            if pose_6dof is None:
                skipped_no_pose += 1
                continue

            T_g2b = pose_to_T(*pose_6dof)
            T_cam2world = T_g2b @ T_c2g  # 完整变换: cam → gripper → base

            # ── 取左右图 ──
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            img_left = cv2.cvtColor(left_mat.get_data()[:, :, :3], cv2.COLOR_BGRA2RGB)
            img_right = cv2.cvtColor(right_mat.get_data()[:, :, :3], cv2.COLOR_BGRA2RGB)

            # ── FFS 推理 ──
            disp = ffs_disparity(model, img_left, img_right, args.valid_iters, padder_cache)
            depth = fx * baseline_m / np.clip(disp, 1e-6, None)  # meters

            # ── 保存 depth ──
            if args.save_depth_npy:
                depth_dir = out_dir / "depth_frames"
                depth_dir.mkdir(exist_ok=True)
                # 彩色深度图
                d_clip = np.clip(depth, args.depth_min, args.depth_max)
                d_norm = ((d_clip - args.depth_min) / (args.depth_max - args.depth_min) * 255).astype(np.uint8)
                invalid = ~np.isfinite(depth) | (depth < args.depth_min) | (depth > args.depth_max)
                d_norm[invalid] = 0
                depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
                depth_color[invalid] = 0
                # ROI 裁剪辅助
                roi = args.depth_roi if args.depth_roi_enabled else None
                def _crop(img):
                    if roi:
                        return img[roi[1]:roi[3], roi[0]:roi[2]]
                    return img
                depth_c = _crop(depth_color)
                depth_np = _crop(depth).astype(np.float32)
                np.save(str(depth_dir / f"depth_{frame_idx:06d}.npy"), depth_np)
                mode = args.depth_output_mode
                if mode == "tri":
                    left_bgr = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
                    right_bgr = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)
                    combo = np.hstack([_crop(left_bgr), _crop(right_bgr), depth_c])
                    cv2.imwrite(str(depth_dir / f"depth_{frame_idx:06d}.png"), combo)
                elif mode == "sep_rgb_depth":
                    left_bgr = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
                    right_bgr = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)
                    stereo = np.hstack([_crop(left_bgr), _crop(right_bgr)])
                    cv2.imwrite(str(depth_dir / f"stereo_{frame_idx:06d}.png"), stereo)
                    cv2.imwrite(str(depth_dir / f"depth_{frame_idx:06d}.png"), depth_c)
                else:  # depth_only
                    cv2.imwrite(str(depth_dir / f"depth_{frame_idx:06d}.png"), depth_c)

            # ── ROI 过滤 ──
            if args.use_roi and depth_ref is not None:
                roi_mask = compute_roi_mask(depth_ref, depth,
                                           grow_threshold_m=args.roi_threshold_mm / 1000.0)
                depth = np.where(roi_mask, depth, np.nan)

            # ── 反投影 → 相机坐标点云 ──
            pts_cam, colors = backproject(
                depth, K,
                stride=args.point_stride,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                color_img=None if args.no_color else img_left,
            )

            if pts_cam.shape[0] == 0:
                continue

            # ── 变换到世界坐标 ──
            pts_world = (T_cam2world @ pts_cam.T).T[:, :3]  # (N, 3) meters

            # ── 世界坐标 bbox 裁剪 ──
            if bbox_min_m is not None and bbox_max_m is not None:
                in_box = np.all(
                    (pts_world >= bbox_min_m) & (pts_world <= bbox_max_m), axis=1
                )
                pts_world = pts_world[in_box]
                if colors is not None:
                    colors = colors[in_box]

            if pts_world.shape[0] == 0:
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_world)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # 帧级体素下采样 (减少累积量)
            pcd = pcd.voxel_down_sample(voxel_m)

            if args.save_frames and frames_dir:
                o3d.io.write_point_cloud(
                    str(frames_dir / f"frame_{frame_idx:06d}.ply"), pcd)

            pcd_all += pcd
            processed += 1

            if processed % 10 == 0:
                logging.info(
                    f"帧 {frame_idx}/{end} | 已处理 {processed} | "
                    f"累积 {len(pcd_all.points)} 点 | "
                    f"位姿 x={pose_6dof[0]:.1f} y={pose_6dof[1]:.1f} z={pose_6dof[2]:.1f}"
                )

            if args.max_frames > 0 and processed >= args.max_frames:
                break

    finally:
        zed.close()

    if skipped_no_pose > 0:
        logging.warning(f"跳过 {skipped_no_pose} 帧 (无对应 RSI 位姿)")

    if len(pcd_all.points) == 0:
        logging.error("无有效点云, 请检查时间同步和深度范围")
        sys.exit(1)

    logging.info(f"融合点云: {len(pcd_all.points)} 点 (处理了 {processed} 帧)")

    # ── 全局体素下采样 ──
    pcd_all = pcd_all.voxel_down_sample(voxel_m)
    logging.info(f"下采样后: {len(pcd_all.points)} 点 (voxel={args.voxel_mm}mm)")

    # ── 保存融合点云 ──
    fused_path = out_dir / f"fused_{time_tag}.ply"
    o3d.io.write_point_cloud(str(fused_path), pcd_all)
    logging.info(f"融合点云: {fused_path}")

    # ── 法线估计 + Poisson 重建 ──
    logging.info("估计法线 ...")
    pcd_all.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 4, max_nn=30)
    )
    # 朝向一致化 (假设相机大致在 +Z 方向)
    pcd_all.orient_normals_towards_camera_location(np.array([0.0, 0.0, 2.0]))

    logging.info(f"Poisson 重建 (depth={args.poisson_depth}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_all, depth=args.poisson_depth
    )

    # 去除低密度面
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(densities < threshold)

    mesh_path = out_dir / f"mesh_{time_tag}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    logging.info(f"网格已保存: {mesh_path}")
    logging.info(f"顶点: {len(mesh.vertices)}, 面: {len(mesh.triangles)}")

    # ── 输出摘要 ──
    summary = {
        "svo": str(svo_path),
        "rsi": str(rsi_path),
        "extrinsics": str(ext_path),
        "model": args.model,
        "frames_processed": processed,
        "skip_frames": args.skip_frames,
        "voxel_mm": args.voxel_mm,
        "fused_points": len(pcd_all.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_faces": len(mesh.triangles),
        "time_offset": args.time_offset,
    }
    summary_path = out_dir / f"summary_{time_tag}.txt"
    with open(summary_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    logging.info(f"摘要: {summary_path}")


if __name__ == "__main__":
    main()
