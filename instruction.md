# Fast-FoundationStereo (FFS) 使用指南

> **模型**: [NVIDIA Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) (CVPR 2026)  
> **功能**: 实时零样本立体匹配 → 深度估计  
> **硬件**: RTX 5070 Ti Laptop GPU (sm_120 Blackwell)  
> **环境**: conda env `ffs` / Python 3.12 / PyTorch nightly cu128

---

## 目录

1. [环境配置](#1-环境配置)
2. [项目结构](#2-项目结构)
3. [模型权重](#3-模型权重)
4. [核心概念](#4-核心概念)
5. [使用方式](#5-使用方式)
   - 5.1 [Demo 图片测试](#51-demo-图片测试)
   - 5.2 [自定义图像对](#52-自定义图像对)
   - 5.3 [图像序列文件夹](#53-图像序列文件夹)
   - 5.4 [视频深度估计](#54-视频深度估计)
   - 5.5 [SVO2 直接推理 (一键)](#55-svo2-直接推理-一键)
   - 5.6 [SVO2 提取为图像/视频](#56-svo2-提取为图像视频)
6. [关键参数说明](#6-关键参数说明)
7. [内参文件 K.txt 格式](#7-内参文件-ktxt-格式)
8. [性能调优](#8-性能调优)
9. [常见问题](#9-常见问题)
10. [离线后印重建](#10-离线后印重建-post-print-reconstruction)

---

## 1. 环境配置

环境已配置完毕。后续使用只需激活:

```powershell
conda activate ffs
```

<details>
<summary>如果需要从零配置 (点击展开)</summary>

```powershell
# 1. 创建环境
conda create -n ffs python=3.12 -y
conda activate ffs

# 2. 安装 PyTorch (RTX 50 系列需要 nightly + cu128)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. 安装 triton (Windows)
pip install triton-windows

# 4. 安装其他依赖
pip install -r Fast-FoundationStereo/requirements.txt

# 5. 下载权重
pip install gdown
cd Fast-FoundationStereo/weights
gdown --folder "https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"
# 如果下载到了 weights/weights/，移动一层:
# Move-Item -Path weights\* -Destination . -Force
```
</details>

---

## 2. 项目结构

```
Repo/
├── Fast-FoundationStereo/          # 原始 repo (不要修改)
│   ├── core/                       # 模型核心代码
│   ├── weights/                    # 模型权重
│   │   ├── 23-36-37/              # 最精确
│   │   ├── 20-26-39/              # 中等
│   │   └── 20-30-48/              # 最快
│   ├── demo_data/                  # 示例数据
│   │   ├── left.png, right.png
│   │   └── K.txt
│   ├── scripts/
│   │   ├── run_demo.py            # 官方 demo
│   │   └── make_onnx.py           # ONNX 导出
│   └── Utils.py
│
└── ffs/                            # 自定义脚本 (本文件夹)
    ├── instruction.md              # ← 你正在看的文件
    ├── run_depth_images.py         # 图像对 → 深度
    ├── run_depth_video.py          # 视频 → 深度视频
    ├── run_depth_svo.py            # SVO2 → 深度视频 (一键)
    └── svo2_to_stereo.py           # SVO2 → 左右图像/视频
```

---

## 3. 模型权重

三个预训练模型已下载到 `Fast-FoundationStereo/weights/`:

| 模型名 | valid_iters | 推理时间 (ms) | 精度 (bp2) | 适用场景 |
|---------|-------------|-------------|-----------|---------|
| **23-36-37** | 8 | 49.4 | 23.4 | 最高精度, 离线分析 |
| **23-36-37** | 4 | 41.1 | 18.4 | 精度/速度平衡 |
| **20-26-39** | 8 | 43.6 | 19.4 | 中等 |
| **20-30-48** | 4 | 29.3 | 14.0 | 最快, 实时应用 |

> 推理时间在 RTX 3090, 640×480 上测量。5070 Ti 上会更快。

---

## 4. 核心概念

### 这是立体匹配模型，不是单目深度

- **输入**: 一对已校正的**左图 + 右图** (来自双目相机)
- **输出**: 视差图 (disparity) → 可转为深度图 (需要内参)
- **不支持**: 单个摄像头拍的视频 (那种场景用 Depth Anything)

### 视差 vs 深度

```
depth = focal_length × baseline / disparity
```

- `focal_length`: 相机焦距 (像素), 即 K 矩阵的 fx
- `baseline`: 左右相机间距 (米)
- 没有内参文件也能看视差彩图, 但无法得到物理距离

### ZED 相机的数据

录制的 SVO2 文件包含左右图像。可以:
- 用 `svo2_to_stereo.py` 先提取为图像/视频, 再用 `run_depth_video.py`
- 或直接用 `run_depth_svo.py` 一键处理

---

## 5. 使用方式

> **所有命令都在 `Repo/ffs/` 目录下运行, 需先 `conda activate ffs`**

### 5.1 Demo 图片测试

验证环境是否正常:

```powershell
cd C:\Users\888y9\Desktop\Repo\Fast-FoundationStereo

python scripts/run_demo.py `
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth `
  --left_file demo_data/left.png `
  --right_file demo_data/right.png `
  --intrinsic_file demo_data/K.txt `
  --out_dir output/ `
  --get_pc 0 `
  --valid_iters 4
```

会弹出窗口显示 左图|右图|视差彩图，按任意键关闭。  
输出保存在 `Fast-FoundationStereo/output/disp_vis.png`。

---

### 5.2 自定义图像对

```powershell
cd C:\Users\888y9\Desktop\Repo\ffs

# 单对图像
python run_depth_images.py `
  --left path/to/left.png `
  --right path/to/right.png `
  --out_dir output_test
```

带内参 + 保存深度 npy:
```powershell
python run_depth_images.py `
  --left left.png --right right.png `
  --intrinsic_file K.txt `
  --save_npy `
  --out_dir output_test
```

---

### 5.3 图像序列文件夹

准备如下结构:
```
my_data/
├── left/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── right/
    ├── 000001.png    # 文件名必须和 left/ 一一对应
    ├── 000002.png
    └── ...
```

```powershell
python run_depth_images.py `
  --left_dir my_data/left `
  --right_dir my_data/right `
  --out_dir output_seq
```

---

### 5.4 视频深度估计

#### 并排 (SBS) 立体视频

一个视频文件，画面是 [左|右] 水平拼接:

```powershell
python run_depth_video.py `
  --sbs_video path/to/sbs_stereo.mp4 `
  --out_dir output_video
```

#### 独立左右视频

```powershell
python run_depth_video.py `
  --left_video left.mp4 `
  --right_video right.mp4 `
  --out_dir output_video
```

#### 图像序列当视频处理

```powershell
python run_depth_video.py `
  --left_dir frames/left `
  --right_dir frames/right `
  --out_dir output_video
```

#### 快速模式 (降分辨率 + 少迭代 + 跳帧)

```powershell
python run_depth_video.py `
  --sbs_video sbs.mp4 `
  --scale 0.5 `
  --valid_iters 4 `
  --skip_frames 2 `
  --out_dir output_fast
```

输出: `output_video/depth_video.mp4` — 画面是 [原图|深度彩图] 并排。

---

### 5.5 SVO2 直接推理 (一键)

**最方便的方式**: 直接读 SVO2, 自动提取内参, 一步到位。

> ⚠️ 需要安装 ZED SDK + pyzed。如果没有，请用 5.6 先提取。

```powershell
cd C:\Users\888y9\Desktop\Repo\ffs

# 基本用法
python run_depth_svo.py `
  --svo C:\Users\888y9\Desktop\rsi_printing\recorded_data\20260327_151924\recording_20260327_151924.svo2

# 快速模式 + 指定帧范围
python run_depth_svo.py `
  --svo C:\Users\888y9\Desktop\rsi_printing\recorded_data\20260327_151924\recording_20260327_151924.svo2 `
  --start 0 --end 100 `
  --scale 0.5 --valid_iters 4 `
  --skip_frames 2

# 使用最快的模型
python run_depth_svo.py `
  --svo recording.svo2 `
  --model 20-30-48 `
  --valid_iters 4

# 保存每帧深度 npy + 无头模式
python run_depth_svo.py `
  --svo recording.svo2 `
  --save_depth_npy `
  --no_display
```

输出目录 (自动创建在 SVO 同级):
```
ffs_depth_recording_20260327_151924/
├── depth_video.mp4        # [原图|深度彩图] 并排视频
├── K.txt                  # 自动提取的相机内参
├── depth_000001.npy       # (可选) 每帧深度矩阵
└── ...
```

---

### 5.6 SVO2 提取为图像/视频

如果没有 ZED SDK, 或者想先提取再处理:

```powershell
cd C:\Users\888y9\Desktop\Repo\ffs

# 提取为 PNG 序列
python svo2_to_stereo.py `
  --svo C:\Users\888y9\Desktop\rsi_printing\recorded_data\20260327_151924\recording_20260327_151924.svo2 `
  --mode frames `
  --export_K `
  --start 0 --end 200

# 提取为独立左右视频
python svo2_to_stereo.py --svo recording.svo2 --mode video --export_K

# 提取为并排视频
python svo2_to_stereo.py --svo recording.svo2 --mode sbs_video --export_K
```

然后用提取的结果跑深度:

```powershell
# 用提取的 PNG 序列
python run_depth_images.py `
  --left_dir stereo_extract/left `
  --right_dir stereo_extract/right `
  --intrinsic_file stereo_extract/K.txt `
  --save_npy `
  --out_dir depth_output

# 用提取的并排视频
python run_depth_video.py `
  --sbs_video stereo_extract/sbs_stereo.mp4 `
  --intrinsic_file stereo_extract/K.txt `
  --out_dir depth_output
```

---

## 6. 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` / `--model_dir` | 23-36-37 | 模型选择。23-36-37 最精确, 20-30-48 最快 |
| `--valid_iters` | 8 | 迭代精化次数。4=快但稍粗, 8=精确 |
| `--max_disp` | 192 | 最大视差。192 够用, 除非需要感知极近物体 (<0.1m) |
| `--scale` | 1.0 | 图像缩放。0.5=半分辨率, 速度翻倍 |
| `--skip_frames` | 1 | 跳帧。2=每隔一帧处理 |
| `--no_display` | false | 不弹出预览窗口 (无头服务器/批处理) |
| `--save_depth_npy` | false | 保存每帧深度矩阵 (float32, 单位米) |
| `--intrinsic_file` | None | K.txt 路径, 提供后可计算真实深度 |

---

## 7. 内参文件 K.txt 格式

```
fx 0.0 cx 0.0 fy cy 0.0 0.0 1.0
baseline
```

- 第 1 行: 3×3 内参矩阵展平为 9 个数, 空格分隔
- 第 2 行: 左右相机基线距离, 单位**米**

示例 (ZED 相机):
```
754.668 0.0 489.379 0.0 754.668 265.162 0.0 0.0 1.0
0.063
```

> 使用 `svo2_to_stereo.py --export_K` 或 `run_depth_svo.py` 会自动生成。

---

## 8. 性能调优

### 速度优先

```powershell
python run_depth_svo.py --svo X.svo2 --model 20-30-48 --valid_iters 4 --scale 0.5
```
预期 ~30ms/帧 @ 320×240

### 精度优先

```powershell
python run_depth_svo.py --svo X.svo2 --model 23-36-37 --valid_iters 8 --scale 1.0
```
预期 ~50ms/帧 @ 原始分辨率

### 内存不足 (OOM)

- 降低 `--scale` (0.5 或更低)
- 降低 `--max_disp` (128)
- 使用更小的模型 (20-30-48)

### 第一帧慢

正常现象。PyTorch 的 `torch.compile` 在第一次 forward 时会编译优化 kernel。后续帧会快很多。

---

## 9. 常见问题

### Q: "CUDA capability sm_120 is not compatible"
**A**: RTX 50 系列 (Blackwell) 需要 PyTorch nightly + CUDA 12.8:
```powershell
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps
```

### Q: "Cannot find a working triton installation"
**A**: Windows 需要安装 triton-windows:
```powershell
pip install triton-windows
```

### Q: "'rm' is not recognized"
**A**: 原始 repo 的 `run_demo.py` 有 Linux 命令。无影响, 只需手动创建 output 目录:
```powershell
mkdir Fast-FoundationStereo\output -Force
```

### Q: 我的视频是单目的 (只有一个摄像头)
**A**: Fast-FoundationStereo 是**立体匹配**模型, 必须有左右图像对。单目深度估计请用 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)。

### Q: 输出的深度 npy 怎么用?
```python
import numpy as np
depth = np.load("depth_000100.npy")  # shape: (H, W), 单位: 米
print(f"中心点深度: {depth[H//2, W//2]:.3f} m")
```

### Q: 左右图能交换吗?
**A**: 不能。左图必须来自左侧相机 (场景中的物体在左图中偏右)。交换会导致完全错误的结果。

### Q: 图片需要预处理吗?
**A**: 输入图像应已经过**校正** (rectified) 和**去畸变** (undistorted)。ZED 相机默认已处理。推荐用 PNG 无损格式。

---

## 10. 离线后印重建 (Post-Print Reconstruction)

### 10.1 概述

`reconstruct_svo.py` 将 SVO2 录像 + RSI 机器人位姿 + 手眼标定外参 → 世界坐标系下的 3D 网格重建。

完整链路:
```
SVO2 (L/R 图) → FFS 深度 → 反投影3D → T_cam2gripper → T_gripper2base(RSI) → 世界坐标 → 点云融合 → Poisson mesh
```

### 10.2 前置条件

| 数据 | 说明 | 位置示例 |
|------|------|----------|
| SVO2 录像 | ZED 双目录制 | `recorded_data/20260324_174844/*.svo2` |
| RSI CSV | 机器人位姿时序 | `rsi_data/rsi_data_20260324_174844.csv` |
| 外参 T_cam2gripper | 手眼标定结果 | `calibration/.../extrinsics.txt` |

### 10.3 基本用法

```powershell
cd C:\Users\888y9\Desktop\Repo\ffs
conda activate ffs

# 最简测试 (1fps 采样, ~300帧→15s 处理)
python reconstruct_svo.py `
  --svo  C:\Users\888y9\Desktop\rsi_printing\recorded_data\20260324_174844\recording_20260324_174844.svo2 `
  --rsi  C:\Users\888y9\Desktop\rsi_printing\rsi_data\rsi_data_20260324_174844.csv `
  --extrinsics C:\Users\888y9\Desktop\rsi_printing\build\Preparation\calibration\aut_cal\extrinsics_res\20260226_150931\extrinsics.txt
```

### 10.4 常用参数

```powershell
# 更密采样 + ROI + 快速模型
python reconstruct_svo.py `
  --svo ... --rsi ... --extrinsics ... `
  --skip_frames 15 `
  --model 20-30-48 --valid_iters 4 `
  --use_roi --ref_frame 5 `
  --voxel_mm 2.0 `
  --depth_min 0.1 --depth_max 1.5 `
  --save_frames
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--skip_frames` | 30 | 每N帧取一帧 (30=1fps) |
| `--model` | 23-36-37 | FFS 模型 (20-30-48 最快) |
| `--valid_iters` | 8 | FFS 迭代 (4=快, 8=精) |
| `--voxel_mm` | 3.0 | 体素下采样 (mm) |
| `--use_roi` | 否 | 深度差分 ROI (检测打印区域) |
| `--time_offset` | 0.0 | SVO→RSI 时间偏移校正 (秒) |
| `--depth_min/max` | 0.1/2.0 | 深度范围过滤 (米) |
| `--point_stride` | 2 | 像素采样步长 |

### 10.5 输出

```
reconstruction_recording_20260324_174844/
  fused_20260401_123456.ply       # 融合点云
  mesh_20260401_123456.ply        # Poisson 网格
  summary_20260401_123456.txt     # 重建参数摘要
  frames_20260401_123456/         # (可选) 每帧点云 PLY
  depth_000100.npy ...            # (可选) 每帧深度 numpy
```

### 10.6 详细分析

参见 [reconstruction_analysis.md](reconstruction_analysis.md)，包含:
- 数据同步机制分析
- 采样策略对比
- 几何变换数学推导
- 缺失数据清单

---

## 可用的 SVO2 录像

```
rsi_printing/recorded_data/
├── 20260324_172436/recording_20260324_172436.svo2
├── 20260324_172655/recording_20260324_172655.svo2
├── 20260324_173229/recording_20260324_173229.svo2
├── 20260324_173749/recording_20260324_173749.svo2
├── 20260324_174348/recording_20260324_174348.svo2
├── 20260324_174844/recording_20260324_174844.svo2
└── 20260327_151924/recording_20260327_151924.svo2
```

快速试一个:
```powershell
cd C:\Users\888y9\Desktop\Repo\ffs
conda activate ffs

python run_depth_svo.py `
  --svo C:\Users\888y9\Desktop\rsi_printing\recorded_data\20260327_151924\recording_20260327_151924.svo2 `
  --end 50 `
  --scale 0.5 `
  --valid_iters 4
```
