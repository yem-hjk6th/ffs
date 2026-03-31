# 离线后印重建 (Post-Print Reconstruction) 完整分析

## 1. 数据同步机制分析

### 1.1 现有时间戳体系

| 数据源 | 时间戳格式 | 精度 | 频率 |
|--------|-----------|------|------|
| RSI CSV | Unix epoch (float) | 微秒级 (~6-9位小数) | ~170 Hz (~5.9ms/行) |
| SVO2 视频 | 帧索引 → `frame_idx / fps` | 33.3ms (30fps) | 30 fps |
| SVO 录制开始 | 文件夹名 `YYYYMMDD_HHMMSS` | 秒级 | — |

### 1.2 如何对齐 SVO 帧与 RSI 位姿

**步骤 1 — 粗同步 (秒级)**

SVO 文件夹名即为录制启动时刻，例如 `20260324_174844` → `2026-03-24 17:48:44`。
RSI CSV 第一行 `timestamp = 1774388924.690708`。
将文件夹名转为 epoch：

```python
from datetime import datetime
t0_svo = datetime.strptime("20260324_174844", "%Y%m%d_%H%M%S").timestamp()
# → epoch (秒级，无毫秒)
```

每帧 SVO 的估计 epoch：`t_frame = t0_svo + frame_idx / 30.0`

**步骤 2 — 精同步 (层边界对齐)**

由于 SVO 启动时刻仅有秒级精度，需要利用打印事件做精确对齐：

1. 在 RSI CSV 中检测第一个 layer 边界（Z 高度跳变，从 `layers_*.csv` 中 `t_offset_s` 得到）
2. 在 SVO 深度序列中检测 "深度开始减少" 的帧（= 打印开始）
3. 两个事件的时间差即为 `dt_offset` 校正量

校正后：`t_frame_corrected = t0_svo + frame_idx / 30.0 + dt_offset`

**步骤 3 — RSI 插值**

给定 `t_frame`，在 RSI CSV 中找到时间最近的两行，线性插值得到 (x,y,z,a,b,c)。
RSI 频率 ~170Hz，SVO 30fps → 每帧 SVO 对应 ~5-6 个 RSI 包，延迟 < 6ms。

### 1.3 已知同步误差

- **camera_lag**: 标定数据显示 ZED 帧到达比 RSI 晚 15-30ms
- **时钟漂移**: 短时间打印 (<10 min) 内漂移可忽略 (<数毫秒)
- **总误差估计**: < 50ms → 以打印速度 ~10mm/s 计，位置误差 < 0.5mm，可接受

---

## 2. 采样策略

### 2.1 数据规模

- 5 分钟打印 @ 30fps = **9000 帧**
- FFS 推理 ~50ms/帧 (RTX 5070 Ti, 23-36-37模型, 8 iter) → ~20 fps
- 用户目标: < 1 fps 即可满足重建需求

### 2.2 推荐策略: 均匀采样 + 层感知

**方案 A — 简单均匀 (推荐起步)**

```
skip_frames = 30  →  1 fps, 5分钟 → ~300帧
skip_frames = 15  →  2 fps, 5分钟 → ~600帧
```

300 帧 @ 50ms/帧 = **15 秒**处理时间。600 帧 = 30 秒。远快于视频时长。

**方案 B — 层感知采样 (进阶)**

```
每层内部: 均匀取 ~20 帧 (覆盖一整圈轨迹)
层过渡区: 跳过 (机器人抬升/位移，非打印态)
```

判断打印态: RSI 中 `vel > 0 && z < threshold`

**方案 C — 运动多样性采样 (最优)**

```python
# 当机器人平移 > 5mm 或旋转 > 2° 时采样
delta_pos = norm(xyz_i - xyz_prev)
delta_ang = max(|a_i - a_prev|, |b_i - b_prev|, |c_i - c_prev|)
if delta_pos > 5.0 or delta_ang > 2.0:
    sample this frame
```

**推荐**: 先用方案 A (`skip_frames=30`) 跑通完整流程，再根据效果切换 B/C。

---

## 3. 最适合改造的脚本

### 3.1 对比

| 脚本 | 优点 | 缺点 | 改造难度 |
|------|------|------|----------|
| `ffs/run_depth_svo.py` | 已集成 SVO+FFS，输出 depth npy | 无点云/位姿逻辑 | ★★☆ 中 |
| `svo_extract/svo_pt_mesh_imu.py` | 有完整点云融合+Poisson | 用 ZED SLAM 位姿而非 RSI | ★★☆ 中 |
| `svo_extract/svo_pt_mesh_base.py` | 最简洁的点云融合 | 无位姿，无 FFS | ★★★ 易 |

### 3.2 推荐: 基于 `run_depth_svo.py` 改造

理由:
1. **已有 SVO→FFS 完整链路**: 读 SVO、取左右图、推理 disparity、计算 depth
2. **已有 K.txt 输出**: 内参和 baseline 已经提取
3. **只需添加**: RSI CSV 加载 → 时间戳匹配 → KUKA 位姿构建 → 点云变换 → 融合 → 导出

新脚本命名: `reconstruct_svo.py`

---

## 4. 缺失数据清单

### 4.1 必须有 (Blocking)

| 数据 | 状态 | 说明 |
|------|------|------|
| **T_cam2gripper (外参)** | ❌ 需重新标定 | 机器人已移位，现有 R6 外参已过时 |
| **配对的 SVO2 + RSI CSV** | ⚠️ 仅 1 组完整 | `20260324_174844` 有 SVO + RSI；`20260326_202106` 仅有 RSI |

### 4.2 可以推算 (Non-blocking)

| 数据 | 状态 | 说明 |
|------|------|------|
| SVO 帧↔Unix epoch 映射 | ✅ 可推算 | `folder_name → epoch + frame_idx/fps` |
| 层边界时刻 | ✅ 已有 | `layers_*.csv` 已提供 |
| 相机内参 K + baseline | ✅ 自动提取 | `pyzed` API 直接读 SVO 元数据 |

### 4.3 准备步骤

1. **重新标定外参**: 运行 `exp04_sync_capture.py` 采集新的 ArUco-RSI 同步数据，然后 `R6/calibrate.py` 计算新的 T_cam2gripper
2. **确认配对数据**: 录制新的打印实验，同时启动 RSI recorder 和 SVO recorder，确保文件夹时间戳一致
3. **可选**: 在 SVO recorder 中保存精确的 `time.time()` 作为录制开始时间（当前只有秒级的文件夹名）

---

## 5. 几何重建数学推导

### 5.1 坐标系定义

```
Base Frame (世界坐标)
  ↑ T_gripper2base (来自 RSI: x,y,z,a,b,c)
Gripper Frame (法兰坐标)
  ↑ T_cam2gripper (来自手眼标定 extrinsics.txt)
Camera Frame (相机坐标)
  ↑ 反投影 (来自 K + depth)
Image Plane (像素坐标)
```

### 5.2 Step 1: Disparity → Depth

FFS 输出视差 $d(u,v)$ (像素)，深度计算:

$$Z_{cam} = \frac{f_x \cdot B}{d(u,v)}$$

其中 $f_x$ = 焦距 (像素), $B$ = 基线 (米)

### 5.3 Step 2: Pixel → Camera 3D

反投影到相机坐标系:

$$X_{cam} = \frac{(u - c_x) \cdot Z_{cam}}{f_x}$$

$$Y_{cam} = \frac{(v - c_y) \cdot Z_{cam}}{f_y}$$

$$P_{cam} = \begin{bmatrix} X_{cam} \\ Y_{cam} \\ Z_{cam} \\ 1 \end{bmatrix}$$

### 5.4 Step 3: Camera → World

KUKA ZYX 欧拉角 → 旋转矩阵:

$$R_{gripper \to base} = R_z(A) \cdot R_y(B) \cdot R_x(C)$$

$$T_{gripper \to base} = \begin{bmatrix} R_{gripper \to base} & \frac{1}{1000}\begin{bmatrix} x \\ y \\ z \end{bmatrix} \\ 0 & 1 \end{bmatrix}$$

完整变换链:

$$P_{world} = T_{gripper \to base} \cdot T_{cam \to gripper} \cdot P_{cam}$$

### 5.5 Step 4: 多帧融合

```python
pcd_all = o3d.geometry.PointCloud()
for frame_i in selected_frames:
    P_cam_i = backproject(depth_i, K)           # Nx4
    T_world_i = T_g2b_i @ T_c2g                 # 4x4
    P_world_i = (T_world_i @ P_cam_i.T).T[:,:3] # Nx3
    pcd_i = o3d.geometry.PointCloud(points=P_world_i)
    pcd_all += pcd_i

pcd_all = pcd_all.voxel_down_sample(voxel_size=0.002)  # 2mm
```

### 5.6 Step 5: Surface Reconstruction

```python
pcd_all.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_all, depth=9
)
# 去除低密度面 (边缘噪声)
densities = np.asarray(densities)
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
o3d.io.write_triangle_mesh("reconstructed.ply", mesh)
```

### 5.7 ROI 检测 (轻量替代 SAM2)

不需要 SAM2。利用 **深度差分** 方法 (已有实现在 `depth_layer_detect.py`):

```python
# 取打印前的参考帧 depth_ref (frame 0-10)
# 与当前帧 depth_cur 比较
grown_mask = (depth_ref - depth_cur) > 3.0  # mm, 深度减少 = 物体长高
# grown_mask 即为打印区域的 ROI
```

优点: **零额外推理开销**，纯 numpy 操作，< 1ms/帧

---

## 6. 完整重建流程图

```
录制数据                              标定数据
┌──────────┐  ┌──────────┐         ┌────────────────┐
│ SVO2     │  │ RSI CSV  │         │ extrinsics.txt │
│ (视频)   │  │ (位姿)   │         │ T_cam2gripper  │
└────┬─────┘  └────┬─────┘         └───────┬────────┘
     │              │                       │
     ▼              ▼                       │
 ┌───────────────────────┐                  │
 │  时间戳对齐           │                  │
 │  SVO frame → epoch    │                  │
 │  → 查找最近 RSI 行    │                  │
 │  → 线性插值 pose      │                  │
 └──────────┬────────────┘                  │
            │                               │
     ┌──────┴──────┐                        │
     ▼             ▼                        │
 ┌────────┐  ┌──────────┐                  │
 │ L/R    │  │ RSI pose │                  │
 │ images │  │ (x,y,z,  │                  │
 └───┬────┘  │  a,b,c)  │                  │
     │       └────┬─────┘                  │
     ▼            │                         │
 ┌────────┐       │                         │
 │ FFS    │       │                         │
 │ depth  │       │                         │
 └───┬────┘       │                         │
     │            │                         │
     ▼            ▼                         ▼
 ┌───────────────────────────────────────────┐
 │  P_cam = backproject(depth, K)            │
 │  T = T_gripper2base(pose) @ T_cam2gripper │
 │  P_world = T @ P_cam                     │
 └──────────────────┬────────────────────────┘
                    │
                    ▼
 ┌──────────────────────────┐
 │  多帧点云融合             │
 │  voxel downsample        │
 │  Poisson reconstruction  │
 │  → mesh.ply              │
 └──────────────────────────┘
```

---

## 7. 下一步行动

1. **立即可做**: 用 `reconstruct_svo.py` 在 `20260324_174844` 数据上测试 (使用旧外参，验证流程)
2. **录制新数据**: 同时启动 RSI recorder + SVO recorder，确保配对完整
3. **重新标定**: 运行手眼标定获取新的 T_cam2gripper
4. **用新数据重建**: 完整流程
