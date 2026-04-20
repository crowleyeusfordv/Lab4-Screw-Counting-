# Lab4 视频螺丝计数 - 团队作业

## 团队成员

| 学号 | 姓名 | 分工 |
|------|------|------|
| 523030910127 | 杨文潇 | yolo-seg目标检测与实例分割模型训练 |
| 523030910126 | 欧明亮 | 分析代码各模块消耗时间占比优化运行速度 |
| 523030910128 | 莫韫恺 | 视频处理计数流水线构建 |
| 523030910103 | 魏思齐 | 视频处理链路的工程级性能优化与工程封装 |

---

## 环境配置

### 1. 创建并激活 Conda 环境（推荐）

本项目推荐使用 **Anaconda / Miniconda** 管理环境。

```bash
# 创建名为 screw_count 的 Python 3.10 环境
conda create -n screw_count python=3.10 -y

# 激活环境
conda activate screw_count
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

**（备选）使用 venv 标准虚拟环境：**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# 然后安装依赖
pip install -r requirements.txt
```

### 3. 放置模型权重

将训练好的模型权重文件放至 `models/` 目录：

```
models/
  detector.pt       # B 负责提供：one-class YOLO 螺丝检测器
  classifier.pt     # C 负责提供：5 类螺丝分类器（Lab2 迁移/fine-tune）
```

> **注意**：若 `models/` 目录下缺少权重文件，系统会自动切换到兜底模式
> （OpenCV 检测 + 随机分类），精度极低，**最终提交前必须确保权重文件存在**。

---

## 运行方式

**运行环境**：请在 Conda 环境 **`screw_count`** 下执行（见上文「创建并激活 Conda 环境」）。  
若不想每次手动 `activate`，也可用：

```bash
conda run -n screw_count python run.py --data_dir ... --output_path ... --output_time_path ... --mask_output_path ...
```

### 标准运行命令（作业规范接口）

```bash
python run.py \
    --data_dir /path/to/test_videos_folder \
    --output_path ./result.npy \
    --output_time_path ./time.txt \
    --mask_output_path ./mask_folder/ \
    --device cuda:0
```

### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--data_dir` | str | ✅ | 包含测试视频的文件夹路径 |
| `--output_path` | str | ✅ | result.npy 输出路径（含文件名） |
| `--output_time_path` | str | ✅ | time.txt 输出路径（含文件名） |
| `--mask_output_path` | str | ✅ | 掩膜图像输出文件夹路径 |
| `--device` | str | ❌ | 推理设备，如 `cuda:0` 或 `cpu`（默认自动选择） |
| `--no_fp16` | flag | ❌ | 禁用 FP16 推理（CPU 环境建议添加此参数） |
| `--keyframe_strategy` | str | ❌ | 关键帧策略：`motion`（默认）或 `uniform` |
| `--dist_thresh` | float | ❌ | 去重聚类距离阈值（像素，默认 40.0） |
| `--verbose` | flag | ❌ | 输出详细调试日志 |

### 在开发视频上测试

```bash
# 激活环境后，在 submission/code/ 目录下运行：
conda activate screw_count

# 使用开发视频（GPU）
python run.py \
    --data_dir ../../vedio_exp/ \
    --output_path ./output/result.npy \
    --output_time_path ./output/time.txt \
    --mask_output_path ./output/masks/ \
    --device cuda:0

# 使用 CPU（无 GPU 时，添加 --no_fp16）
python run.py \
    --data_dir ../../vedio_exp/ \
    --output_path ./output/result.npy \
    --output_time_path ./output/time.txt \
    --mask_output_path ./output/masks/ \
    --device cpu --no_fp16
```

---

## 项目结构

```
code/
├── run.py                      # 主入口（作业规范接口）           [D]
├── pipeline.py                 # 视频处理流程编排                 [D]
├── interfaces.py               # 团队协作数据接口定义             [D]
├── requirements.txt            # Python 依赖列表                 [D]
├── README.md                   # 本文档                         [D]
│
├── modules/                    # 核心算法模块
│   ├── detector.py             # 螺丝检测器（YOLO + SAHI）       [B]
│   ├── registration.py         # 锚帧几何配准（AKAZE + Homography）[A]
│   ├── dedup.py                # 全局去重聚类（DBSCAN）           [A]
│   └── classifier.py           # 5 类螺丝分类器                  [C]
│
├── utils/                      # 工程工具包
│   ├── video_io.py             # 视频读取与帧提取                [D]
│   ├── output_formatter.py     # 输出格式封装（npy/time/mask）   [D]
│   └── visualizer.py           # 掩膜叠加与可视化               [D]
│
├── tools/                      # 数据工具（独立可运行脚本）
│   ├── extract_keyframes.py    # 关键帧批量抽取（标注数据准备）   [D]
│   ├── export_crops.py         # 检测 Crop 导出                  [D]
│   ├── convert_annotations.py  # 标注格式转换（CVAT/YOLO/COCO）  [D]
│   ├── benchmark.py            # 速度 Benchmark                  [D]
│   └── ablation.py             # 消融实验记录与对比              [D]
│
└── models/                     # 模型权重目录
    ├── detector.pt             # YOLO 检测器权重（B 提供）
    └── classifier.pt           # 分类器权重（C 提供）
```

---

## 算法概述

```
视频读取 + 方向校正
        ↓
   关键帧提取（默认 uniform=30；可选 motion，失败时回退 uniform）
        ↓
   多类螺丝检测（YOLO Detector）
        ↓
   帧间几何配准（ORB 特征 + Homography）
        ↓
   投影到参考坐标系 + 去重聚类（incremental / dbscan）
        ↓
   按 Cluster 计数（run.py 默认 detector_votes）
       ↓
   中间帧检测结果叠加 + 计数面板绘制
        ↓
   输出 result.npy / mask / time.txt        [D]
```

**核心思路（与 run.py 当前实现一致）**：
`run.py` 会调用 `count_videos` 后端，先抽取关键帧并做多类检测，再通过配准将检测框映射到统一参考坐标系，最后用去重聚类得到唯一实例并统计 5 类数量。默认计数模式是 `detector_votes`（直接使用检测器类别投票），并非默认依赖 `classifier.pt`；`classifier` 模式属于可选路径。mask 图像来自每段视频中间帧的检测叠加结果。

---

*本项目代码由团队共同开发，各模块分工见上方表格及各文件开头的 Owner 注释。*