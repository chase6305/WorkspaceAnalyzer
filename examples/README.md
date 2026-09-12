# 真实机器人示例

优先维护 Marvin M6 和 Dexforce W1 的真实 URDF 示例。以下程序运行的是模型分析，
不会连接或驱动实体机器人；结果不包含碰撞检查。

## 准备资产

```bash
python -m pip install -e '.[viser,sampling]'
```

默认查找与 WorkspaceAnalyzer 同级的 `HumanoidAssets` 仓库。资产放在其他目录时：

```bash
export HUMANOID_ASSETS=/path/to/HumanoidAssets
```

目录中应包含 `Marvin_M6_S_CCS_696_V4.0/robot.urdf` 或
`Dexforce_W1_V3/robot.urdf` 及其引用的 mesh。也可通过 `--urdf` 指定 Marvin
文件；对比程序使用 `--w1-urdf` 和 `--marvin-urdf`，显式路径优先。
资产未随 Python 包分发。安装 wheel 后应设置上述环境变量或传入路径。

## 可视化入口

在项目根目录执行。每次启动一个 demo，在浏览器打开 `http://localhost:8080`；
需要同时运行时用 `--port` 指定不同端口。

| 示例 | 实际模型 | 用途 |
| --- | --- | --- |
| `marvin_single_arm.py` | Marvin | 关节采样工作空间、位置/固定姿态 IK |
| `plane_reachability.py` | Marvin | 指定坐标系和平面上的可达性分布 |
| `trajectory_gallery.py` | Marvin | 直线、圆、八字、螺旋轨迹 IK 和播放 |
| `marvin_dexterity.py` | Marvin | 灵活度点云和连续轨迹分析 |
| `compare_w1_marvin.py` | W1 + Marvin | 共同目标与轨迹下的双机器人对比 |

```bash
python examples/marvin_single_arm.py --backend numpy --samples 20000 --viser
python examples/plane_reachability.py --backend numpy --position-only --resolution 21 --viser
python examples/trajectory_gallery.py --backend numpy --shape figure8 --viser
python examples/marvin_dexterity.py --backend numpy --samples 10000 --viser
python examples/compare_w1_marvin.py --backend numpy --viser
```

均支持 `--arm right` 切换右臂。Marvin 的默认参考姿态为 J4 弯肘 90°，
对应此 URDF 的关节值 −π/2；其余关节取限位中心。平面扫描使用显式固定坐标系，
不随参考姿态变化。对比 demo 为便于阅读轨迹，初始隐藏点云。

## 实际模型回归

```bash
python -m pip install -e '.[dev,viser]'
make test-robot
```

此入口检查真实 Marvin 左右臂的 Jacobian、J4 默认姿态，以及两臂的直线、圆、
八字和螺旋轨迹：验证 FK 位置误差、关节限位、连续性和速度限位。
安装 trimesh 后也检查模型 mesh。报告写入 `outputs/marvin-tests.xml`。
缺少 URDF 时明确失败，避免把跳过误当作实际模型已验证。

`benchmark_marvin.py` 和 `stress_marvin.py` 保留作真实模型性能及稳定性工具。
`reachability_workflow.py` 是可接收实际 URDF 的高级诊断工具；
`benchmark_candidates.py`、`benchmark_perturbations.py`、`benchmark_result_io.py`
及 `tests/fixtures` 中的合成模型主要用于内部回归，不作为可视化 demo 的维护重点。
