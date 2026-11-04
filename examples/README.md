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
| `plane_reachability.py` | Marvin / W1 / 通用 URDF | 浏览器选择平面、采样计算、取消与导出 |
| `reachability_suite.py` | Marvin / W1 / 通用 URDF | 三组评测与单窗口切换结果 |
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
该预设仅用于 URDF robot 名为 `marvin` 或 `marvin_*` 的模型；W1 等其他模型
默认所有关节取限位中心，不会因关节同名套用 Marvin 姿态。

## 平面任务与评测入口

`plane_reachability.py --viser` 先启动页面，不会在启动前运行整片 IK。
在 **Plane reachability** 中选择平面、坐标系、中心、边长及每轴点数，再点击计算。
可以使用规则网格，或选择采样策略并指定样本总数。
每次成功完成会更新 JSON、NPZ、SVG 文件，页面提供对应下载按钮及二维图。
点击下载按钮后，再点击 Viser 弹出的文件链接保存到本地。
默认与主 CLI 一致：XY/base、中心为零、仅位置。
未完成或取消的任务保留上一次结果。计算完成但导出失败时，新结果仍可下载；
修复路径后用 **Retry saving last plane result** 重试失败文件，已保存文件不重复写入。三个轴向平面、显式链选择和完整 CLI 用法见
[平面任务指南](../docs/plane-reachability.md)。

```bash
python examples/plane_reachability.py --urdf /path/to/robot.urdf \
  --base-link left_arm_base --tip-link left_ee \
  --frame base --plane xy --constant 0.2 --center 0.4 0 0.2 \
  --span 0.6 --resolution 21 --position-only --backend numpy --viser

python examples/reachability_suite.py --urdf /path/to/robot.urdf \
  --backend numpy --samples 1000 --batch-size 256 --viser --port 8081
```

套件先生成三份 NPZ 和总报告，再在一个 viewer 内通过 **Experiment** 切换。
保存失败时仍尝试其他结果和总报告，退出码为 2，并列出成功保存的路径。
总报告的 `exports_complete` 表示本次三份 NPZ 是否全部保存，
每个 case 的 `saved` 和失败时的 `export_error` 标明本次保存状态。
失败路径上可能仍有上次运行的文件，应先检查 `saved` 再读取产物。
支持 `--strategy`，位置 IK 和固定姿态 IK 使用相同的采样点。总报告记录 URDF SHA-256、
实际后端/精度、采样策略与种子、IK 预算、参考关节和位姿、关节顺序及每个产物路径。
`reproduce_args` 保存重新运行的参数列表，并固定本次实际 bounds。

```python
import json
import subprocess
import sys

with open("outputs/reachability_suite/report.json") as stream:
    report = json.load(stream)
subprocess.run(
    [
        sys.executable,
        "examples/reachability_suite.py",
        *report["reproduce_args"],
        "--output-dir",
        "outputs/suite-replay",
    ],
    check=True,
)
```

复现要求相同 URDF、软件版本及运行环境；报告中的 SHA-256 可用于核对模型。
主 CLI、单臂 demo 和评测套件的 `--seed` 同时设置采样与 IK 随机重启种子。
单臂 demo 未指定 `--bounds` 时，自动范围估计也使用这个种子。
Cartesian 结果记录 `sampling_seed`、`ik_random_seed` 和 `solver_settings`；
自动估计范围的单臂结果另有 `bounds_source` 与 `bounds_estimation`。
Python API 中可分别设置 `SamplingConfig.seed` 与 `SolverConfig.random_seed`。
当前页面的 **Center plane on current TCP** 可将平面中心定位到当前 FK 位置；
连杆坐标系、采样种子与 IK 种子可分别选择。
计算中捕获不同的 FK 参考配置会取消当前任务，防止旧任务覆盖显示或导出；
最近完成的结果保持可用。重复捕获同一配置不会取消，取消结束后可用新配置重新计算。
`marvin_single_arm.py` 也支持 `--base-link`、`--tip-link` 和 Cartesian 模式的
`--bounds`；识别到标准手臂链名时默认使用纯手臂链，默认随机采样无需 SciPy。

## 实际模型回归

```bash
python -m pip install -e '.[dev,viser]'
make test-robot
```

此入口检查真实 Marvin 左右臂的 float32 / float64 空间 Jacobian、J4 默认姿态，以及两臂的直线、圆、
八字和螺旋轨迹：验证 FK 位置误差、关节限位、连续性和速度限位。
安装 trimesh 后也检查模型 mesh。报告写入 `outputs/marvin-tests.xml`。
缺少 URDF 时明确失败，避免把跳过误当作实际模型已验证。
NumPy 检查必跑；Torch CPU / CUDA 组合在对应可选运行时可用时执行。

`benchmark_marvin.py` 和 `stress_marvin.py` 保留作真实模型性能及稳定性工具。
`reachability_workflow.py` 是可接收实际 URDF 的高级诊断工具；
`benchmark_candidates.py`、`benchmark_perturbations.py`、`benchmark_result_io.py`
及 `tests/fixtures` 中的合成模型主要用于内部回归，不作为可视化 demo 的维护重点。

## 基准与持续数值检查

`benchmark_marvin.py`、`stress_marvin.py` 同样支持 W1、通用 URDF 和显式链端点。
脚本名称为兼容保留，报告以实际 `robot/base_link/tip_link/joint_names` 为准。
主 CLI 的 URDF 是位置参数；示例程序使用 `--urdf`。
基准/压力工具的重启参数是 `--restarts`、`--rescue-restarts`、`--rescue-rounds`；
主 CLI 和平面 demo 使用带 `ik-` 前缀的对应参数。

```bash
python examples/benchmark_marvin.py --urdf tests/fixtures/cartesian_stage.urdf \
  --batch-size 4096 --ik-targets 256 --repeats 5 --warmup 1 \
  --seed 42 --report outputs/benchmark.json

python examples/stress_marvin.py --urdf /path/to/robot.urdf \
  --base-link left_arm_base --tip-link left_ee --duration 60 \
  --batch-size 1024 --ik-targets 32 --min-success-rate 0.99 \
  --report outputs/stress.json
```

基准使用固定随机种子、预热和多次计时的中位数，CUDA 计时前后同步；
IK 目标数量独立于 FK 批大小。压力检查同时验证 FK 旋转正交性、
线速度和角速度 Jacobian 的中心差分、成功 IK 的独立 FK 位置误差及关节限位，
并轮换覆盖所有关节列。正的极短 `--duration` 也至少执行一批。
每 30 秒向 stderr 输出进度，结束后写严格 JSON 报告。
默认 IK 成功率验收为 99%；不满足时保留报告并返回非零退出码。

平面选择、任意方向与指定数量采样，以及离线 `inspect` 复评和逐项验收示例见
[平面任务指南](../docs/plane-reachability.md)。

## Demo 实跑记录

[2026-09-14 检查记录](../docs/demo-check-2026-09-14.md)覆盖全部 13 个脚本、
25 组无界面任务、六组 CUDA 任务及六个浏览器入口，并说明任务成功率与程序退出码的区别。
轨迹、灵巧度、对比和基准入口的非法参数会返回清晰错误；报告路径不能覆盖输入模型。
