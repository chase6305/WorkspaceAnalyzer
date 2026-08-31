# WorkspaceAnalyzer

[English](README.md) | [简体中文](README.zh-CN.md)

![WorkspaceAnalyzer 架构概览](docs/assets/workspace-analyzer-overview.png)

WorkspaceAnalyzer 是一个与仿真器解耦的机器人工作空间分析 Python 库。它能够直接从 URDF 自动构造通用串联机械臂求解器，通过 NumPy 或 PyTorch 批量执行 FK、几何 Jacobian 和数值 IK，并使用 Viser 展示机器人模型与可达性结果。

核心包只依赖 NumPy，不要求 EmbodiChain、Isaac Lab、ROS 或特定 Robot 封装。

## 功能

- 自动处理 URDF 根节点、末端、活动关节、固定变换、关节轴和限位。
- NumPy CPU 与 PyTorch CPU/CUDA 使用统一批量 API。
- 解析 FK、解析几何 Jacobian，以及带限位的阻尼最小二乘 IK。
- 多初值并行 IK，并为每个目标自动选择最低残差结果。
- 关节空间分析：采样关节位置，通过 FK 映射到笛卡尔空间。
- 笛卡尔空间分析：直接采样 XYZ，通过批量 IK 分类可达与不可达点。
- Random、grid、Gaussian、Halton、Sobol 和 Latin Hypercube 采样。
- 平移可操作度指标。
- Viser URDF 模型、关节控制、骨架、TCP 和工作空间点云。
- Torch、Viser 和 SciPy 均作为可选依赖管理。

## 环境要求与安装

- Python 3.10+
- NumPy 1.24+
- 可选：PyTorch 2.1+、SciPy 1.10+、Viser、trimesh

```bash
# 仅安装 NumPy 核心
python -m pip install -e .

# 按需安装
python -m pip install -e '.[torch]'
python -m pip install -e '.[viser]'
python -m pip install -e '.[sampling]'

# 完整运行及开发环境
python -m pip install -e '.[all,dev]'
```

CUDA 版本由已安装的 PyTorch 决定。本项目不固定 CUDA wheel；应先根据目标机器驱动安装合适的 PyTorch，再安装本项目。

## 快速开始

### 自动构造求解器

```python
from workspace_analyzer import create_solver

solver = create_solver(
    "robot.urdf",
    base_link="base_link",  # 单根 URDF 可省略
    tip_link="tool0",       # 可省略，默认选择活动关节最多的链
    backend="torch",        # auto / numpy / torch
    device="cuda",          # auto / cpu / cuda / cuda:1
    dtype="float32",
)

poses = solver.forward(q_batch)                 # (N, 4, 4)
jacobians = solver.jacobian(q_batch)            # (N, 6, DoF)
ik = solver.inverse(target_poses, seed=q_seed)
robust_ik = solver.inverse(target_poses, restarts=4)

if not bool(ik.success.all()):
    print("部分 IK 目标未收敛", ik.residual)
```

自由度不足 6 或只关心位置时，使用 `position_only=True`。多初值 IK 会把初值合并成一个批次并行求解，然后逐目标选择最低残差结果。实时跟踪通常应优先使用上一帧关节状态作为 `seed`。

### 关节空间工作空间分析

```python
from workspace_analyzer import WorkspaceAnalyzer, WorkspaceConfig

result = WorkspaceAnalyzer(solver, WorkspaceConfig()).analyze()
result.save("workspace.npz")
```

数据流为 `关节采样 -> FK -> 笛卡尔点`，并可为每个点计算平移可操作度。

### 笛卡尔空间可达性分析

```python
import numpy as np
from workspace_analyzer import CartesianConfig, SamplingConfig, WorkspaceAnalyzer

config = CartesianConfig(
    bounds=np.array([
        [-0.7, 0.7],
        [-0.3, 0.9],
        [0.4, 1.8],
    ]),
    sampling=SamplingConfig(num_samples=20_000, batch_size=1024),
    position_only=True,
    restarts=4,
    reference_joints=q_reference,
    reference_pose=solver.forward(q_reference),
)
result = WorkspaceAnalyzer(solver).analyze_cartesian(config)
print(result.metadata["success_rate"])
result.save("cartesian_reachability.npz")
```

数据流为 `XYZ 目标 -> IK -> 可达/不可达分类`。结果包含全部查询点、最佳关节解、逐点可达标记和 IK 残差。

## CLI

包内 CLI 执行关节空间分析：

```bash
workspace-analyzer robot.urdf \
  --base-link base_link --tip-link tool0 \
  --backend torch --device cuda \
  --strategy sobol --samples 100000 --batch-size 8192 \
  --output workspace.npz --viser --port 8080
```

笛卡尔空间流程可使用下文的 Marvin 示例。Viser 是 Web 服务，无需桌面显示服务器，打开进程输出的网址即可。

## Viser 可视化

支持以下内容：

- URDF mesh、box、cylinder 和 sphere visual。
- visual 原点、RPY、mesh scale 和 GLB 材质。
- 实时关节滑条与复位按钮。
- 独立控制机器人模型、骨架、关节点、TCP 和工作空间显隐。
- FK 工作空间按 manipulability 着色。
- 笛卡尔分析中绿色表示可达，红色表示不可达。
- 超过 25 万点时仅对显示数据进行确定性降采样，不修改分析结果。

```python
from workspace_analyzer.visualization import ViserWorkspace

viewer = ViserWorkspace(solver, port=8080)
viewer.add_workspace(result)
viewer.wait()
```

## Marvin M6 单臂示例

示例默认引用外部资产：

```text
/home/ubuntu/workspace/chase/HumanoidAssets/Marvin_M6_S_CCS_696_V4.0/robot.urdf
```

在其他机器上可通过 `--urdf` 覆盖。运动链包含躯干固定变换，但只采样所选手臂的 7 个关节。

关节空间分析：

```bash
PYTHONPATH=src python examples/marvin_single_arm.py \
  --mode joint --arm left --backend torch --device auto \
  --samples 100000 --batch-size 8192 --viser
```

笛卡尔位置可达性：

```bash
PYTHONPATH=src python examples/marvin_single_arm.py \
  --mode cartesian --arm left --backend torch --device auto \
  --samples 20000 --batch-size 1024 --ik-restarts 4 --viser
```

加入 `--full-pose` 可同时约束 FK 参考姿态；未提供 `--reference-joints` 时默认使用关节限位中心。使用 `--arm right` 切换右臂。

完整位姿分析可以显式提供可复现的参考关节：

```bash
PYTHONPATH=src python examples/marvin_single_arm.py \
  --mode cartesian --arm left --full-pose --viser \
  --reference-joints 0.0 0.2 -0.4 0.0 0.3 0.0 0.0
```

参考关节先通过 FK 得到 `R_ref`，每个笛卡尔目标构造为 `T_target = [R_ref, p_sample]`。同一组关节也是 IK 的第一组 seed，其余 restart 使用确定性随机初值。在 Viser 中可拖动关节滑条，点击 **Capture current FK pose**，再点击 **Recompute Cartesian reachability** 交互式重新计算。Display 面板可分别拖动工作空间点大小、可达点透明度和不可达点透明度。

可复现性能测试：

```bash
PYTHONPATH=src python examples/benchmark_marvin.py \
  --backend numpy --batch-size 4096 --ik-targets 256 --restarts 4
```

CUDA benchmark 会在计时边界显式同步，避免异步 kernel 造成虚高。

## 自动构造规则

1. 解析 URDF link、joint、origin、axis、类型和限位。
2. 未指定 `base_link` 时要求 URDF 只有一个根。
3. 未指定 `tip_link` 时选择活动关节数最多的叶节点。
4. 固定关节保留在变换链中。
5. revolute、continuous 和 prismatic 关节作为求解变量。
6. `backend="auto"` 在 Torch 可导入时选择 Torch；只有 CUDA 可用时才选择 CUDA。

双臂、多分支或多末端机器人应为每个 `(base_link, tip_link)` 分别构造 solver。solver 之间不共享可变运动学状态。

## 精度与性能说明

- FK 和几何 Jacobian 均为解析批量实现。
- IK 是带限位的通用数值 DLS，不是特定构型闭式解。
- 使用 IK 结果前必须检查 `success` 和 `residual`。
- 奇异位形、严格关节限位和远初值可能需要多初值求解。
- 验证建议使用 `float64`，大批量 CUDA 通常使用 `float32`。
- Viser 不应进入硬实时控制关键路径。

Marvin 集成测试会通过有限差分验证解析位置 Jacobian。实际硬件吞吐应运行 `examples/benchmark_marvin.py` 测量。

## 开发

```bash
python -m pip install -e '.[all,dev]'
pytest -q
ruff check src tests examples
python -m build
```

外部 Marvin URDF 不存在时，对应集成测试会自动跳过。开发流程参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 工程结构

```text
src/workspace_analyzer/
  model.py           URDF 模型与链选择
  kinematics.py      NumPy/Torch FK、Jacobian 与 IK
  sampling.py        采样策略
  analyzer.py        关节空间与笛卡尔空间分析
  visualization.py   Viser 与 URDF visual 加载
  cli.py             命令行入口
examples/             Marvin 分析与 benchmark
tests/                单元测试与可选资产集成测试
```

## 许可证

Apache License 2.0，参见 [LICENSE](LICENSE)。
