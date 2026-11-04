# WorkspaceAnalyzer

[English](README.md) | [简体中文](README.zh-CN.md)

直接从 URDF 分析机器人工作空间：采样关节配置、用 IK 检查笛卡尔目标，
通过 Viser 在浏览器中查看机器人和分析结果。
核心仅依赖 **Python 3.10+ 和 NumPy**，无需仿真器或 ROS。

## 先选分析方式

| 想回答的问题 | 模式 | 计算流程 |
| --- | --- | --- |
| TCP 能到哪些位置？ | `joint` | 关节采样 → FK → 工作空间点云 |
| 一个空间范围内哪些点可达？ | `cartesian` | XYZ 采样 → IK → 可达性 |
| 能完成我给定的任务点或位姿吗？ | `targets` | 输入目标 → IK + 构型质量 |
| 工作台上哪些区域可达？ | `plane` | 平面采样 → IK + 质量分布 |

位置单位为米，关节角和 RPY 单位为弧度。IK 失败表示**在当前求解预算内未找到解**。
结果不包含碰撞检查、动力学或实机验证。

## 安装

在当前仓库中执行：

```bash
python -m pip install -e .
```

按需添加功能：

| 安装命令 | 增加的功能 |
| --- | --- |
| `python -m pip install -e '.[viser]'` | 浏览器可视化与 URDF mesh 加载 |
| `python -m pip install -e '.[torch]'` | PyTorch CPU / CUDA 后端 |
| `python -m pip install -e '.[sampling]'` | Sobol、Halton、拉丁超立方采样 |
| `python -m pip install -e '.[all,dev]'` | 全部运行依赖与开发工具 |

CUDA 需要匹配的 PyTorch 安装。初次运行建议显式使用 `--backend numpy`；
`auto` 会在已安装 Torch 时选择 Torch。

## 跑一个真实机器人示例

Marvin M6 和 Dexforce W1 示例使用独立的 `HumanoidAssets` 资产仓库。
将其放在本仓库同级目录，或指定位置：

```bash
export HUMANOID_ASSETS=/path/to/HumanoidAssets
python -m pip install -e '.[viser]'
python examples/marvin_single_arm.py --backend numpy --samples 20000 --viser
```

打开程序输出的网址，默认是 `http://localhost:8080`。
使用 `--arm right` 切换右臂。平面扫描先打开控制界面，
选好工作平面后点击 **Compute plane reachability**：

```bash
python examples/plane_reachability.py --backend numpy --resolution 21 --viser
```

自己的 URDF 可以直接使用主 CLI。双臂或多末端机器人应显式选择运动链：

```bash
workspace-analyzer /path/to/robot.urdf \
  --base-link base_link --tip-link tool0 \
  --backend numpy --samples 20000 --output outputs/workspace.npz --viser
```

资产路径、轨迹播放、机器人对比和性能测试见[真实机器人示例](examples/README.md)。
没有外部资产时，可在仓库根目录用小型回归模型检查安装：

```bash
workspace-analyzer tests/fixtures/cartesian_stage.urdf \
  --backend numpy --samples 1000 --output outputs/smoke.npz
```

## Python 用法

```python
import workspace_analyzer as wa

solver = wa.create_solver("robot.urdf", backend="numpy")
q = solver.joint_limits.mean(axis=1)
pose = solver.forward(q)          # (4, 4)
jacobian = solver.jacobian(q)     # (6, DoF)
ik = solver.inverse(pose, seed=q)
# 使用 ik.positions 前检查 IK 是否成功。
assert bool(ik.success)

config = wa.WorkspaceConfig(
    sampling=wa.SamplingConfig(num_samples=10_000)
)
analyzer = wa.WorkspaceAnalyzer(solver, config)
result = analyzer.analyze()
result.save("workspace.npz")
print(result.points.min(0), result.points.max(0))
```

批量关节输入 `(N, DoF)` 对应位姿 `(N, 4, 4)` 和 Jacobian `(N, 6, DoF)`。
NumPy 求解器返回数组；Torch 求解器保留张量和自动求导。
分析结果统一为 NumPy 数组，便于存储和生成报告。
笛卡尔分析、任务质量、轨迹、缓存和取消见 [Python 指南](docs/python-api.md)。

## 保存和复查结果

四种 CLI 模式都可输出 NPZ 测量数据和 JSON 报告。
例如，无界面扫描 Z = 0.2 m 的 XY 平面：

```bash
workspace-analyzer /path/to/robot.urdf --backend numpy --mode plane \
  --plane xy --constant 0.2 --center 0.4 0 0.2 --span 0.6 --resolution 21 \
  --output outputs/plane.npz --report outputs/plane.json --plot outputs/plane.svg

workspace-analyzer inspect outputs/plane.npz
```

`inspect` 直接读取测量数据，无需加载机器人或重新运行 IK。
对于目标和平面分析，还能离线调整质量门槛：

```bash
workspace-analyzer inspect outputs/plane.npz \
  --min-isotropy 0.05 --min-joint-limit-margin 0.1 \
  --output outputs/reassessed.npz
```

可达性记录 IK 是否收敛；质量门槛评价返回构型的灵活度和关节限位余量。
调整门槛会复用已测数据，不会搜索其他 IK 分支。

全部选项见 `workspace-analyzer --help`。
坐标系与姿态约束见[平面指南](docs/plane-reachability.md)，
目标文件和权重见[可达性测试](docs/reachability-testing.md)。

## 架构与开发

主流程是 **URDF 模型 → 批量求解器 → 分析 → 保存结果 / 查看器**。
NumPy 是唯一必需的运行依赖；可选库按需加载，运动学和分析不依赖渲染。

模块职责和依赖边界见[架构说明](docs/architecture.md)。高级验证分别见
[扰动测试](docs/robustness-testing.md)、[边界测试](docs/boundary-testing.md)、
[IK 稳定性](docs/stability-testing.md)和[候选解筛选](docs/candidate-selection.md)。

```bash
python -m pip install -e '.[all,dev]'
make check
```

`make check` 执行代码检查、测试、解析模型工作流和打包。
`make test-robot` 要求真实 Marvin 资产。
开发约定见 [CONTRIBUTING.md](CONTRIBUTING.md)，验收流程见[测试指南](docs/testing-workflow.md)。

Apache License 2.0，见 [LICENSE](LICENSE)。
