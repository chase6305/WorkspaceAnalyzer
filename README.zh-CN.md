# WorkspaceAnalyzer

[English](README.md) | [简体中文](README.zh-CN.md)

![WorkspaceAnalyzer 架构概览](docs/assets/workspace-analyzer-overview.png)

WorkspaceAnalyzer 是一个与仿真器解耦的机器人工作空间分析 Python 库。它能够直接从 URDF 自动构造通用串联机械臂求解器，通过 NumPy 或 PyTorch 批量执行 FK、几何 Jacobian 和数值 IK，并使用 Viser 展示机器人模型与可达性结果。

核心包只依赖 NumPy，不要求 EmbodiChain、Isaac Lab、ROS 或特定 Robot 封装。

**示例维护重点：真实机器人模型。** 从 [Marvin / W1 示例入口](examples/README.md)
运行工作空间、平面可达性、轨迹和机器人对比。合成模型主要用于算法回归测试。

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

如需一次性运行完整可达性套件（关节采样工作空间、position-only IK、固定姿态 IK），
可直接使用：

```bash
python3 examples/reachability_suite.py --urdf /path/to/robot.urdf --arm left --viser
```

固定某个高度（或任意一个坐标）平面上的可达性分析：

```bash
python3 examples/plane_reachability.py \
  --urdf /path/to/robot.urdf --arm left \
  --frame world --plane xy --constant 0.8 --span 0.8 --resolution 41 \
  --position-only --output outputs/plane.json
```

`--frame world` 使用 URDF 根坐标系，上游关节置零；`base` 使用机械臂基座坐标系。
默认 `reference` 使用第二个活动关节子连杆在零关节状态下的坐标系，单关节链使用
第一个子连杆。该参考系在整个扫描中保持固定。
`--plane` 选择 `xy`、`xz` 或 `yz`，`--constant` 覆盖 `--center` 的法向坐标（米）。
中心和 `--rpy`（弧度）均在所选坐标系下定义。`--span` 和 `--resolution` 控制平面
网格。输出 JSON 的 `grid` 行对应平面名称中的第一个轴，列对应第二个轴；
去掉 `--position-only` 可改为固定 TCP 姿态约束。

`--batch-size`（默认 1024）限制每批 IK 目标数，`--seed`（默认 42）控制重启采样。
同一批大小、种子、后端和求解器设置下可重复扫描；改变批大小可能改变重启候选和
可达性分类。报告记录这些设置、实际平面中心及到求解器基座的变换；可视化使用
相同目标坐标和真实求解出的关节配置。

[新功能分析与路线图](docs/feature-roadmap-2026-09-05.md) 给出了通用任务输入、对比
接口、碰撞检查、姿态覆盖图和流式分析的范围、成本与验收标准，并区分已完成的
目标评估基础能力和后续提案。

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

自由度不足 6 或只关心位置时，使用 `position_only=True`。多初值 IK 会把初值合并成一个批次并行求解，然后逐目标选择最低残差结果。离线笛卡尔分析还可通过 `rescue_restarts` 和 `rescue_rounds` 仅对失败目标追加新的确定性随机初值。实时跟踪通常应优先使用上一帧关节状态作为 `seed`。

失败补救采用迭代方式，不再累积递归调用栈；无梯度分析时会释放已完成轮次的中间数组。
`IKResult.iterations` 统计首次求解与所有实际补救轮次的迭代循环次数之和，
不乘以并行起点数量。

### 关节空间工作空间分析

```python
from workspace_analyzer import WorkspaceAnalyzer, WorkspaceConfig

result = WorkspaceAnalyzer(solver, WorkspaceConfig()).analyze()
result.save("workspace.npz")
```

数据流为 `关节采样 -> FK -> 笛卡尔点`，并可为每个点计算平移可操作度。

### 笛卡尔空间可达性分析

上下界相等即可固定坐标轴：`bounds=[[-0.5, 0.5], [-0.5, 0.5], [0.2, 0.2]]`
直接采样 Z = 0.2 米的平面；固定两轴得到线段，固定三轴则按请求数量重复同一点。
所有采样器只对变化轴采样，均匀平面网格不会因固定轴的重复层级浪费样本。
CLI 同样支持：`--bounds -0.5 0.5 -0.5 0.5 0.2 0.2`。

```python
import numpy as np
from workspace_analyzer import CartesianConfig, SamplingConfig, WorkspaceAnalyzer

q_reference = solver.joint_limits.mean(axis=1)
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
restored = type(result).load("cartesian_reachability.npz")
```

数据流为 `XYZ 目标 -> IK -> 可达/不可达分类`。结果包含全部查询点、最佳关节解、逐点可达标记和 IK 残差。

### 直接测试目标点与配置质量

```python
from workspace_analyzer import ReachabilityConfig

assessment = WorkspaceAnalyzer(solver).analyze_targets(
    targets,  # (N, 3) 位置；完整位姿使用 (N, 4, 4) 和 position_only=False
    ReachabilityConfig(minimum_isotropy=0.05, minimum_joint_limit_margin=0.1),
)
print(assessment.metadata["assessment"])
print(assessment.reachable)                # IK 成功
print(assessment.metrics["quality_pass"])  # 选中配置通过质量要求
```

支持任务权重、输入坐标变换、分批计算、取消和缓存；仅为成功解计算灵巧度。
报告包含最小奇异值、各向同性、条件数、关节余量、任务秩，以及分开的位置和旋转
误差。`require_full_rank=True` 可要求完整任务满秩，避免把紧致 SVD 中的非零指标
误读为完整三维或六维控制能力。质量阈值目前评估选中的 IK 解，不追加质量最优
分支搜索，也不包含碰撞检查。

命令行支持无表头 XYZ CSV、NPY，以及包含 `targets`、可选 `weights` 和
`base_from_targets` 的 NPZ：

```bash
workspace-analyzer robot.urdf --backend numpy --mode targets \
  --targets targets.npz --batch-size 1024 \
  --min-isotropy 0.05 --min-joint-limit-margin 0.1 \
  --output assessment.npz --report assessment.json
```

完整位姿测试增加 `--full-pose`，原始目标位姿经坐标变换后随结果保存。
具体字段、单位、复现要求和无需外部资产的测试示例见
[可达性与灵巧度测试指南](docs/reachability-testing.md)。

`assessment.reassess_quality(minimum_joint_limit_margin=0.2)` 可直接利用已保存
测量完整替换质量要求，无需重跑 IK；未传入的门槛关闭。缓存运行同样支持修改
门槛或任务权重后复用测量。完整位姿结果可通过
`assessment.orientation_coverage(position_ids)` 汇总每个位置的采样姿态 IK
覆盖率与质量覆盖率。[完整测试流程](docs/testing-workflow.md) 提供已知目标、
范围外目标、局部多姿态、质量扫描、验收退出码及报告生成。

在 `examples/reachability_workflow.py` 增加 `--robustness-test`，可测试 XYZ 位移
与三轴姿态偏差，报告每个方向的 IK/质量退化、最差成功质量和全部变体通过率。
Python 接口使用 `PosePerturbations`，支持基座轴/工具轴及离线重评。参见
[扰动测试指南](docs/robustness-testing.md) 与 [后续验证方案](docs/validation-options.md)。

`--boundary-test` 增加固定姿态平移边界细化，批量二分成功/失败区间，并复查最终
失败端点。Python 接口 `refine_translation_boundary` 支持任意平移方向，保留
预算不足、精度不足和端点恢复等状态及完整查询记录，详见 [边界测试指南](docs/boundary-testing.md)。

`--stability-test` 对同一批位姿、扰动目标或边界端点比较迭代预算与随机种子。
`IKTrial` / `analyze_ik_stability` 还支持显式关节初值，记录逐项改善、退化和成功
配置的质量变化；保存后可离线重评质量门槛。详见 [稳定性测试指南](docs/stability-testing.md)。

`study.select_solutions(objective="joint_limit_margin")` 可离线筛选已测候选，
优先满足质量门槛，并保存每个目标的来源试验。工作流使用 `--stability-test`
配合 `--select-ik-solutions joint_limit_margin`，支持选后质量通过率验收。
排序规则、来源追溯及重评/重新选择的区别见 [候选解选择指南](docs/candidate-selection.md)。

`make_ik_seed_trials` / `--stability-initial-guesses K` 可生成可复现的关节初值。
`study.summarize_diversity()` / `--candidate-diversity` 按关节类型和容差统计
新增配置与重复候选，处理连续关节角度等价，并保留所有候选参与质量选择。
详见 [候选多样性指南](docs/candidate-diversity.md)。

### 灵活度与关节连续性

可达性是二值判断，灵活度用于描述可达构型的质量。批量 API 可返回 Jacobian
奇异值、Yoshikawa manipulability、最小奇异值、各向同性、条件数以及归一化
关节限位裕量：

```python
quality = solver.dexterity(q_batch, task="position")
vertical_priority = solver.dexterity(
    q_batch, task="position", weights=[0.5, 0.5, 1.0]
)
safe = (quality.minimum_singular_value > 0.05) & (
    quality.joint_limit_margin > 0.15
)
```

`task="position"` 和 `task="rotation"` 可避免线速度与角速度单位混合。只有在应用
确实接受未加权六维 Jacobian 时才应使用 `task="pose"`。可通过非负 `weights`
强调特定任务空间方向；全零权重没有实际任务含义，因此会被拒绝。关节空间分析会在
`result.metrics` 中保存 `minimum_singular_value`、`isotropy` 和
`joint_limit_margin`；Viser 的 `color_metric` 参数可用其中任一指标为点云着色。

Torch 对未选中的指标分支先保护除数，避免零 Jacobian 或已屏蔽的无穷条件数
通过除零污染反向传播。秩不足的条件数仍为无穷大，构建有限损失时需要排除。
验证范围见[第十六轮记录](docs/optimization-2026-09-08-round16.md)。

连续性必须沿有序轨迹分析，不能从无序工作空间点云推断：

```python
trajectory = solver.solve_trajectory(
    target_poses,
    seed=q_start,
    position_only=False,
    failure_restarts=8,
    dt=0.01,
)
print(trajectory.max_joint_jump)
print(trajectory.velocity_violation.any())
print(trajectory.summary())
```

轨迹求解器会以上一帧成功解作为 warm start，仅在失败时使用多初值，自动展开
continuous joint，并报告关节差分、速度、加速度和 URDF 速度限位违规。IK 失败
帧相邻的轨迹段会标记为 `NaN`，不能当作可执行运动。逐帧最小奇异值和关节
限位裕量还可定位连续轨迹中的奇异区或贴限位区段。非等间隔轨迹可使用
`timestamps=[...]` 代替固定 `dt`。

长轨迹可用 `batch_size=1024`（默认值）限制闭环重投影和灵活度指标的计算批次。
失败帧跳过指标计算，诊断保持为 `NaN`；多起点补救成功后，关节跳变检查复用这次结果。
奇异值采用紧致 SVD，数量为 `min(任务行数, DoF)`；条件数使用相对数值秩阈值，
统一缩放全部任务权重时，在数值精度范围内保持不变。

可用下面的命令在 Marvin 单臂上同时运行两类分析，并在 Viser 中切换指标：

```bash
PYTHONPATH=src python examples/marvin_dexterity.py \
  --arm left --samples 10000 --trajectory-frames 200 \
  --color-metric isotropy --viser
```

### Dexforce W1 与 Marvin M6 对比

使用相同口径比较两款单臂，并在同一个 Viser 页面中并列显示：

```bash
python3 examples/compare_w1_marvin.py \
  --arm left --samples 20000 --shared-targets 1000 \
  --trajectory-frames 200 --circle-preset horizontal --viser --autoplay \
  --report outputs/w1_vs_marvin.json
```

求解器保留 `<arm>_arm_base` 到末端的完整 7 轴链，但比较坐标统一使用第 2 个主动
关节 child link 作为结构化 `link2` 基座。这样既保留 full-pose IK 能力，又让工作
空间、目标和臂长定义不依赖品牌特定 link 名。报告包含关节范围、AABB 与凸包体积、公共体素占用体积、重叠率、同样本
灵活度分位数、相同 Cartesian 目标可达率，以及共享圆轨迹连续性。只有确实要比较
整机链时才使用 `--base-link base_link`；此时 W1 会额外包含可动躯干/下肢关节，
不再是纯 7 轴手臂对比。

交互检查使用 `--ik-profile fast`，常规报告使用默认 `balanced`，最终审核使用
`rigorous`。三档对每个共享目标最多分别尝试 10、36、104 个 seeds，迭代上限分别为
120、200、400；报告会记录实际档位及完整求解参数。

严谨版报告采用配对的归一化关节样本，记录请求样本数 25/50/75/100% 时的凸包体积，
并为共享目标可达率给出 Wilson 95% 区间。同一批共享位置分别进行 position-only IK
和采用圆轨迹固定世界姿态的 full-pose IK。凸包只表示外包络；
`common_voxel_grid.occupied_volume` 是更保守的采样占用估计，两者都不能替代包含
碰撞约束的任务可达性。
尺度归一化默认采用 URDF 链长上界。如果经核验的厂家尺寸采用不同口径，可通过
`--w1-arm-length` 和 `--marvin-arm-length` 输入；报告会记录长度来自 URDF 还是
标称覆盖值。体积与位置可操作度除以长度三次方，位置 Jacobian 奇异值除以长度。

使用 `--align-arm-length` 可进行虚拟等臂长比较：link2 相对 Cartesian 坐标乘以
`共同长度/自身长度`，共享目标在 IK 前再按逆比例映射回真实 URDF。共同长度默认取
较短机械臂，也可用 `--aligned-arm-length` 指定。通用别名 `--robot-a-urdf`、
`--robot-b-urdf`、`--robot-a-arm-length` 和 `--robot-b-arm-length` 可脱离型号比较
任意两份 URDF。

```bash
python3 examples/compare_w1_marvin.py \
  --robot-a-urdf robot_a.urdf --robot-b-urdf robot_b.urdf \
  --reference-link-index 2 --align-arm-length \
  --samples 20000 --shared-targets 1000 --report outputs/aligned_arms.json
```

可通过 `--circle-preset` 选择 `horizontal`（世界 XY 水平面）、`vertical_xz`
（世界 XZ 竖直面）、`vertical_yz`（世界 YZ 竖直面）或 `chest_front`（胸前
世界 YZ 正面）。默认圆心是相对 link2 的
世界坐标偏移；可用 `--circle-center-offset X Y Z` 覆盖，并用 `--circle-radius`
设置两台机器人一致的半径。
左臂 `chest_front` 默认圆心偏移为 `[+0.40, -0.10, -0.10] m`：世界 `+X`
向前、`-Y` 向身体中线、`-Z` 向下；默认半径为 `0.08 m`，固定世界姿态为
RPY `[0, +pi/2, 0]`。该位置与姿态已针对两款机器人共同可达性验证。

对比 Demo 默认使用固定世界姿态 IK：TCP 绕圆运动时保持同一个世界旋转。默认旋转
与两款机器人零位 TCP 的共同方向一致（左臂 `roll=-pi/2`，右臂 `roll=+pi/2`）。
可用 `--target-rpy R P Y` 覆盖；如果只分析位置可达性、不约束工具姿态，则使用
`--orientation-mode position-only`。Viser 会沿圆轨迹抽样绘制 RGB 小坐标系，并可
通过 **Trajectory coordinate frames** 开关显示或隐藏。
工作空间点云默认隐藏，以保证机器人和轨迹清晰；需要观察覆盖范围时再打开
**Workspace**。较大的世界原点与 link2 结构参考坐标系保持平行，link2 坐标系仅做
平移，不能将它误解为 URDF 关节的局部坐标轴。
播放时，较大的目标坐标系表示期望 TCP 位姿，机器人 TCP 坐标系表示 FK 实际位姿；
界面逐帧显示位置误差和姿态误差，并以短线连接实际与目标 TCP 位置。
捕获的 `reference_tool` 位姿以 solver-base 坐标保存并用于 IK，但绘制时会应用
`world_from_solver_base`；面板同时列出两种坐标值。双机器人并排显示偏移仅用于
场景排版，不属于机器人运动学变换。

轨迹 IK 使用上一帧 warm start、朝首个成功构型的小幅零空间偏置，并在相邻关节变化
超过 `--jump-repair-threshold` 时进行多初值修复。闭环构型偏置可通过
`--trajectory-posture-gain` 调整（默认 `0.005`）。报告同时给出原始/归一化最大
跳变、关节路径长度、闭环误差、峰值/RMS 速度、峰值加速度、奇异值、关节限位余量
及速度违规。全局关节居中通过 `--joint-centering-gain` 显式开启；固定姿态约束较紧
时，过大的居中增益可能降低收敛率。
圆轨迹默认开启 `--loop-closure`：求解器将首末关节漂移沿整圈分摊，以插值构型作为
逐帧 IK 初值，再投影回原始 Cartesian 位姿。因此关节首末严格闭合，同时不会用
无约束关节插值替代笛卡尔圆；仅在诊断对照时使用 `--no-loop-closure`。

虽然计算只针对单臂，Viser 会从每个完整 `robot.urdf` 加载全部 visual。未参与分析
的关节保持为 0，所选手臂由 IK 轨迹驱动。每款机器人分别提供
Play/Stop、帧、FPS 和循环控制；`--autoplay` 会立即播放安全共享圆轨迹。网页中
同时显示逐机器人轨迹指标和包含主要结论的对比表。

## CLI

结果默认保存为压缩 NPZ。需要更快的本地存储时，可使用
`result.save("workspace.npz", compressed=False)`，或从 `workspace_analyzer`
导入 `ResultCache` 后使用 `ResultCache("cache", compressed=False)`。
CLI 对应 `--cache-dir cache --cache-uncompressed`。两种编码自动加载，共用缓存键；
缓存压缩设置只影响新写入。保存前重新校验当前数组，并原子替换文件。
空间和速度取舍见[第十五轮记录](docs/optimization-2026-09-08-round15.md)。


包内 CLI 同时支持两种分析模式。关节空间分析：

```bash
workspace-analyzer robot.urdf \
  --base-link base_link --tip-link tool0 \
  --backend torch --device cuda \
  --strategy sobol --samples 100000 --batch-size 8192 \
  --output workspace.npz --viser --port 8080
```

显式 bounds、固定姿态和内容寻址缓存的笛卡尔分析：

```bash
workspace-analyzer robot.urdf \
  --mode cartesian --base-link base_link --tip-link tool0 \
  --bounds -0.7 0.7 -0.3 0.9 0.4 1.8 \
  --full-pose --reference-joints 0 0 0 0 0 0 0 \
  --ik-restarts 4 --ik-rescue-restarts 16 --ik-rescue-rounds 3 \
  --samples 20000 --cache-dir .cache/workspace-analyzer \
  --output cartesian.npz --viser
```

缓存键包含求解器实际使用的已解析 URDF 模型、链选择、solver/backend 设置和完整分析配置。修改或删除源文件不会改变已加载模型的缓存标识；使用更新后的运动学模型时，需要重新构造求解器。缓存模式升级会自动触发重新计算。结果使用带版本的 JSON 加数组 NPZ 格式，以 `allow_pickle=False` 读取并原子写入。Viser 是 Web 服务，无需桌面显示服务器，打开进程输出的网址即可。

两种分析模式都支持 `cancel_event` 和 `progress_callback`。IK 与轨迹求解也支持
`cancel_event`；其他线程设置事件后，会在下一个批次或 IK 迭代边界抛出
`AnalysisCancelled`，包含失败补救与闭环重投影。单次 NumPy/Torch 运算完成后才会
检查取消。关闭查看器时会取消后台任务，并阻止迟到结果继续更新场景。

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
viewer.add_workspace(result, color_metric="isotropy")
viewer.wait()
```

轨迹叠加层使用绿色/红色表示各 TCP 目标的 IK 成功/失败；连续有效轨迹段按最小
奇异值着色。在 `wait()` 前调用 `viewer.add_trajectory(trajectory)` 即可添加。

Display 面板可独立控制机器人、骨架、TCP、工作空间、轨迹目标和轨迹路径的显示，
并可调整工作空间点大小、可达/不可达透明度、轨迹目标大小、轨迹线宽，以及通过带
分位数图例的下拉菜单即时切换工作空间着色指标。多个 viewer 可通过
`server=first_viewer.server` 共享同一服务，并分别设置 `workspace_root`、
`gui_label` 和 `scene_offset`。

### 轨迹案例集

案例以 FK 参考 TCP 坐标系定义偏移，同时支持 position-only 和固定参考方向的
full-pose IK：

```bash
# 直线
python3 examples/trajectory_gallery.py --shape line --viser

# 保持参考方向的圆
python3 examples/trajectory_gallery.py --shape circle --full-pose --viser

# 8 字轨迹
python3 examples/trajectory_gallery.py --shape figure8 --viser

# Dexforce W1 三维螺旋
python3 examples/trajectory_gallery.py \
  --urdf /home/ubuntu/workspace/chase/HumanoidAssets/Dexforce_W1_V3/robot.urdf \
  --shape helix --viser
```

可通过 `--scale`、`--depth`、`--frames` 和 `--dt` 主动构造可达性、奇异区、
关节跳变和速度限位失败案例。

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

加入 `--full-pose` 可同时约束 FK 参考姿态；未提供 `--reference-joints` 时，Marvin J4 默认弯肘 90°（对应此 URDF 的关节值 −90°），其余关节使用限位中心。显式传入的参考关节值单位为弧度。使用 `--arm right` 切换右臂。

可视化启动时自动取景并显示工作空间。**Reset to initial pose** 恢复启动姿态，包括显式指定的参考关节值。展开 **Display** 可调整显示选项，点击 **Frame robot and workspace** 可重新取景。双机器人对比 demo 仍默认隐藏点云，方便查看轨迹。

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

benchmark 默认预热一次，再统计五次运行耗时的中位数，可用 `--warmup` 和
`--repeats` 调整。报告包含随机种子和运行时版本；IK 独立采样准确的
`--ik-targets` 数量，不受 FK 批次大小限制。CUDA 会在每次计时前后显式同步。

持续随机回归：

```bash
PYTHONPATH=src python examples/stress_marvin.py \
  --duration 3600 --backend numpy --report outputs/stress.json
```

压力测试会持续检查 FK 旋转正交性、解析 Jacobian 有限差分误差和随机 full-pose IK 成功率。

在参考 CPU 环境的一小时 NumPy/float64 压力测试中，共检查 24,720,384 个 FK
姿态和 386,256 个随机可达全姿态 IK 目标。IK 成功率为 99.9702%，成功解最大
残差低于 `1e-5`，最大旋转正交误差为 `1.45e-15`，位置 Jacobian 相对有限差分
的最大误差为 `3.78e-9`。这些数据用于回归验证，不代表与硬件无关的性能保证；
部署前应在目标机器上重跑上述命令。

笛卡尔 IK 的内存占用大致随 `batch_size * restarts` 增长。Marvin 示例默认使用
兼顾速度和内存的 2048；CUDA 或大内存机器可调大，内存受限时应调小。

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
- IK 每轮共用一次 FK/Jacobian 遍历，NumPy 保持所选的 `float32` 或 `float64`
  计算精度；包括迭代次数耗尽的情况，返回残差始终对应返回的关节位置。
- 已收敛的 IK 起点退出计算批次，其余起点继续迭代，保留每个目标的最小残差和
  最近种子选择语义。
- `continuous` 关节在 IK 中按周期归一化，不会被夹在 ±π。轨迹对成功帧展开角度，
  避免失败帧引入虚假的整圈转动；相邻失败段仍标记为 `NaN`。
- 闭环重投影使用批量 IK；近似重合的首尾目标保留各自的解与残差，只有完全相同
  的目标才复用首帧解。
- 单个 `(4, 4)` IK 目标返回 `(DoF,)` 解；`(N, 4, 4)` 批次始终返回
  `(N, DoF)`，包含只有一个目标的批次。
- 分析流程预分配结果数组并逐批写入。均匀网格仅生成请求数量的前缀，保留原网格
  顺序，内存与 `num_samples * dimensions` 成正比；截断的网格可能无法均匀覆盖每个轴。
- IK 是带限位的通用数值 DLS，不是特定构型闭式解。
- 使用 IK 结果前必须检查 `success` 和 `residual`。
- 奇异位形、严格关节限位和远初值可能需要多初值求解。
- 验证建议使用 `float64`，大批量 CUDA 通常使用 `float32`。
- Viser 不应进入硬实时控制关键路径。

Marvin 集成测试会通过有限差分验证解析位置 Jacobian。实际硬件吞吐应运行 `examples/benchmark_marvin.py` 测量。

如需评估名义圆轨迹附近的敏感性，可加 `--robustness-test`。程序会测试圆心各轴 ±2 cm、
半径 ±10%，以及固定世界姿态下 RPY 各轴 ±5° 的扰动。JSON 报告会保存每个扰动案例，
并汇总每台机器人最差成功率、最大关节跳变、最小奇异值和最小关节限位裕量；可用
`--robustness-position-delta`、`--robustness-radius-fraction`、`--robustness-angle-deg`
调整扰动幅度。

```bash
python3 examples/compare_w1_marvin.py --arm left --align-arm-length \
  --circle-preset chest_front --robustness-test --viser --autoplay \
  --report outputs/w1_vs_marvin_robustness.json
```

## 开发

```bash
python -m pip install -e '.[all,dev]'
make check
```

外部 Marvin URDF 不存在时，对应集成测试会自动跳过。开发流程参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

`make check` 依次运行代码检查、pytest、解析机器人流程及打包，JUnit 与机器人
报告保存在 `outputs/`。单独运行机器人验收使用 `make test-workflow`；验收失败
时保留诊断数据并返回非零退出码。CI 分别配置 NumPy 核心环境和
Torch CPU/SciPy/Viser 可选环境，并留存测试报告。

## 工程结构

```text
src/workspace_analyzer/
  model.py           URDF 模型与链选择
  kinematics.py      NumPy/Torch FK、Jacobian 与 IK
  sampling.py        采样策略
  metrics.py         灵活度与关节限位质量指标
  trajectory.py      有序轨迹 IK 与连续性诊断
  analyzer.py        关节空间与笛卡尔空间分析
  reachability.py    指定目标评估与离线质量判定
  coverage.py        按位置汇总姿态覆盖
  robustness.py      配对位姿扰动与鲁棒性统计
  boundary.py        平移区间自适应细化与端点复查
  stability.py       IK 预算、初值及配置质量的配对比较
  selection.py       已测 IK 候选的质量优先筛选与来源追溯
  diversity.py       可复现关节初值与已测配置分组
  visualization.py   Viser 与 URDF visual 加载
  cli.py             命令行入口
examples/             Marvin 工作空间、灵活度、轨迹与 benchmark
tests/                单元测试与可选资产集成测试
```

## 许可证

Apache License 2.0，参见 [LICENSE](LICENSE)。
