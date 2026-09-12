# 可达性与灵巧度测试

`WorkspaceAnalyzer.analyze_targets` 测试调用者提供的目标点或位姿，并输出 IK
结果、关节配置、误差、灵巧度指标和质量判定。目标可以来自平面网格、实际抓取点、
轨迹路点或预先生成的姿态集合，不需要先构造一个三维包围盒。

一条命令执行回归、机器人验收与报告生成，见 [完整测试流程](testing-workflow.md)。

## 命令行使用

仓库内置了一个解析三轴平移机构和五个目标，可直接验证测试流程：

```bash
PYTHONPATH=src python -m workspace_analyzer.cli \
  tests/fixtures/cartesian_stage.urdf --backend numpy \
  --mode targets --targets examples/targets/cartesian_stage.csv \
  --batch-size 2 --ik-restarts 1 --ik-rescue-restarts 0 \
  --min-isotropy 0.5 --min-joint-limit-margin 0.1 --require-full-rank \
  --output outputs/stage_assessment.npz --report outputs/stage_assessment.json
```

这个机构的解析位置范围为 `[-1, 1]³`，位置 Jacobian 为单位矩阵。五个目标中，
四个在范围内；接近限位和位于限位的两个目标未达到 0.1 的关节余量要求，因此
最终质量达标数为两个。超出范围的目标属于 IK 失败，和质量不达标分开统计。

对实际机器人，替换 URDF 和目标文件，需要时指定 `--base-link` 与 `--tip-link`。
`--viser` 可显示目标的 IK 可达性，并在面板中分别列出质量达标与不达标数量。
当前点云绿色仍表示 IK 成功，不代表已通过质量或碰撞检查。

输入格式如下：

| 格式 | 内容 |
| --- | --- |
| `.csv` | 无表头的 `x,y,z`，每行一个位置目标 |
| `.npy` | `(N, 3)` 位置或 `(N, 4, 4)` 齐次位姿 |
| `.npz` | 必需 `targets`；可选 `(N,)` 的 `weights` 和 `(4, 4)` 的 `base_from_targets` |

单个 `(3,)` 或 `(4, 4)` 目标也可以直接传入。空任务集被拒绝。完整位姿测试使用
`--full-pose`，此时必须提供位姿数组。位置测试不会把位姿的旋转部分当作约束。
NPY/NPZ 读取不启用 pickle。

`--samples` 和 `--strategy` 属于采样模式，不会重采样目标文件；目标模式保留所有
输入目标及原始行顺序。`--seed` 控制 IK 重启种子，`--batch-size` 控制每批目标数。

## Python 接口

```python
import numpy as np
from workspace_analyzer import ReachabilityConfig, ResultCache, WorkspaceAnalyzer

targets = np.array([[0.3, 0.1, 0.4], [0.4, 0.1, 0.5]])
config = ReachabilityConfig(
    position_only=True,
    batch_size=1024,
    restarts=4,
    rescue_restarts=8,
    rescue_rounds=2,
    dexterity_task="position",
    minimum_isotropy=0.05,
    minimum_joint_limit_margin=0.10,
)
result = WorkspaceAnalyzer(solver).analyze_targets(
    targets,
    config,
    weights=[2.0, 1.0],
    base_from_targets=np.eye(4),
    cache=ResultCache("outputs/task_cache"),
)
print(result.metadata["assessment"])
print(result.reachable)                 # IK 是否成功
print(result.metrics["quality_pass"])   # 选中解是否达到全部质量要求
result.save("outputs/task_assessment.npz")
```

`seed` 可选 `(DoF,)`、`(1, DoF)` 或 `(N, DoF)`；二维逐目标初值与输入行对应。
`cancel_event` 与 `progress_callback` 的用法和已有工作空间分析一致，取消后不发布
结果、不写入新的缓存。成功配置的 FK 与 Jacobian 共用遍历；失败目标不计算
灵巧度，相关值保留 `NaN`。

`base_from_targets` 将输入坐标转换到求解器基座，默认单位变换。位置单位遵循
URDF 的米，姿态使用旋转矩阵；`result.points` 和可选 `result.target_poses` 都保存
基座坐标下的目标。位姿输入的完整目标矩阵随结果保存，旧的结果文件仍可加载。

## 指标与判定口径

| 输出 | 含义 |
| --- | --- |
| `reachable` | 给定求解预算下 IK 是否找到成功配置 |
| `residual` | IK 使用的任务误差范数；全位姿时不是单独的位置误差 |
| `metrics["position_error"]` | 返回配置的 TCP 平移误差，米，包括失败目标 |
| `metrics["rotation_error"]` | 完整位姿测试的旋转夹角误差，弧度；位置任务为 `NaN` |
| `manipulability` | 所选任务 Jacobian 的紧致 SVD 奇异值乘积 |
| `minimum_singular_value` | 最小紧致奇异值，反映局部弱方向能力，受单位和权重影响 |
| `isotropy` | 最小与最大紧致奇异值之比，范围 `[0, 1]` |
| `condition_number` | 使用相对秩阈值的条件数，秩亏时为无穷大 |
| `joint_limit_margin` | 最差关节的归一化余量：中心为 1、限位为 0；连续关节不受该限位惩罚 |
| `task_rank` | 加权任务 Jacobian 的数值秩；IK 失败时为 -1 |
| `quality_pass` | IK 成功且通过所有配置的质量阈值 |
| `task_weight` | 原始任务权重，用于加权成功率与质量达标率 |

`dexterity_task` 可选 `position`、`rotation`、`pose`。前两者任务维度为 3，后者为 6。
少自由度机器人的紧致 SVD 指标可能大于零，但不代表完整任务可控。
`require_full_rank=True` 明确要求数值秩等于任务维度；例如三轴平移机构不能通过
六维位姿任务的满秩要求，即使能到达某个特定姿态。

`dexterity_weights` 对 Jacobian 行缩放，与目标的 `weights` 含义不同。前者长度为
3 或 6，必须非负且至少一个正值；零权重使该方向的导数为零，但不降低声明的任务
维度，因此会影响满秩判定。全位姿 Jacobian 混合线性与角运动，比较不同机器人时
需要统一尺度；不要直接把其最小奇异值当成无单位指标。

可选阈值为 `minimum_singular_value`、`minimum_isotropy`、
`minimum_joint_limit_margin`，以及 `require_full_rank`。阈值要求对应指标为有限值，且大于或等于门槛；
未设置任何质量要求时，`quality_pass` 与 `reachable` 相同。

质量门槛评估的是**已选中的 IK 配置**，本轮没有为了提高质量继续搜索其他 IK
分支。因此“质量未达标”不能证明不存在更好的配置。数值 IK 失败也不证明目标
在数学上不可达。报告明确记录 `quality_scope="selected_ik_solution"` 和
`collision_checked=false`。

启用门槛时，对应指标中的 `NaN` 和正负无穷大均计为该门槛失败。原始数值
继续保留，IK 成功标记不改变；关闭某指标的门槛后，该指标不参与质量判定。
条件数在秩亏时仍可为无穷大，这是独立诊断值。
旧候选池若保存了正无穷指标的达标标记，先调用 `study.reassess_quality(...)`
明确提供完整门槛，再重新选择或分析多样性。见[第十七轮记录](optimization-2026-09-08-round17.md)。

## 离线调整质量门槛

保存的目标评估可以直接重评，无需加载机器人或再次执行 IK/FK/Jacobian：

```python
from workspace_analyzer import AnalysisResult

original = AnalysisResult.load("outputs/task_assessment.npz")
stricter = original.reassess_quality(
    minimum_isotropy=0.05,
    minimum_joint_limit_margin=0.20,
)
print(stricter.metadata["assessment"])
```

每次调用**完整替换质量要求**：未传入的阈值关闭，`require_full_rank` 默认关闭。
`original.reassess_quality()` 因而得到仅要求 IK 成功的判定。`weights=None` 保留
原始任务权重，也可传入新的 `(N,)` 权重。原始结果的门槛、判定与元数据不会改变；
新结果共享原始关节解和测量数组以避免复制，调用者应将这些共享数组视为只读。
离线接口不能改变 Jacobian 行权重或任务维度，这些变化需要重新测量。

## 按位置统计姿态覆盖

对 `position_only=False` 的完整位姿结果，可使用输入行对应的位置 ID 分组：

```python
poses = np.repeat(np.eye(4)[None], 3, axis=0)
poses[:, :3, 3] = [0.3, 0.1, 0.4]
poses[1, :3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
poses[2, :3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
pose_result = WorkspaceAnalyzer(solver).analyze_targets(
    poses, ReachabilityConfig(position_only=False)
)
coverage = pose_result.orientation_coverage(["station_a"] * 3)
print(coverage.summary())
print(coverage.to_dict())
```

ID 必须为 `(N,)` 的整数或字符串数组，输出保留 ID 首次出现的顺序，允许交错分组。
同组位置必须一致，默认欧氏距离容差 `position_tolerance=1e-8` 米；只约束位置
的 IK 结果不能用于姿态覆盖统计。每组返回样本数、IK/质量通过数，以及原始和
任务权重加权的覆盖率。组内总权重为零时，加权率在数组中为 `NaN`，JSON 中为
`null`，原始覆盖率仍有效。

覆盖率描述提供的有限姿态样本。重复姿态按输入次数计数，不自动去重，也不代表
完整 SO(3) 覆盖。重新评估质量后，可再次调用此方法更新质量覆盖率。

## 报告与复现

`metadata["assessment"]` 包含原始及加权 IK 成功率、质量达标率、失败/拒绝数量、
各阈值失败数量、任务满秩数量和成功配置的指标分位数。阈值失败统计可重叠；
`quality_rejected_count` 则是按目标去重后的数量。加权统计允许零权重，要求
至少一个正权重，并先缩放权重以避免有限大数求和溢出。

分位数只使用有限值，同时单独报告非有限值计数；无成功解时统计值为 `null`。
JSON 报告不写入非标准的 `NaN` 或 `Infinity`，数组中的失败诊断仍保留原值。

测量缓存标识包含目标内容、初值、坐标变换、模型、求解配置、灵巧度任务和
Jacobian 行权重；大数组按二进制摘要处理，避免展开成 Python 列表。修改质量
门槛或任务权重时复用 IK 与灵巧度测量，返回按当前要求重新计算的判定与摘要。
`metadata["cache_key"]` 标识测量数据，因此不同门槛可共用同一个键。
旧目标缓存使用不同命名空间，升级后首次运行会重新测量。

重复测试应固定目标顺序、batch size、随机种子、求解器配置、后端和精度。当前
每批使用 `random_seed + start_index`，改变批大小可能改变重启候选。输入输出仍
整体保存在内存中，这个接口不等于流式处理或断点续算。

可继续扩展的测试能力见 [路线图](feature-roadmap-2026-09-05.md)：全局姿态采样与
覆盖图、工作空间边界扫描、扰动鲁棒性、多分支质量搜索与碰撞检查。

## 输入精度转换

目标、初值和坐标变换会按求解器精度转换；转换后必须仍为有限值。
例如 `1e100` 在 float64 中有限，但 float32 不能表示，会在计算和缓存读取前报错。
方向权重也按求解器精度校验：转换后必须有限、非负且至少有一个正值；
非零权重全部下溢到零时会拒绝，不能静默变成无任务约束。
一般的浮点舍入仍会发生，部分极小分量也可能下溢；此检查不保证无损转换。
离线任务权重使用 float64，同样拒绝转换后非有限或全部失去正权重的输入。
参见[第十八轮记录](optimization-2026-09-08-round18.md)。

## 损坏结果与缓存恢复

直接调用 `AnalysisResult.load(...)` 时，损坏的 ZIP/DEFLATE 数据、缺失必要字段
或误传 NPY 文件会返回明确的 `ValueError`；文件不存在仍为 `FileNotFoundError`。
`ResultCache` 将这些无效条目视为未命中，由分析器重算并原子写回有效结果。
读取仍禁用 pickle。见[第二十一轮记录](optimization-2026-09-08-round21.md)。

## 派生报告的一致性

姿态覆盖、扰动报告和稳定性报告会核对当前指标、完整质量门槛与 `quality_pass`
是否一致。手动修改测量、标记或门槛后，过期结果会被拒绝；先显式调用
`result.reassess_quality(...)` 或 `study.reassess_quality(...)`，再生成报告。
重评参数完整替换门槛。校验本身不修改来源，也不重跑运动学或统计分位数。
参见[第二十三轮记录](optimization-2026-09-08-round23.md)。

## 取消离线覆盖统计

`result.orientation_coverage(ids, cancel_event=event)` 和底层
`summarize_orientation_coverage(..., cancel_event=event)` 接受 `threading.Event`。
取消检查覆盖输入读取前、分组后、聚合阶段及返回前；取消抛出 `AnalysisCancelled`。
单次 NumPy 运算会先完成，再到下一检查点响应。见[第二十四轮记录](optimization-2026-09-08-round24.md)。
