# 候选初值生成与配置多样性测试

`make_ik_seed_trials` 生成可复现的关节初值，`study.summarize_diversity()` 统计
按关节容差区分的已测配置。两者配合现有稳定性分析和质量选择，可以检查增加
初值后是否找到了不同配置，以及这些配置是否满足原质量门槛。

## 工作流

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf tests/fixtures/cartesian_stage.urdf \
  --stability-test --stability-initial-guesses 2 \
  --candidate-diversity --select-ik-solutions joint_limit_margin \
  --output-dir outputs/diversity-workflow
```

`--stability-initial-guesses K` 为每个预算/重启种子组合增加 K 个显式关节初值，
同时保留原来的默认初值试验。默认 K 为 0；默认两个迭代预算、两个重启种子，
因此 K 为 2 时总共运行 **2 × 2 × (1 + 2) = 12 组**。K 大于 0 时也允许只设
一个预算和一个重启种子，因为默认初值与生成初值已构成不同试验。

同一批 K 个初值在所有预算/重启种子条件间复用，以便逐项比较。生成过程使用
`--seed` 派生的独立随机数流，不消耗已知 FK 目标或 IK 重启使用的随机数流。
试验先排列全部默认初值组合，再排列显式初值组合。生成初值数量增加时，原有
向量保持不变；多条件工作流的整体试验顺序仍应以保存的名称和索引为准。

`--candidate-diversity` 可独立于质量选择启用，但必须使用 `--stability-test`。
位置来源、扰动和边界目标继续由 `--stability-targets` 指定。容差选项为：

| 参数 | 默认值 | 使用对象 |
| --- | ---: | --- |
| `--candidate-angular-tolerance-rad` | 0.001 rad，约 0.057° | revolute、continuous |
| `--candidate-linear-tolerance-m` | 0.0001 m，即 0.1 mm | prismatic |

报告保存在 `stability/diversity.json`，并写入总报告与 Markdown；各组关节解和
显式初值仍保存在原始试验目录。Make 和可选依赖 CI 配置已经启用生成初值、
多样性报告和质量选择，并保存完整稳定性目录作为 artifact。

多样性是诊断指标，当前没有按“不同配置数量”设置验收门槛。原有 IK、稳定性
分歧及选后质量验收仍然生效，不能用配置数量替代任务成功或质量通过率。

## Python 接口

下面的解析位置任务有两个已知解，基准解靠近肘关节限位：

```python
import numpy as np
from workspace_analyzer import (
    IKTrial, IKDiversityConfig, ReachabilityConfig,
    create_solver, make_ik_seed_trials, analyze_ik_stability,
)

solver = create_solver("tests/fixtures/two_branch.urdf", backend="numpy",
                       dtype="float64", max_iterations=100)
trials = [
    IKTrial("baseline", seed=[np.pi / 2, -np.pi / 2]),
    *make_ik_seed_trials(solver, 4, random_seed=77),
]
study = analyze_ik_stability(
    solver, [[1, 1, 0], [5, 5, 0]], trials,
    ReachabilityConfig(restarts=1, rescue_restarts=0,
                       minimum_isotropy=0.05, minimum_joint_limit_margin=0.2),
)
diversity = study.summarize_diversity(IKDiversityConfig(
    angular_tolerance_rad=1e-3, linear_tolerance_m=1e-4,
))
print(diversity["summary"])
study.save("outputs/generated-study")
study.select_solutions().save("outputs/generated-study/selected.npz")
```

生成器返回 `IKTrial` 元组，各初值为只读 `(DoF,)` 数组，按求解器精度保存，
在同一试验中广播给全部目标。`random_seed` 在此控制**初值生成**；返回试验的
求解预算和 IK 重启种子未覆盖，仍继承 `ReachabilityConfig` 和求解器配置。

`joint_fraction` 默认为 0.5，在求解器提供的关节区间内均匀抽样。设置为 0.1 时，
抽样范围是区间中点左右各 10% 的全区间长度，即中心 20% 区域。连续关节使用
求解器提供的有限代表区间。生成器不求解 IK、不进行碰撞或质量筛选，也不保证
生成的不同初值最终收敛到不同关节解。

增加 `count` 保留同环境、同种子和同精度下已有初值序列的前缀。原始向量保存在
`initial_seeds.npz`，复现时应优先使用这些已保存数据。需要逐目标不同的初值或
指定局部初值时，仍可直接创建带 `(N, DoF)` 数组的 `IKTrial`。

## 配置分组规则

每个目标单独按原始试验顺序处理，仅对 IK 成功候选分组：

1. 第一个成功配置成为该目标的第一个代表。
2. 后续配置与已有代表比较，**每个关节**的距离都不超过相应容差时，归入最早
   匹配的代表组；否则新增代表。
3. revolute 和 prismatic 比较原始关节坐标差；仅 continuous 使用模 `2π` 的
   最短角距离。因此连续关节的 `π` 和 `-π` 等价，有界多圈转动关节则不自动等价。
4. 某组只要有一个成员通过当前质量门槛，就计为有达标配置的组；不要求代表本身
   达标。质量筛选继续使用全部候选，分组不删除任何测量。

这是依赖试验顺序的代表分组，并非传递聚类。若 A 接近 B、B 接近 C，而 A 不
接近 C，A 先出现时可能分为两组，B 先出现时可能只有一组。它也不是全部 IK 分支
枚举：冗余机器人同一分支内可能有很多不同配置，数值误差和分组容差也影响数量。

## 报告与离线复查

| 字段 | 含义 |
| --- | --- |
| `summary.ik_candidate_count` | 全部目标行的成功试验总次数 |
| `summary.configuration_count` | 按目标分别分组后的配置组数量之和 |
| `summary.duplicate_candidate_count` | 已归入此前代表组的成功候选数量 |
| `summary.quality_configuration_count` | 至少包含一个质量达标候选的组数量 |
| `per_target` | 每行成功次数、配置组数、质量达标组数 |
| `representative_trial_indices` | `(试验数, 目标数)` 的代表试验索引；失败为 `-1` |
| `by_trial` | 每组新增配置、重复候选、新增质量达标组及累计配置数量 |

配置组数按输入目标行独立计数，重复目标行也分别计数，不是跨目标的全局配置
去重。输出为严格 JSON 可序列化字典，不修改原始结果。

```python
from workspace_analyzer import IKStabilityResult

saved = IKStabilityResult.load("outputs/generated-study")
report = saved.summarize_diversity()  # 使用默认关节容差
reassessed = saved.reassess_quality(minimum_joint_limit_margin=0.5)
updated = reassessed.summarize_diversity()
```

离线质量重评只改变达标组数量，不改变同一顺序、同一容差下的几何分组。新生成
试验保存 `joint_kinds`；较早的结果没有该字段时，必须显式传入与关节顺序对应的
`joint_kinds=["revolute", ...]`，不能从角度数值猜测类型。有已保存类型时，显式
参数必须与其一致。

顶层函数 `summarize_ik_diversity(study, config, ...)` 与方法等价，均支持取消。
分组及质量重评不调用 IK/FK/Jacobian；会检查成功残差、配置、质量标记和试验
兼容性，但不能替代对被外部修改的关节解重新做运动学验证。

当前实现按试验对和活跃目标比较，避免创建完整的两两距离张量；最坏计算量随
`试验数² × 目标数 × DoF` 增长，标签和报告随 `试验数 × 目标数` 增长。适合有
明确预算的试验池，不提供流式聚类或自动停止。连续几组没有新配置也不保证后续
没有新配置，本轮 Marvin 记录就出现了这样的反例。见 [本轮验证报告](optimization-2026-09-05-round11.md)。

## 大候选池的比较开销

分组只扫描至少产生过一个代表配置的试验，并在当前试验的成功候选全部匹配后
结束比较。重复试验的测量和质量贡献仍完整保留，首个匹配代表的选择顺序不变。
加速效果取决于重复候选比例；大量不同配置时，最坏情况仍需二次数量级的比较。
本机对照及独立标量验证见[第十九轮记录](optimization-2026-09-08-round19.md)。
