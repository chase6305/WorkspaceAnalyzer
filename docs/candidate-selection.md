# 已测 IK 候选解的质量选择

`study.select_solutions()` 从稳定性试验已经保存的关节解中，逐目标选择一个配置，
返回可保存、可离线重评的 `AnalysisResult`。筛选不调用 IK、FK 或 Jacobian，
可用于比较初值和求解预算带来的配置质量差异。

## 选择规则

每个目标独立排序，优先级从高到低为：

1. 有候选通过当前全部质量门槛时，仅在这些候选中选择。
2. 没有质量达标候选时，在 IK 成功候选中选择，保留质量拒绝标记。
3. 同一成功等级内，最大化指定的有限质量指标。
4. 指标完全相等时，选择有限残差更小者；残差也相等时，保留原始试验顺序靠前者。
5. 所有候选均 IK 失败时，只按有限非负残差和试验顺序选择诊断记录，仍标记失败。

| `objective` | 含义 |
| --- | --- |
| `joint_limit_margin`，默认 | 最大化归一化关节限位余量 |
| `isotropy` | 最大化所选任务 Jacobian 的各向同性 |
| `minimum_singular_value` | 最大化所选任务 Jacobian 的最小奇异值 |

NaN 或无限的排序指标排在有限指标之后；全部缺失时仍保留已观察到的 IK 成功。
质量门槛优先于排序指标，因此为了满足另一项质量要求，指定指标也可能比首组
下降；报告会保留该目标行号。有限浮点指标直接比较，没有额外的近似并列容差。

对未修改的测量，在相同门槛下，选择后的 IK 成功集合等于各组 IK 成功集合的并集，
质量通过集合也等于各组质量通过集合的并集。该性质只针对所提供的候选；试验中
可能重复出现同一配置，候选数量不能解释为不同 IK 分支数。

## Python 使用与保存

```python
from workspace_analyzer import IKStabilityResult, AnalysisResult

study = IKStabilityResult.load("outputs/study")
selected = study.select_solutions(objective="joint_limit_margin")
selected.save("outputs/study/selected.npz")

print(selected.metadata["assessment"])
print(selected.metrics["selected_trial_index"])  # 对应 study.names 的零基索引
print(selected.metadata["candidate_selection"]["selection_time_summary"])
loaded = AnalysisResult.load("outputs/study/selected.npz")
```

顶层函数 `select_ik_solutions(study, objective=...)` 等价。两种入口均支持
`cancel_event`。生成候选的预算和关节初值接口见 [稳定性测试指南](stability-testing.md)。

输出中各行的关节位置、残差及质量指标来自同一候选，不拼接不同候选的指标。
目标顺序、任务权重、后端和精度保持不变，重新计算整体及加权统计。输出数组
独立于原始试验，修改输出不会改写候选记录。

筛选前检查任务、模型摘要、关节名称、坐标来源、非预算求解设置和门槛的一致性。
成功候选须有有限关节值及不超过原 IK 容差的有限非负残差；保存的质量标记须与
当前测量及门槛一致。
这些检查不能代替对被外部修改的关节解重新执行 FK。

`metadata.candidate_selection` 是**选择时的审计快照**：

| 字段 | 内容 |
| --- | --- |
| `objective`、`priority` | 排序指标与优先级 |
| `selection_time_quality_thresholds` | 选择时使用的完整质量门槛 |
| `selection_time_summary` | 相对首组的 IK/质量改善与退化、指标改善/下降行号、改变来源的行号 |
| `ik_candidate_counts` | 每个目标的成功试验数量 |
| `quality_candidate_counts_at_selection` | 选择时每个目标的质量通过试验数量 |
| `trials` | 候选试验名称、完整求解配置、初始化来源及被选行数 |

`selected_trial_index` 对所有行都有值，包含全部候选失败时保留的诊断记录。
必须同时检查 `reachable` 和 `quality_pass`。顶层 metadata 不继承某一组的迭代
预算、随机种子、重启次数或缓存命中标记；这些配置保留在各组来源中。

单个 `selected.npz` 不重复保存全部候选和原始关节初值。需要复现或重新选择时，
应保留完整 `study.save(...)` 目录，包括逐组 NPZ 和 `initial_seeds.npz`。

## 更换门槛时区分重评与重新选择

```python
# 只检查已经选定的配置；来源索引不变，选择时的审计快照保留。
checked = selected.reassess_quality(minimum_joint_limit_margin=0.5)

# 对全部候选统一替换门槛，然后重新选择配置。
reselected = study.reassess_quality(
    minimum_isotropy=0.03,
    minimum_joint_limit_margin=0.5,
).select_solutions(objective="isotropy")
```

两种操作均离线执行；未指定的门槛关闭。当前质量统计在 `metadata.assessment`
中，不能把 `candidate_selection` 中选择时的数量误读为重评后的数量。

## 工作流与验收

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf tests/fixtures/cartesian_stage.urdf \
  --stability-test --select-ik-solutions joint_limit_margin \
  --output-dir outputs/selection-workflow
```

`--select-ik-solutions` 必须与 `--stability-test` 一起使用，可选择位姿、配对扰动
或边界端点三种稳定性目标集。工作流将原始试验保存在 `stability/`，选定配置保存
为 `stability/selected.npz`；总报告记录该文件、来源映射、配置与选择时的统计。

可用 `--min-selected-quality-rate` 指定选后质量通过率下限。分母是**整个稳定性
目标集**，包含范围外目标、稳定失败目标和重复行；它不同于只检查已知 FK 目标的
`--min-quality-rate`。例如解析机构默认有 16 个已知目标及 6 个范围外目标，即使
全部已知目标达标，该比例也只有 `16/22`。

超限时返回 1，并保留测量与报告。选择不会覆盖原有稳定性分歧、已知目标、扰动
或边界验收失败。`make check`、`make test-workflow` 和可选依赖 CI 配置已启用
候选选择流程。

## 验证边界

解析二连杆例子提供同一位置的两个已知解，一个靠近肘关节限位，另一个通过质量
门槛；选择结果还由独立三角公式检查位置。伸直奇异配置与范围外目标提供质量和
IK 失败反例。NumPy/Torch CPU、float32/float64 都覆盖三种排序指标。

Marvin 已测端点说明指标之间存在取舍：提升各向同性可能降低关节余量，且未必
足以通过原门槛。结果见 [第十轮验证报告](optimization-2026-09-05-round10.md)。

当前功能只选择已测候选，不保留求解器内部全部重启候选、不自动生成新解、不证明
全局最优，也不检查碰撞。逐目标独立选择可能切换分支，不能直接当作连续可执行
轨迹。候选数据随试验数和目标数增长；当前流程不提供流式存储。

需要增加候选时，可用 `make_ik_seed_trials` 生成固定预算的显式初值试验，并用
`study.summarize_diversity()` 区分重复候选和不同配置。分组不删除候选，因此接近
质量门槛两侧的配置仍参与选择。见 [初值与多样性指南](candidate-diversity.md)。
