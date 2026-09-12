# IK 初值与求解预算稳定性测试

`analyze_ik_stability` 对同一批有序目标运行多组 IK 条件，记录成功/失败交集、
分歧目标、相对首组的改善与退化，以及残差和成功配置质量的变化。它不合并各组
的最佳解，也不把多次结果一致解释为几何可达性证明。需要选择配置时，可显式调用
`study.select_solutions()`，按质量门槛及指定指标筛选已测候选，保留原始试验。
见 [候选解选择指南](candidate-selection.md)。

## 工作流用法

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf tests/fixtures/cartesian_stage.urdf --stability-test \
  --stability-seeds 77 78 --stability-iterations 30 60 \
  --max-ik-disagreement 0 --output-dir outputs/stability-workflow
```

默认采用当前 `--seed` 及其加一、当前 `--max-iterations` 及其两倍，交叉组成
4 组试验。用户列表按首次出现顺序去重；未增加显式初值时，至少需要 2 个不同的
预算/种子组合。默认各组使用求解器默认关节初值；不同随机种子影响重启和补救候选。改变种子
不一定改变最终解，特别是在未使用额外随机候选时。

可通过 `--stability-initial-guesses K` 为各预算/种子组合增加相同的一批显式关节
初值，再用 `--candidate-diversity` 报告新增配置和重复候选。默认 K 为 0。
数量计算、关节容差及 Python 生成接口见 [候选多样性指南](candidate-diversity.md)。

`--stability-targets` 控制输入：

| 值 | 测试目标 |
| --- | --- |
| `pose`，默认 | 已知 FK 位姿和故意构造的范围外位姿 |
| `perturbations` | 已启用 `--robustness-test` 的全部配对扰动目标 |
| `boundary` | 已启用 `--boundary-test` 的最终两端，按每条线段 lower、upper 顺序排列 |

边界复查示例：

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf tests/fixtures/cartesian_stage.urdf --boundary-test \
  --stability-test --stability-targets boundary \
  --stability-seeds 77 78 --stability-iterations 30 60 \
  --output-dir outputs/boundary-stability
```

默认只报告分歧。若设置 `--max-ik-disagreement`，分歧目标占整个选定目标集的
比例必须不超过该值，否则返回 1，并保留报告与逐组测量。某个目标只要在至少
一组成功、至少一组失败，就计为分歧；重复目标按输入行计数。稳定失败的目标
不属于分歧，因此低分歧率不等于高成功率。质量分歧单独报告，不受该参数验收。
工作流原有的已知目标、扰动或边界验收继续生效。

`make check`、`make test-workflow` 及可选依赖 CI 配置已包含解析机构的稳定性流程。

## Python 接口

Python 还支持为每组指定关节初值和重启/补救预算：

```python
from workspace_analyzer import (
    IKTrial, ReachabilityConfig, ResultCache, analyze_ik_stability,
)

study = analyze_ik_stability(
    solver,
    targets,
    [
        IKTrial("baseline", max_iterations=150, random_seed=77),
        IKTrial("more iterations", max_iterations=400, random_seed=77),
        IKTrial("other restart seed", max_iterations=150, random_seed=78),
        IKTrial("nearby initial configuration", max_iterations=150,
                random_seed=77, seed=nearby_joint_positions),
    ],
    ReachabilityConfig(position_only=False, restarts=16,
                       rescue_restarts=32, rescue_rounds=2,
                       minimum_isotropy=0.05, minimum_joint_limit_margin=0.1),
    cache=ResultCache("outputs/stability-cache"),
)
print(study.to_dict()["summary"])
study.save("outputs/study")
```

目标支持 `(N, 3)` 位置和 `(N, 4, 4)` 位姿，以及相应单个输入。完整位姿约束需
显式设置 `position_only=False`。`weights` 和 `base_from_targets` 与目标评估
接口一致，在全部试验间保持相同。可传入取消事件和进度回调。

`IKTrial` 未指定的预算项继承共同配置。`seed=None` 使用求解器默认初值，不从
其他试验继承解；显式初值支持 `(DoF,)`、`(1, DoF)` 和 `(N, DoF)`，创建只读
副本。随机种子 `random_seed` 与关节初值 `seed` 是不同输入。

各组从已加载模型构造求解器，保持相同链、后端、设备、精度、容差、阻尼和步长，
只覆盖声明的预算/初值。原求解器不被修改，也不重新读取 URDF。目标、权重、
坐标变换和共同求解配置在开始时快照，回调修改原始输入不会改变后续试验。

## 报告与离线重评

大规模试验的指标极值与有限计数按试验逐组累积，不再堆叠全部浮点测量。
报告生成可调用 `study.to_dict(cancel_event=event)`，其中 `event` 为
`threading.Event`；取消时抛出 `AnalysisCancelled`，不会返回部分报告。
候选选择与多样性分析的兼容性检查也会逐组响应已有的取消事件。
这不是磁盘流式存储：原始试验、布尔状态矩阵和最终逐目标报告仍保存在内存中。
性能与内存对照见[第十四轮记录](optimization-2026-09-08-round14.md)。

| 内容 | 解释 |
| --- | --- |
| `summary.*_all_success_count` | 所有试验均通过的目标数 |
| `summary.*_no_success_count` | 所有试验均未通过的目标数 |
| `summary.*_disagreement_count/rate` | 有成功也有失败的目标数及占比，分母为全部目标 |
| `baseline_comparisons` | 每组与首组比较的改善/退化数量及原始行号，分别记录 IK 和质量 |
| `per_target.*_success_counts` | 每个目标在多少组中通过 |
| `per_target.*_disagreement_indices` | 发生分歧的原始目标行号 |
| `per_target.residual_minimum/maximum` | 全部组中有限残差的范围，包含失败结果 |
| `per_target.isotropy_* / joint_limit_margin_*` | 仅成功配置的有限质量指标范围，不能当作全局最优解范围 |
| `trials` | 每组完整求解设置、质量统计、缓存标记和初始化来源 |

没有有限测量时指标输出 `null`。比较按等权目标计数；每组目标评估的加权统计仍
保存在其 metadata 中。位姿残差混合位置与旋转误差，不能直接把其数值解释为米。

保存目录包含 `report.json`、逐组 `trial_000.npz` 等结果，以及显式初值文件
`initial_seeds.npz`；没有显式初值时该文件为空 NPZ。结果文件采用序号命名，试验
名称只作为显示标签。工作流将目录放在 `stability/`，并在总报告中保存输入来源
文件和原始行号映射；总报告中试验文件路径相对于该子目录。

```python
from workspace_analyzer import IKStabilityResult

saved = IKStabilityResult.load("outputs/study")
updated = saved.reassess_quality(
    minimum_isotropy=0.03,
    minimum_joint_limit_margin=0.2,
)
print(updated.to_dict()["per_target"]["quality_disagreement_indices"])
```

加载时核对目标顺序、任务、模型摘要、非预算求解设置和质量门槛，并从测量数组
重建稳定性摘要。离线重评统一完整替换所有组的质量要求：未指定门槛关闭，原始
结果不变，测量数组共享；不调用 IK/FK/Jacobian。

相同配置可复用测量缓存，修改显示名称或质量门槛不会重跑 IK。更改预算、随机
种子或关节初值会改变对应缓存条目。试验串行执行，数据随目标数和试验数增长；
未提供流式保存、自动停止或生产成功概率估计。

## 试验初值预检查

所有试验初值都会在首次求解、缓存访问和进度回调前完成形状与求解器精度下的
有限性检查，避免执行到后续试验时才发现初值转换溢出。预检查保留原始只读
初值及其精度，用于保存和来源追溯；实际计算使用求解器精度。
最终兼容性检查也会响应传入的取消事件。见[第二十轮记录](optimization-2026-09-08-round20.md)。

## 研究目录加载校验

加载先检查报告版本、试验条目、唯一名称和初始化类型，再读取数组。
初值不仅要与报告中记录的形状一致，也必须符合实际目标数和机器人自由度。
损坏初值压缩包、缺少初值成员或误传 NPY 会得到明确的 `ValueError`；缺失文件
继续保留文件系统异常。公共 `initial_seeds` 修改后，报告、保存和候选分析也会
重新检查其有限性及形状。见[第二十二轮记录](optimization-2026-09-08-round22.md)。
