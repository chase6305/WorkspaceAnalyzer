# 配对位姿扰动测试

本测试回答：同一个任务在指定位置或方向偏差下，数值 IK 是否仍找到解，已选
配置的质量是否仍达标。每个参考位姿对应一组固定变体，报告保留逐目标配对关系。

## 运行

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf tests/fixtures/cartesian_stage.urdf \
  --robustness-test --perturb-translation-m 0.01 --perturb-rotation-deg 5 \
  --output-dir outputs/robustness
```

替换 URDF 并指定必要的 `--base-link`、`--tip-link` 可用于实际机器人。
`make test-workflow` 和 `make check` 已包含解析机构的扰动流程。

默认每个参考位姿生成 13 个目标：参考本身、基座 XYZ 各 ±1 cm 平移、工具 XYZ
各 ±5° 旋转。各变体只改变一个方向，平移与旋转不叠加。幅度设为 0 会关闭该类
变体，不生成六份重复目标；两类均为 0 时仅测试参考，扰动率返回 `null`。

解析平移机构可以完成其范围内的平移，无法完成非零旋转。默认流程报告这种能力
限制；扰动后的成功率本身不作为默认失败条件。参考位姿仍须满足
`--min-known-success`。若需要验收所有扰动均成功的任务占比，增加：

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf tests/fixtures/cartesian_stage.urdf --robustness-test \
  --perturb-rotation-deg 0 --min-robust-rate 1 \
  --output-dir outputs/translation-acceptance
```

`--min-robust-rate` 的分母是全部参考任务；一个任务必须参考及全部变体均 IK 成功
才计入分子。它要求显式启用扰动测试且至少一类幅度非零。质量鲁棒性同时报告，
该参数只验收 IK。失败时退出码为 1，仍保留全部报告和逐目标测量。

## Python API 与离线分析

```python
import json
import numpy as np
from workspace_analyzer import PosePerturbations, ReachabilityConfig, WorkspaceAnalyzer

# reference_poses 是求解器基座坐标下的 (N, 4, 4) 位姿。
study = PosePerturbations(
    reference_poses,
    translation_m=0.01,
    rotation_rad=np.deg2rad(5),
    translation_frame="base",
    rotation_frame="tool",
)
result = WorkspaceAnalyzer(solver).analyze_targets(
    study.targets,
    ReachabilityConfig(position_only=False, minimum_joint_limit_margin=0.1),
)
report = study.summarize(result)
print(json.dumps(report, allow_nan=False))

# 不重新运行 IK/FK/Jacobian；未指定的质量门槛关闭。
stricter = study.summarize(result.reassess_quality(minimum_joint_limit_margin=0.2))
```

`PosePerturbations` 对输入创建只读快照，目标按“参考 0 的所有变体、参考 1 的所有
变体……”排列。`variant_names[0]` 恒为 `reference`，其后按平移 XYZ 正/负、旋转
XYZ 正/负排列。float32 输入保持 float32，其余有效数值输入转为 float64。

位移轴可选择 `base` 或 `tool`，后者使用参考工具轴。旋转同样可选：`base` 为
`R_offset @ R_reference`，`tool` 为 `R_reference @ R_offset`；两者都只旋转 TCP
方向，不使 TCP 位置绕基座原点公转。汇总仅接受完整位姿评估，并核对基座目标、
顺序和位置数组，防止错配其他测量。若任务来自其他坐标系，先把参考转换到基座
再生成本测试集。

## 如何读报告

`report.json` 的 `perturbation_robustness` 包含：

| 字段 | 解释 |
| --- | --- |
| `summary.references` | 参考任务数量 N |
| `summary.perturbed_samples` | 不含参考的变体总数 N × (K − 1) |
| `ik / quality.reference_success_count` | 参考本身的 IK / 质量通过数 |
| `perturbed_success_rate` | 不含参考的全部变体通过率 |
| `all_variants_pass_rate` | 参考及全部变体均通过的任务占全部参考的比例 |
| `retained_success_rate` | 仅在参考通过的任务中统计变体通过率；参考全部失败时为 `null` |
| `lost_pairs / gained_pairs` | 相比各自参考，由通过变失败 / 由失败变通过的变体数 |
| `by_variant` | 按偏移方向统计通过数和配对变化，用于定位敏感方向 |
| `per_reference` | 每个参考的通过状态、变体通过率和成功配置中的最差指标 |

所有率按等权样本计数，不使用评估结果的任务权重。`worst_solved_*` 仅对本组
IK 成功且指标有限的配置取最小值；没有成功测量时为 `null`，不能把它单独用作
整个扰动组通过的证据。最差质量可能包含参考配置。各向同性和关节余量等质量
指标对应 `ReachabilityConfig.dexterity_task`；默认仍为位置 Jacobian，即使 IK
约束完整位姿。

新增文件 `perturbation_targets.npz` 保存参考位姿、所有变体和顺序名称；
`perturbations.npz` 保存逐目标结果，配置及统计保存在总报告中。可用保存的参考
和报告幅度重建 `PosePerturbations`，再加载结果做离线重评。JSON/Markdown 报告
显示当次启用的用例；比较不同实验建议使用不同输出目录。

## 验证范围

当前已验证解析立方体边界内外变化、没有旋转自由度的反例、零幅度退化、工具坐标
扰动在刚体换基下的一致性，以及 NumPy/Torch CPU、float32/float64 和缓存重评。
目标最多整体展开为 13N 个位姿，IK 仍分批；这不是流式目标生成。

单轴有限变体不覆盖组合偏差、整个误差球或真实误差概率分布。报告也不包含碰撞
检查。参考与变体独立求解，批次和重启候选可能不同；质量变化也可能包含返回
分支变化的影响。变体失败可能受数值 IK 初值和预算影响，不足以证明目标在数学上不可达。
Newton–Raphson IK 对初值和局部 Jacobian 的依赖可参考
[Modern Robotics 数值 IK 教程](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/)。
后续建议和验收方法见 [可引入的验证方案](validation-options.md)。

## 取消离线报告

可传入 `perturbations.summarize(result, cancel_event=event)`，其中 `event` 为
`threading.Event`。取消时抛出 `AnalysisCancelled`，不返回部分报告，也不修改
源测量。检查点位于输入读取前、各变体及指标聚合阶段和返回前；单次 NumPy
运算不会被强制打断。见[第二十四轮记录](optimization-2026-09-08-round24.md)。

## 报告性能基准

可运行 `PYTHONPATH=src python examples/benchmark_perturbations.py`，对重复解析
测量的报告生成计时，并输出峰值跟踪分配与完整报告摘要。输入构建及 IK 不计入
时间，内存跟踪不等于进程 RSS。优化对照见[第二十五轮记录](optimization-2026-09-08-round25.md)。
