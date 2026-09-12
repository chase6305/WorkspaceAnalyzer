# 固定姿态平移线段的边界细化

`refine_translation_boundary` 在给定线段上细化“找到 IK 解”和“当前预算下未找到解”
之间的区间。只继续计算尚未达到分辨率的线段，并复用成功端点的关节解作为初值。
它输出实际测量的两个端点、残差、停止原因和全部查询记录。

## 运行工作流

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf tests/fixtures/cartesian_stage.urdf --boundary-test \
  --boundary-distance-m 2 --boundary-tolerance-m 0.0001 \
  --boundary-max-rounds 16 --output-dir outputs/boundary-workflow
```

脚本从每个 FK 参考位姿沿**基座 X 正方向**平移，默认距离 2 米，保持参考方向。
初值来自前一步位姿测试的返回配置，参考端点仍重新求解验证。参数单位均为米。
`--boundary-max-rounds` 是二分轮数，不是单次 IK 的迭代上限；后者由
`--max-iterations` 控制。`make check` 和 `make test-workflow` 已包含此解析机构验收。

启用 `--boundary-test` 后，工作流要求每条线段都得到成功/失败端点，并且全部
细化到设定分辨率。起点失败、终点仍成功、预算不足、浮点精度不足或最终复查
恢复成功都会使验收退出码为 1，诊断数据仍保留。若终点成功，可增大扫描距离；
该状态说明当前线段没有提供所需端点，不能据此认定整段都可达。

## Python 接口

API 支持同时处理多个任意方向的平移线段，输入均为求解器基座坐标：

```python
import numpy as np
from workspace_analyzer import (
    BoundaryConfig, ReachabilityConfig, create_solver, refine_translation_boundary,
)

solver = create_solver("tests/fixtures/cartesian_stage.urdf", backend="numpy")
reference = np.eye(4)
boundary = refine_translation_boundary(
    solver,
    reference,                # (4, 4) 或 (N, 4, 4)
    [2.0, 0.0, 0.0],          # (3,) 或与参考数量相同的 (N, 3)
    BoundaryConfig(tolerance_m=1e-4, max_rounds=16),
    reachability=ReachabilityConfig(position_only=False, batch_size=128),
)
print(boundary.to_dict())
boundary.save("outputs/my_boundary")
```

每个终点的旋转与参考完全相同，不对旋转做插值。若显式设置
`ReachabilityConfig(position_only=True)`，仅求解位置约束，报告的
`orientation_constrained` 为 `false`。质量门槛只提供配置诊断，不控制二分方向。

可传入 `seed`、`ResultCache`、`cancel_event` 和 `progress_callback`。
`seed` 接受 `(DoF,)`、`(1, DoF)` 或逐参考 `(N, DoF)`。输入会复制，空任务、零长
线段、非有限数据和无效姿态会被拒绝。`max_rounds=0` 允许只检查端点。

## 算法与停止状态

1. 分别验证全部参考与终点；终点使用对应参考返回的关节配置作初值。
2. 只对参考成功、终点失败的线段进行二分。中点成功时移动成功端，否则移动失败端。
3. 达到实际端点距离要求的线段退出细分；其余线段继续，直至轮数预算耗尽。
4. 若中点在求解器精度下已与任一端点重合，停止并报告浮点分辨率限制。
5. 使用最终成功配置复查失败端点。若恢复成功，原区间不再作为已细化边界发布。

| 状态 | 含义 |
| --- | --- |
| `refined` | 存在实际成功端和复查后仍失败的端点，间距不超过 `tolerance_m` |
| `start_failed` | 参考未找到解，没有可用于此流程的成功起点 |
| `end_succeeded` | 初始参考和终点都找到解，没有成功/失败区间 |
| `budget_exhausted` | 达到最大轮数，保留尚未达到分辨率的区间 |
| `precision_limit` | 中点舍入到已有端点，继续二分无法增加位置分辨率 |
| `end_recovered` | 最后复查使失败端点恢复成功，需要重新设置扫描或求解条件 |

二分不假设整条线段只存在一次可达性变化；它仅保留沿本次搜索路径找到的一组
成功/失败端点。不会自动发现同状态端点之间的空洞，也不能保证找到最靠近起点
或最远的边界。完整平面/三维粗扫描、多区间发现和质量边界尚未实现。

## 结果、成本和复现

`BoundaryResult.measurements` 是包含全部测量的 `AnalysisResult`，模式为
`translation_boundary`。包含位姿、关节解、IK 成功、残差及质量数组；不冒充通用
目标评估的单次汇总。`segment_ids`、`parameters` 和 `stages` 给出每次查询对应的
线段、线段参数及阶段。参数 0、1 对应原始参考和终点。

| 文件 | 内容 |
| --- | --- |
| `measurements.npz` | 可用 `AnalysisResult.load` 加载的全部逐目标测量 |
| `trace.npz` | 查询到线段的映射、端点索引、阶段、轮数和停止状态，无需 pickle |
| `report.json` | 原始求解配置、每批缓存信息、逐线段区间和严格 JSON 摘要 |

工作流将以上文件放在输出目录的 `boundary/` 下，并在总 JSON/Markdown 中展示
边界摘要。`lower` 和 `upper` 表示沿原始线段参数顺序的两端，不表示空间 XYZ
大小顺序；只有 `refined` 状态承诺其中一个成功、另一个复查失败。

每条有效区间最多查询 `max_rounds + 3` 个目标：两个初始端点、每轮至多一个中点、
一次最终复查。未形成区间的线段只查询两个端点。计数包含缓存查询和复查，不是
底层 IK 迭代或重启次数。测量仍整体保存在内存中，未实现断点续算。

相同目标、配置、初值、batch size 和顺序可重放细化过程并复用已有测量缓存。
进度按轮数预算报告，提前结束时直接到 1；取消后不返回完整边界结果，已完成的
测量批次缓存可以保留。设置更小的空间分辨率并不保证几何精度相应提高：数值 IK
容差、初值、预算和分支选择仍影响分类。本功能未包含碰撞检查。
