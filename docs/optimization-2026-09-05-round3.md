# 第三轮持续优化验证记录（2026-09-05）

本轮以上一轮完成后的工作区为基线，快照保存在本机
`/tmp/workspace-analyzer-round3-baseline`，原有未提交改动保持保留。

## 修复与优化

- 失败补救改为迭代执行，避免递归栈和中间计算数组逐轮累积。原实现设置
  `rescue_rounds=1100` 会触发 `RecursionError`，现已可正常完成。
- `IKResult.iterations` 累计首次求解与所有实际补救轮次的迭代循环次数，不乘以
  并行种子数量。Torch 补救结果合并保留自动微分能力。
- 轨迹在多起点补救成功后复用结果，避免关节跳变检查再次运行完全相同的多起点求解。
- `solve_trajectory(..., batch_size=1024)` 限制闭环重投影和指标计算的批次大小。
  失败帧跳过灵活度计算，并保持 `NaN` 诊断；指标批次之间检查取消事件。
- 闭环重投影中任一批次失败时，返回整条原轨迹，避免混用新旧解。
- 条件数使用相对奇异值阈值，修复统一缩小任务权重后被误判为无穷大的问题。
  无效权重在计算 Jacobian 前即被拒绝。
- URDF 关节轴先缩放再归一化，有限的 `1e-300`、`1e300` 量级轴不会再发生范数
  下溢或上溢。数值测试检查归一化方向、旋转正交性及实际旋转角。

## 专项内存测量

环境：AMD Ryzen 9 9950X，Linux 6.8.0-106，Python 3.10.0，NumPy 1.26.3，CPU、float64。
使用 `tracemalloc` 测量运算期间跟踪的 Python/NumPy 分配峰值，输入数组提前创建，
不计入此峰值；数值不表示整个进程的 RSS 或 GPU 显存。

| 用例 | 优化前 | 优化后 | 峰值减少 |
| --- | ---: | ---: | ---: |
| 20000 帧闭环重投影 | 37.11 MiB | 3.21 MiB | 91.4% |
| 64 个目标、50 轮失败补救 | 12.44 MiB | 0.76 MiB | 93.9% |

重投影使用 Marvin M6 左臂七轴链，20000 帧均为关节限位中心的相同姿态，用于隔离
重投影阶段的批次分配。优化前一次处理全部帧，优化后默认每批至多 1024 帧；
两版本均全部成功、残差为 0。

补救测试使用仓库中的 `two_link.urdf`，64 个目标均为不可达的 `[10, 0, 0]`，
首次 4 个起点、每轮补救 8 个起点、每次最多迭代 2 次、补救 50 轮。
两版本均判定全部不可达，平均残差均为 8；旧版仅报告 2 次迭代，新版正确报告
`2 × (1 + 50) = 102` 次。

预热一次、三次计时中位数：重投影为 28.91 → 28.39 ms，补救为 53.59 → 44.59 ms。
这些用例侧重内存规模与极端重试，不代表所有轨迹或 IK 工作负载。

```bash
PYTHONPATH=/tmp/workspace-analyzer-round3-baseline/src \
  python /tmp/workspace-analyzer-round3-memory.py
PYTHONPATH=src python /tmp/workspace-analyzer-round3-memory.py
```

测量脚本保存在本机 `/tmp/workspace-analyzer-round3-memory.py`。
完整轨迹仍需保存与帧数成正比的输入和结果数组，`batch_size` 限制的是中间计算规模。
启用 Torch 自动微分时，求导所需计算图仍会保留。

## 普通 IK 一致性与计时

Marvin 左臂，128 个全姿态目标，首次 4 个起点，每轮补救 8 个起点，最多补救 3 轮，
随机种子 12。两版本在同一 Python 进程交替运行，每版预热一次、计时十次。

| 项目 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 成功率 | 128/128 | 128/128 |
| FK/Jacobian 批次求值数 | 300 | 300 |
| 起点配置求值总数 | 42604 | 42604 |
| 报告的迭代次数 | 150 | 300 |
| 耗时中位数 | 209.22 ms | 219.55 ms |
| 耗时范围 | 155.16–300.82 ms | 166.74–295.59 ms |

关节解和残差逐元素一致，最大差异均为 0。中位耗时约增加 4.9%，两组计时波动范围
较大并有重叠；本轮不据此宣称普通 IK 提速。确认的主要收益是峰值分配和极端重试稳定性。

```bash
PYTHONPATH=src python /tmp/workspace-analyzer-round3-compare.py
```

该脚本加载当前代码与基线快照，在同一进程中交替测量；其统计保存在本机
`/tmp/workspace-analyzer-round3-compare.json`。

## 验证结果

- 完整测试：**212 passed**，包含 NumPy、Torch CPU、Marvin 资产和 Viser 运行测试。
- 新增覆盖：1100 轮补救、确定性最优结果、累计迭代计数、Torch 补救梯度、重复修复
  消除、单帧与尾批重投影、部分重投影失败回退、失败帧跳过指标、指标阶段取消、
  条件数缩放不变性及极端关节轴。
- Ruff、格式检查、`git diff --check` 均通过。
- sdist 与 wheel 构建通过。CUDA 未实测。

```bash
PYTHONPATH=src pytest -q
ruff check src tests examples
ruff format --check src tests examples
git diff --check
PYTHONPATH=/tmp/workspace-analyzer-build python -m build --no-isolation \
  --outdir /tmp/workspace-analyzer-round3-dist
```

Viser 测试需要允许本地端口监听。打包使用本机已有的
`/tmp/workspace-analyzer-build` 构建依赖。
