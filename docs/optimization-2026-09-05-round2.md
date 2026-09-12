# 第二轮持续优化验证记录（2026-09-05）

本轮以上一轮优化完成后的工作区为基线，并保留原有未提交改动。基线快照保存在
本机 `/tmp/workspace-analyzer-round2-baseline`。

## 变更

- 已收敛的 IK 起点退出后续 FK/Jacobian/DLS 计算批次，其他起点继续求解；最终仍
  根据残差或与种子的距离选择结果。Torch 更新保留自动微分所需的中间值。
- `continuous` 关节按周期归一化，避免在 ±π 停住；关节居中不作用于连续关节，
  姿态偏置使用最短角度差。轨迹仅对成功帧展开角度，避免失败配置引入虚假的圈数。
- 闭环重投影改为批量 IK，轨迹输出预分配；首尾目标近似重合时，保留各自的关节解
  与实际残差，只有目标完全相同时才复用首帧解。
- 关节分析增加进度和取消支持；IK、失败补救、轨迹跟踪及闭环重投影都传递
  `cancel_event`。单次底层运算无法中断，取消在批次或迭代边界生效。
- 查看器启动后台任务时加锁，关闭时取消并等待工作线程；禁止关闭后的迟到结果
  写入场景、重复启动重计算，以及关闭后重新播放。
- 缓存模式升级为 3，避免复用旧的连续关节分析结果；NPZ 结果文件格式仍为版本 1。

## 性能对照

环境：AMD Ryzen 9 9950X，Linux 6.8.0-106，Python 3.10.0，NumPy 1.26.3，
Torch 2.13.0+cu130。所有测试均在 CPU 上执行，未验证 CUDA。

模型为 Marvin M6 左臂七轴链，float64，随机种子 12。全姿态 IK 使用 128 个目标、
4 个起点、最多 150 次迭代，不启用失败补救。两版本顺序运行相同脚本；预热一次，
NumPy 取七次计时中位数，Torch CPU 取五次中位数。

| 测量 | 本轮优化前 | 本轮优化后 | 耗时减少 |
| --- | ---: | ---: | ---: |
| NumPy 全姿态 IK | 200.681 ms | 108.479 ms | 45.9% |
| Torch CPU 全姿态 IK | 190.482 ms | 161.028 ms | 15.5% |
| NumPy 101 帧闭环轨迹 | 165.522 ms | 144.313 ms | 12.8% |

两个后端在两版本中的 IK 成功率均为 123/128（96.09375%），残差 p95 均约
`9.5485e-6`。已收敛起点退出计算没有改变本次测试中的成功率和残差。

闭环轨迹预热一次、计时五次，逐帧使用前一成功解作为种子，完成后批量重投影。
两版本均 101/101 收敛，最大残差 `9.7852e-6`，最大相邻关节变化 `0.00674547`，
归一化闭合误差为 0。本轮数据是当前机器上的测量，不保证其他模型、批次和硬件
具有相同提升幅度。

```bash
PYTHONPATH=/tmp/workspace-analyzer-round2-baseline/src \
  python examples/benchmark_marvin.py --backend numpy --batch-size 4096 \
  --ik-targets 128 --restarts 4 --repeats 7
PYTHONPATH=src python examples/benchmark_marvin.py --backend numpy \
  --batch-size 4096 --ik-targets 128 --restarts 4 --repeats 7

# Torch 对照使用相同命令，将 --backend 改为 torch，--repeats 改为 5，
# 并指定 --device cpu。
```

轨迹脚本保存在本机 `/tmp/workspace-analyzer-trajectory-benchmark.py`，两版本通过
切换 `PYTHONPATH` 运行。其目标构造与求解参数如下：

```python
phase = np.linspace(0, 2 * np.pi, 101)[:, None]
q = (
    solver.joint_limits.mean(axis=1)
    + np.sin(phase) * np.linspace(0.05, 0.1, solver.dof)
    + (np.cos(phase) - 1) * np.linspace(0.02, 0.04, solver.dof)
)
targets = solver.forward(q)
targets[-1] = targets[0]
result = solver.solve_trajectory(targets, seed=q[0], failure_restarts=4, dt=0.02)
```

## 验证

- 完整回归：**187 passed**，包含连续关节、批次缩减、Torch 自动微分、取消、
  近似闭环残差和 Viser 并发关闭测试。
- 连续关节从 3.0 到 3.4 rad 的九帧路径，不使用重启时，原实现仅前三帧收敛；
  修复后 NumPy 和 Torch CPU 均九帧收敛。反方向跨界、多圈种子和整圈轨迹也有测试。
- 10 秒 Marvin NumPy float64 随机回归覆盖 169984 个 FK 姿态、2656 个 IK 目标，
  2655 个成功（99.9623%）；最大成功残差 `9.98e-6`，最大旋转正交误差 `1.33e-15`，
  最大位置 Jacobian 有限差分误差 `6.59e-10`。仍有一个数值 IK 未收敛目标。
- Ruff 检查、格式检查、`git diff --check` 通过；sdist 与 wheel 构建通过。

```bash
PYTHONPATH=src pytest -q
ruff check src tests examples
ruff format --check src tests examples
git diff --check
PYTHONPATH=/tmp/workspace-analyzer-build python -m build --no-isolation \
  --outdir /tmp/workspace-analyzer-round2-dist
PYTHONPATH=src python examples/stress_marvin.py --duration 10 --backend numpy \
  --report /tmp/workspace-analyzer-round2-stress.json
```

Viser 集成测试需要允许监听本地端口。打包使用本机已有的
`/tmp/workspace-analyzer-build` 构建依赖。
