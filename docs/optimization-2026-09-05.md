# 整体优化验证记录（2026-09-05）

本轮在任务开始时的工作区基础上修改，保留此前尚未提交的功能扩展。
对照版本是包含这些扩展的工作区快照，保存在本机
`/tmp/workspace-analyzer-optimization-baseline`，不是仓库原有 HEAD。

## 主要变更

- FK 仅在需要全部链接时保存中间变换；轴的反对称矩阵不再按样本重复构造。
- IK 每轮共用一次 FK/Jacobian 遍历，末次更新后重新计算残差再选择最优重启结果。
- 旋转误差通过四元数的最大分量计算，避免小角度数值抵消和混合符号轴的半周歧义。
- NumPy 保持配置的计算精度；单目标批次保留批次维度。
- 工作空间输出预分配、逐批写入；关节限位指标逐批计算；读取 NPZ 不再复制刚解压的数组。
- 均匀网格只生成请求数量的前缀，避免先构造指数规模的完整网格。
- 缓存根据实际加载的模型计算标识，并升级缓存模式；末批取消不会发布或缓存结果。
- 完善参数、URDF 限位和结果格式校验，修复安装后命令行成功计算却返回失败状态的问题。
- 基准增加预热、多次运行中位数、运行环境记录和精确的 IK 目标数量。

兼容性调整：`inverse()` 对 `(1, 4, 4)` 输入现在返回 `(1, DoF)`，其诊断数组
形状为 `(1,)`；需要标量诊断时传入 `(4, 4)`。旧缓存会重新计算，已保存的版本 1
NPZ 文件仍可读取。

## 性能对照

环境：AMD Ryzen 9 9950X，Linux 6.8.0-106，Python 3.10.0，NumPy 1.26.3。
模型为 Marvin M6 左臂七轴链，NumPy CPU，float64。两版本使用同一份更新后的
基准脚本、相同随机目标和参数，顺序运行；预热一次、计时七次并取中位数。

| 测量 | 优化前 | 优化后 | 耗时减少 |
| --- | ---: | ---: | ---: |
| FK，4096 个姿态 | 3.245 ms | 3.173 ms | 2.2% |
| Jacobian，4096 个配置 | 3.829 ms | 3.467 ms | 9.4% |
| 全姿态 IK，128 个目标、4 个起点 | 251.143 ms | 181.604 ms | 27.7% |

两次 IK 成功率均为 96.09375%（123/128），未开启失败目标补救。FK 的小幅差异可能受
运行波动影响；这些数据仅表示此次本机测量，不能直接外推至其他硬件或 CUDA。

```bash
PYTHONPATH=/tmp/workspace-analyzer-optimization-baseline/src \
  python examples/benchmark_marvin.py --backend numpy --batch-size 4096 \
  --ik-targets 128 --restarts 4 --repeats 7

PYTHONPATH=src python examples/benchmark_marvin.py --backend numpy \
  --batch-size 4096 --ik-targets 128 --restarts 4 --repeats 7
```

5 万个关节样本、批次 4096、相同模型和精度下，使用 `tracemalloc` 测量分析调用
期间 Python/NumPy 跟踪的分配峰值，不包含进程完整 RSS 或 GPU 显存：

| 分析 | 优化前 | 优化后 | 峰值减少 |
| --- | ---: | ---: | ---: |
| 仅 FK 工作空间 | 15.47 MiB | 8.08 MiB | 47.8% |
| FK 与灵活度指标 | 21.28 MiB | 13.68 MiB | 35.7% |

测量脚本位于本机 `/tmp/workspace-analyzer-memory-check.py`。复测方式：

```bash
PYTHONPATH=/tmp/workspace-analyzer-optimization-baseline/src \
  python /tmp/workspace-analyzer-memory-check.py
PYTHONPATH=src python /tmp/workspace-analyzer-memory-check.py
```

## 验证结果

- 完整测试：**161 passed**，包含 NumPy、Torch CPU、Marvin 资产及 Viser 运行测试。
  Viser 测试需要允许监听本地端口。
- Ruff 检查、格式检查、`git diff --check` 均通过。
- 成功构建 sdist 和 wheel；从 wheel 安装后的命令行可保存单样本结果并以状态 0 退出。
- Marvin float32 全姿态测试：32 个随机配置以自身为种子，NumPy 和 Torch CPU
  均 32/32 收敛，残差为 0。修复旋转公式前，同一检查分别仅 4/32、31/32 收敛。
- 10 秒 NumPy float64 随机回归覆盖 139264 个 FK 姿态、2176 个全姿态 IK 目标；
  IK 成功 2175 个，成功率 99.954%。最大成功残差 `9.98e-6`，最大旋转正交误差
  `1.33e-15`，最大位置 Jacobian 有限差分误差 `6.56e-10`。仍存在一个数值 IK
  未收敛目标，应继续检查结果中的 `success` 和 `residual`。

```bash
PYTHONPATH=src pytest -q
ruff check src tests examples
ruff format --check src tests examples
git diff --check
PYTHONPATH=/tmp/workspace-analyzer-build python -m build --no-isolation \
  --outdir /tmp/workspace-analyzer-optimized-dist
PYTHONPATH=src python examples/stress_marvin.py --duration 10 --backend numpy \
  --report /tmp/workspace-analyzer-stress.json
```

打包命令使用本机已有的 `/tmp/workspace-analyzer-build` 构建依赖；常规开发环境安装
`.[dev]` 后可直接运行 `python -m build`。此次没有在 CUDA 上验证性能。
