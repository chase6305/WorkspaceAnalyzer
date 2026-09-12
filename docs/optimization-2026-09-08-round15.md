# 第十五轮：结果保存可靠性与缓存 I/O

日期：2026-09-08。

## 改动

`AnalysisResult` 的数组允许调用者编辑。原先保存不重新检查这些数组，编辑成
对象数组或错误形状后可能覆盖原文件，随后无法通过安全加载接口读回。
保存现在先验证浅层快照，保留调用者对象，并在成功写完临时文件后原子替换。
非法指标名、错误元数据、无效可达标记也不能覆盖原有结果。

缓存写入使用独立的顶层元数据快照。只有保存成功后才更新调用者的 `cache_key`，
写入失败时旧缓存文件与调用者原有标识均保留，临时文件清理。
这不提供对其他线程同时修改测量数组的同步保证。

新增可选无压缩存储，默认行为继续压缩：

```python
from workspace_analyzer import ResultCache

result.save("workspace.npz", compressed=False)
cache = ResultCache("cache", compressed=False)
```

CLI 对应 `--cache-dir cache --cache-uncompressed`，缺少缓存目录时明确报错。
压缩策略只影响新写入，不进入计算缓存键；压缩与无压缩 NPZ 可相互读取，
仍禁用 pickle。存储策略本身不影响分析数组或质量报告。

## 本机合成数据对照

5 万行 float64 合成数据，包含 XYZ、7 关节、可达标记、残差及质量指标。
各运行 3 次取中位数，包含原子保存及校验加载，不含构建输入与结果比较。

| 指标 | 压缩 NPZ | 无压缩 NPZ |
| --- | ---: | ---: |
| 文件字节数 | 5,360,702 | 5,652,552 |
| 写入中位耗时 | 117.06 ms | 4.05 ms |
| 读取中位耗时 | 15.29 ms | 2.40 ms |
| 数组与元数据往返一致 | 是 | 是 |

此测例文件增大约 5.4%，写入约快 29 倍，读取约快 6.4 倍。
数据以随机浮点数为主，不能代表含大量重复值的机器人结果的压缩率；本地重复
读取可能受操作系统页缓存影响。默认保留压缩，空间与速度由调用者选择。
原始记录在 `outputs/round15-storage/benchmark.json`。

```bash
PYTHONPATH=src python examples/benchmark_result_io.py \
  --samples 50000 --repeats 3 \
  --output outputs/round15-storage/benchmark.json
```

## 验证

- 全量 pytest：594 passed、4 skipped，新增 22 项回归。
- 两种 ZIP 编码、精度与指标往返一致；跨压缩策略的缓存命中与 CLI 重放通过。
- 非法数组修改不会覆盖原文件；模拟部分写入失败保留旧文件并清理临时文件。
- 缓存写入失败不改变调用者标识；歧义压缩参数与缺少缓存目录得到明确错误。
- Ruff、50 个 Python 文件格式、完整解析工作流、sdist/wheel 构建及
  `git diff --check` 通过。

本轮没有运行 CUDA 或远程 CI；没有提交或推送。
