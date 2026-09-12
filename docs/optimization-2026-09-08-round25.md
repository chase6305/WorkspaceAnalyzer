# 第二十五轮：扰动报告聚合性能

日期：2026-09-08。

## 改动

最差成功指标改用带掩码的最小值归约，避免先创建完整浮点替换数组。
输出中的缺失值批量转为 None，替代逐标量有限性检查；整数记录在需要时先
转浮点，保持原有归约与 JSON 行为。

配对统计只计算一次每参考点的扰动成功数，再推导保留率、退化数、改善数及
全部变体通过数。避免复制成功参考点对应的整个扰动矩阵，以及重复生成
“基准成功且扰动失败”等全尺寸布尔数组。

新增 `examples/benchmark_perturbations.py` 用于复现完整报告的计时、跟踪分配
及 SHA-256 对照，计时与内存跟踪分开进行。

## 本机测量

2 万参考位姿，每个 13 种变体，共 26 万条重复解析机构测量；运行 5 次取中位数。
输入构建及 IK 不计入时间。

| 扰动报告 | 改动前 | 改动后 |
| --- | ---: | ---: |
| 中位耗时 | 28.91 ms | 10.15 ms |
| 峰值跟踪分配 | 6,515,452 bytes | 5,204,328 bytes |

此测例约快 2.8 倍，峰值跟踪分配减少约 20%。完整报告 SHA-256 前后均为
`a7396f8d1a31d3044ccd38d0b19c5d3f94d6eca0cac4a628f19666e676163f90`。
跟踪分配不包含预先构建的输入，也不是进程 RSS；时间会随机器负载和数据变化。
原始记录在 `outputs/round25-perturbations/`。

```bash
PYTHONPATH=src python examples/benchmark_perturbations.py \
  --references 20000 --repeats 5 \
  --output outputs/round25-perturbations/reproduction.json
```

## 验证

新增 8 项独立标量回归：全成功、全失败、混合结果、仅参考位姿、无成功基准的
配对统计，以及 float32/float64/int64 的最差指标归约。覆盖无有效测量、正负
无穷、NaN、失败候选排除、源数据不变和严格 JSON 序列化。

全量 pytest：711 passed、4 skipped。Ruff、53 个 Python 文件格式、完整解析
工作流、sdist/wheel 构建及 `git diff --check` 通过。

本轮没有运行 CUDA 或远程 CI；没有提交或推送。
