# 第八轮：平移边界细化与失败端点复查

日期：2026-09-05。本轮把上一轮出现的 X 正向扰动失败进一步定位为可检查的
数值区间，并将定位过程接入现有测试流程。

## 完成内容

- 新增 `BoundaryConfig`、`BoundaryResult`、`refine_translation_boundary`，
  支持任意方向、固定参考姿态的多个平移线段。
- 仅细分尚未达到分辨率的成功/失败区间，成功配置作为下一步初值；中点已无法
  区分于端点时及时停止，避免重复计算相同浮点目标。
- 最终失败端点重新使用最近成功配置检查。若恢复成功，保留原始与复查测量，
  标记 `end_recovered`，不将该区间计入已完成定位。
- 起点失败、终点成功、预算不足、浮点精度限制都有独立状态。质量门槛仅用于
  配置诊断，二分依据为 IK 成功标记。
- 保存逐次测量、线段参数、对应线段、阶段、端点索引、实际区间宽度、求解配置
  和批次缓存信息；重复相同配置可复用全部测量。
- 工作流新增 `--boundary-test`、距离、分辨率和轮数参数，未满足边界验收条件
  时返回 1，保留诊断数据。Make 入口与可选运行时 CI 配置已接入解析机构验收。

## 验证

| 检查 | 结果 |
| --- | --- |
| 全量 pytest | **395 passed，11.59 秒**，比上一轮增加 35 项 |
| Ruff 与格式检查 | 通过，42 个 Python 文件 |
| 解析三轴机构完整流程 | 16 条线段全部形成有效区间并达到 0.1 mm 分辨率 |
| 独立解析基准 | 立方体正负方向和二连杆半径 2 m 的外边界通过 |
| 后端与精度 | NumPy/Torch CPU、float32/float64 回归通过 |
| 预算、精度和取消 | 轮数不足、初始区间已足够小、精度停滞、开始/中途/最终取消均通过 |
| 缓存与报告 | 重放不调用 IK；查询顺序、NPZ 和严格 JSON 往返验证通过 |
| 端点恢复反例 | 模拟近端初值使失败恢复的案例，确认不会虚报已定位 |
| 源码包与 wheel | 构建通过 |
| 安装后验证 | 独立目录导入，Torch CPU/float32 细化和 JSON/NPZ 往返通过；Marvin 保存端点与轨迹对应关系通过 |
| 文档链接、CI YAML、`git diff --check` | 通过 |

完整检查使用本机已有构建依赖：

```bash
PYTHONPATH=/tmp/workspace-analyzer-build make check \
  BUILD_FLAGS='--no-isolation --outdir /tmp/workspace-analyzer-round8-dist'
```

一般开发环境安装 `.[dev]` 后直接执行 `make check`。远程 GitHub CI 和 CUDA
没有在本轮本地验证中运行；此次没有提交或推送。

## Marvin 上一轮失败目标的细化结果

输入来自 `outputs/round7-marvin-workflow/perturbations.npz` 中失败的行 144、157，
对应原始参考 11、12，均为零基索引。线段起点为各自参考位姿，终点为基座 X
正向偏移 1 cm，保持完整姿态约束。起点初值来自参考已找到的 IK 配置。

配置：Marvin `left_arm_base → left_ee`，NumPy/float64，种子 77，batch size 16，
最大迭代 400，16 个重启、失败补救 32 个重启共 2 轮。二分分辨率 0.1 mm，最多
16 轮。质量门槛为位置 Jacobian 各向同性 0.05、关节余量 0.1。

| 原始参考 | 成功端 X 偏移 | 失败端 X 偏移 | 区间宽度 | 二分轮数 |
| --- | ---: | ---: | ---: | ---: |
| 11 | 7.265625 mm | 7.343750 mm | 0.078125 mm | 7 |
| 12 | 7.343750 mm | 7.421875 mm | 0.078125 mm | 7 |

共查询 **20 个目标**，包含 4 个初始端点、14 个中点和 2 次最终复查。两条线段
均为 `refined`，最终失败端使用最近成功解复查后仍未找到解。成功端的任务残差
分别约为 `3.21e-6`、`2.87e-6`；失败端约为 `3.30e-5`、`1.55e-5`。它们是完整
位姿任务的混合残差范数，不能直接当成米。

两个成功端都没有达到本次质量门槛，因此仅用 IK 可达边界作为作业区边界仍不足。
后续需要继续考虑质量要求和多分支选择。

结果保存于 `outputs/round8-marvin-boundary/`：`report.json`、`measurements.npz`
和 `trace.npz`，可逐项核对端点配置、残差及原始线段参数。读取示例：

```python
import json
import numpy as np
from pathlib import Path
from workspace_analyzer import AnalysisResult

folder = Path("outputs/round8-marvin-boundary")
measurements = AnalysisResult.load(folder / "measurements.npz")
report = json.loads((folder / "report.json").read_text())
with np.load(folder / "trace.npz", allow_pickle=False) as trace:
    assert measurements.reachable[trace["lower_indices"]].all()
    assert not measurements.reachable[trace["upper_indices"]].any()
```

## 范围与下一步

这里定位的是给定初值、求解预算和容差下观察到的分类变化区间，并非数学意义的
几何边界证明。算法不发现同状态端点之间的空洞，也不保证得到沿线最早或最外的
边界；全区域粗扫描、多区间发现和质量边界尚未实现。

下一步可围绕这些区间增加种子/预算对比，并将线段细化用于区域粗网格中发现的
多个变化段。使用方法见 [边界测试指南](boundary-testing.md)，总体计划见
[验证方案分析](validation-options.md)。
