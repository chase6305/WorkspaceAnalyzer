# 第六轮：测试流程、离线质量重评与姿态覆盖

日期：2026-09-05。重点是让机器人测试有固定输入、明确验收条件、逐目标诊断、
可重复执行的检查入口，以及可供后续质量分析复用的测量数据。

## 完成内容

- `make check` 依次执行 lint/格式检查、pytest、解析机器人测试流程和打包。
  `make test-workflow` 可单独执行机器人验收；pytest 输出 JUnit。
- `examples/reachability_workflow.py` 通过 FK 生成已知可达目标，并根据串联链
  长度上界生成六个范围外目标，分别验证位置和完整位姿求解。IK 不使用答案配置
  作为初值。每个已知位置再测试参考姿态与工具局部 X 轴 ±15° 姿态。
- 已知目标成功率和可选质量达标率具有显式验收门槛。验收失败返回 1，并保留
  NPZ、JSON 和 Markdown 报告；参数或输入错误返回 2。默认质量门槛用于诊断，
  需设置 `--min-quality-rate` 才作为验收条件。
- `AnalysisResult.reassess_quality` 使用已有数组重新判定质量，不调用 IK、FK
  或 Jacobian。质量要求完整替换，未指定的门槛关闭；任务权重默认保留。
- 目标缓存按测量内容寻址，修改质量门槛或任务权重只更新判定与统计；目标、
  初值、模型、求解配置或 Jacobian 行权重变化时仍重新测量。
- `AnalysisResult.orientation_coverage(position_ids)` 按位置统计有限姿态样本的
  原始和加权 IK/质量覆盖，保留 ID 首次出现顺序；零权重组的加权率输出为 JSON
  `null`，重复姿态按输入次数计数。
- CI 配置保留 Python 3.10/3.12 核心运行时矩阵，增加 Torch CPU/SciPy/Viser
  可选运行时作业，验证导入及本地端口并留存测试报告。

## 验证结果

本地 Python 3.10、NumPy 1.26.3、Torch CPU 可用环境：

| 检查 | 结果 |
| --- | --- |
| 完整 pytest | **323 passed，6.18 秒**，相较上一轮新增 32 项测试 |
| Ruff 与格式检查 | 通过，38 个 Python 文件 |
| 三轴平移机构工作流 | 全部默认验收项通过 |
| Marvin 左臂工作流 | 全部默认验收项通过，详见下表 |
| 源码包与 wheel 构建 | 通过 |
| 安装后 wheel | 从独立目录导入，加载已保存结果、离线重评、覆盖汇总与严格 JSON 导出通过 |
| 安装后 Torch CPU 流程 | 使用仓库内解析机构完成验收 |
| CI 配置 | YAML 解析通过；远程 GitHub 作业尚未运行 |
| `git diff --check` | 通过 |

本环境使用已有的本地构建依赖执行完整检查：

```bash
PYTHONPATH=/tmp/workspace-analyzer-build make check \
  BUILD_FLAGS='--no-isolation --outdir /tmp/workspace-analyzer-round6-dist'
```

普通开发环境安装 `.[dev]` 后直接运行 `make check` 即可。新增回归覆盖门槛重置、
原始结果不变、缓存复用不调用求解器、测量缺失后重算、取消、交错位置分组、
极端与零权重、输入校验，以及成功/失败退出码和失败报告保留。

### Marvin 验收

资产：`Marvin_M6_S_CCS_696_V4.0/robot.urdf`，求解链
`left_arm_base → left_ee`。NumPy/float64、种子 77、16 个已知目标、batch size 16、
最大迭代 150、4 个重启、失败补救 8 个重启共 2 轮。质量门槛为各向同性 0.05、
归一化关节余量 0.1。

| 用例 | 已知目标 IK 成功 | 范围外目标拒绝 | 全用例质量通过 |
| --- | ---: | ---: | ---: |
| 位置 | 16/16 | 6/6 | 10/22 |
| 完整位姿 | 16/16 | 6/6 | 14/22 |
| 局部姿态，共 48 个样本 | 参考姿态 16/16；全部姿态 46/48 | 不适用 | 35/48 |

结果目录：`/tmp/workspace-analyzer-round6-marvin-workflow`。重复执行命中测量缓存，
保持相同判定。可按以下命令复现，调整资产路径即可：

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf ../HumanoidAssets/Marvin_M6_S_CCS_696_V4.0/robot.urdf \
  --base-link left_arm_base --tip-link left_ee \
  --output-dir outputs/marvin-workflow
```

这些目标来自关节中位附近的有限采样；结果不代表全关节范围或所有工具姿态。

## 缓存门槛扫描性能

AMD Ryzen 9 9950X，Python 3.10、NumPy/float64。比较第五轮代码备份与本轮代码：
Marvin 左臂 64 个 FK 位姿和 8 个范围外位姿，共 72 个目标；batch size 24，
最大迭代 80，4 个重启，失败补救 8 个重启、1 轮。

先用无质量门槛的相同输入预热各自缓存，再扫描 3 个关节余量门槛并保留各向同性
门槛 0.05。重复 3 轮，轮间使门槛增加 0.001，交替新旧运行顺序；避免旧实现命中
已经计算过的完整门槛缓存。计时包含每轮 3 次缓存查询、读取及统计。

| 指标 | 第五轮 | 第六轮 |
| --- | ---: | ---: |
| 每轮扫描中位耗时 | 549.310 ms | 8.812 ms |
| 三轮耗时 | 450.027 / 549.310 / 560.863 ms | 8.812 / 9.372 / 8.540 ms |
| 9 次门槛变体中的 IK 调用，含补救 | 36 | 0 |

该测例的扫描约快 **62 倍**；所有变体的关节解、质量判定数组和 assessment 摘要
与旧实现一致。原始计时保存在 `/tmp/workspace-analyzer-round6-cache-benchmark.json`。
这是已缓存测量后修改门槛的收益，不是首次 IK 或一般工作空间分析的加速比。

## 行为约定与后续范围

离线重评返回独立元数据和新的质量判定，但共享原始测量数组，调用者应将共享
数组视为只读。缓存键现在标识测量，不唯一标识质量判定；旧目标缓存首次会失效。

姿态覆盖要求完整位姿测试，不能用位置约束 IK 推断方向覆盖。质量门槛仅检查
选中的 IK 解，尚不搜索更优分支；未执行碰撞检查。CUDA 和远程 CI 未在本次本地
验证中覆盖。通用姿态采样、覆盖图展示、边界精细扫描与扰动鲁棒性仍属后续范围。

使用步骤见 [测试流程](testing-workflow.md)，API 语义见
[可达性与灵巧度测试指南](reachability-testing.md)，后续功能见
[路线图](feature-roadmap-2026-09-05.md)。
