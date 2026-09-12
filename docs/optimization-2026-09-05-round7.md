# 第七轮：可验证的配对扰动测试

日期：2026-09-05。本轮将“任务偏移后还能否完成”落为可复现的测试用例，并为
后续边界扫描、求解稳定性、独立运动学对照和碰撞检查给出具体的验收方法。

## 实现

- 新增 NumPy 核心接口 `PosePerturbations`：每个参考最多生成 13 个有固定顺序
  的位姿，包含参考、XYZ 正负位移与三轴正负旋转；基座轴和工具轴语义明确。
- 输入创建只读快照，float32 保持精度类型。零幅度关闭对应变体，避免重复目标；
  所有求解复用现有分批、取消和测量缓存，不展开额外重启维度。
- 汇总前检查目标坐标和顺序，按参考配对统计 IK/质量的失败与恢复，输出全变体
  通过率、条件保持率、方向敏感性以及成功配置中的最差指标。
- 汇总不调用运动学求解，支持从保存结果离线修改质量门槛。零扰动、无成功参考
  和缺失有限指标有明确的 JSON `null` 语义。
- 工作流增加 `--robustness-test`、位移/旋转幅度及 `--min-robust-rate`。
  Markdown 报告补充观察值、验收值和逐方向配对变化；失败继续保留原始数据。
- `make check`、`make test-workflow` 及可选运行时 CI 已接入解析机构扰动流程。
  CI 修改仅为配置，本轮未运行远程 GitHub 作业。

## 验证结果

本地环境：Python 3.10、NumPy 1.26.3，Torch CPU、SciPy 和 Viser 可用。

| 检查 | 结果 |
| --- | --- |
| 全量回归 | **360 passed，9.92 秒**，比上一轮增加 37 项 |
| Ruff / 格式检查 | 通过，40 个 Python 文件 |
| 解析机器人完整流程 | 通过；正确识别平移可达、非零旋转无法完成 |
| Marvin NumPy/float64 工作流 | 全部已配置验收项通过；扰动成功率作为诊断保存 |
| 源码包和 wheel | 构建通过 |
| 安装后离线分析 | 独立目录导入新 API，保存结果汇总与原报告完全相同；质量门槛重置通过 |
| 安装后 Torch CPU/float32 | Marvin 的 4 个参考、48 个扰动全部 IK 成功，严格 JSON 输出通过 |
| CI YAML、文档链接、`git diff --check` | 通过 |

完整检查命令，本机使用已有构建依赖：

```bash
PYTHONPATH=/tmp/workspace-analyzer-build make check \
  BUILD_FLAGS='--no-isolation --outdir /tmp/workspace-analyzer-round7-dist'
```

普通环境安装 `.[dev]` 后运行 `make check`。新增测试包含解析边界、质量门槛、
基座/工具轴、旋转保持 TCP 位置、刚体换基、零幅度、空成功集、输入错配、
NumPy/Torch CPU 和 float32/float64、缓存不调用求解器，以及验收失败报告保留。

## Marvin 测例与失败复查

```bash
PYTHONPATH=src python examples/reachability_workflow.py \
  --urdf ../HumanoidAssets/Marvin_M6_S_CCS_696_V4.0/robot.urdf \
  --base-link left_arm_base --tip-link left_ee --robustness-test \
  --output-dir outputs/round7-marvin-workflow
```

NumPy/float64，种子 77，16 个关节中位附近的参考，batch size 16，最大迭代 150，
4 个重启、失败补救 8 个重启共 2 轮。位移为基座轴 ±1 cm，旋转为工具轴 ±5°。
质量门槛为位置 Jacobian 各向同性 0.05、归一化关节余量 0.1。

| 指标 | 结果 |
| --- | ---: |
| 参考 IK 成功 | 16/16 |
| 扰动 IK 成功，不含参考 | 190/192，98.96% |
| 参考及全部变体均 IK 成功的任务 | 14/16，87.5% |
| 参考质量达标 | 14/16 |
| 扰动质量达标，不含参考 | 152/192，79.17% |
| 参考及全部变体均质量达标的任务 | 3/16，18.75% |
| 相比通过的参考，变体质量失败 | 27 对 |
| 相比未通过的参考，变体质量恢复 | 11 对 |

两个 IK 失败均为基座 X 正向 1 cm 偏移，在原始目标数组中的行号为 144、157。
另行仅对这两个目标复查：最大迭代 400、16 个重启、失败补救 32 个重启共 2 轮，
使用对应参考的成功 IK 配置作为初值。结果仍为 0/2 成功；任务残差分别从
0.004081、0.006167 降为 0.004042、0.005161。完整位姿残差是混合任务范数，
不能把这些数字直接解释为米。

数据位于 `outputs/round7-marvin-workflow/`：`report.json`、`report.md`、
`perturbation_targets.npz`、`perturbations.npz`。额外复查保存为
`retry_diagnostics.json` 与 `failed_target_retry.npz`；安装包 float32 检查保存为
`wheel_float32_check.json`。这些生成产物不进入源码包。

## 结论和后续方案

本测例的普通参考任务均能求解，但扰动揭示了方向敏感性和已选配置的质量变化。
建议优先围绕 X 正向失败目标做局部边界扫描，并补充多初值/预算对比，区分有限
搜索的影响。更多方案按独立判定依据排序见 [验证方案分析](validation-options.md)。

这次测试只涉及有限的单轴偏差，参考与变体独立求解，质量变化可能包含 IK 分支
变化的影响。没有碰撞检查，也不代表完整误差区域或全工作空间的保证。增加预算
后仍未找到解，不足以证明数学不可达。接口和分母见 [扰动测试指南](robustness-testing.md)。
