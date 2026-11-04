# 平面可达性任务：从选择到导出

## 浏览器操作

安装 `.[viser]` 后，在仓库根目录运行以下不依赖外部机器人资产的检查：

```bash
python -m workspace_analyzer tests/fixtures/cartesian_stage.urdf \
  --backend numpy --mode plane --plane xy --frame base \
  --center 0 0 0 --constant 0 --span 3 --resolution 5 \
  --batch-size 7 --ik-restarts 1 --ik-rescue-restarts 0 \
  --viser --port 8080 --output outputs/plane.npz --report outputs/plane.json
```

打开 `http://localhost:8080`，在 **Plane reachability** 面板中：

1. 选择 **Sampling frame** 和 **Plane**（XY/XZ/YZ）；下拉框包括所有 URDF 连杆。
2. 调整中心和边长。在 **Plane sampling** 选择 grid 后设置每轴点数；
   或选择 random 等策略后设置 **Plane sample count** 总数。
3. 可点击 **Center plane on current TCP**，将中心放到当前关节滑块对应的 TCP 位置，
   自动换算到所选采样坐标系。此操作不修改 IK 参考初值或 TCP RPY。
   **Show plane preview** 控制橙色边框；明确配置平面入口时默认开启。
   计算期间禁用采样配置与 TCP 定位。
4. 默认只约束 TCP 位置；勾选 **Constrain TCP orientation** 后设置 TCP RPY（弧度）。
5. 点击 **Compute plane reachability**，查看进度与成功点数。可随时取消。
6. 查看二维采样图，下载最近完成的 JSON、NPZ、SVG；命令指定输出路径时也会自动保存。
   如果保存失败，计算结果仍可查看和下载；修复目标路径后点击
   **Retry saving last plane result**，只重试失败文件，不重新运行 IK。
   重试使用上次完成的结果，不受后续控件改动影响。

这里解析机构的可达位置恰为 `[-1,1]³`。上述 XY 平面的 25 个采样点中，应有
9 个成功。切换到 XZ，将边长改为 1、每轴点数改为 7，再次计算应为 49/49。
纯平移机构不能完成非零旋转，可以用来检查固定姿态选项确实参与求解。

`--viser` 先显示编辑控件，等待点击才计算；去掉 `--viser` 则立即计算并导出。
普通 Viser demo 也提供平面面板，默认不叠加平面预览；开启预览或计算即可使用。
调整设置不会把旧结果当作新结果；取消或失败保留最近完成的结果。
计算中点击 **Capture current FK pose** 捕获不同的参考配置时，会取消正在运行的
平面或 Cartesian 任务，保留最近完成的显示和文件。等待取消完成后可再次计算，
新任务使用刚捕获的配置。重复捕获相同配置不会中断任务。
任务配置无效或后台线程无法启动时，页面显示失败原因并恢复计算控件，
最近完成的结果仍可使用；修正问题后可直接重试，无需重启页面。

## 实际机器人与坐标系

```bash
python examples/plane_reachability.py --urdf /path/to/robot.urdf \
  --base-link left_arm_base --tip-link left_ee \
  --plane xy --frame world --constant 0.8 --center 0.4 0 0.8 \
  --span 0.8 --resolution 21 --position-only --backend numpy \
  --output outputs/robot-plane.json --viser
```

Demo 识别到 `left_arm_base/left_ee` 或右臂对应名称时，使用纯手臂链；
其他 URDF 保留自动链选择，也可明确指定链名。输入错误以退出码 2 和简短信息返回。
单臂分析、三组评测及报告复现见[示例指南](../examples/README.md)。

| 设置 | 含义 |
| --- | --- |
| `--plane xy` | 在采样坐标系中改变 X、Y，固定 Z；XZ 固定 Y，YZ 固定 X |
| `--frame base` | 求解器基座坐标系 |
| `--frame world` | URDF 根坐标系，上游关节置零 |
| `--frame reference` | 第二个活动关节子连杆在零关节状态下的坐标系；单关节链使用第一个 |
| `--frame LINK_NAME` | 指定连杆在零关节状态下的坐标系 |
| `--constant C` | 覆盖中心的法向分量；不指定时保留 center 的对应分量 |
| `--span L --resolution R` | 边长 L 米，包含端点的 R×R 方形网格，至少每轴 2 个点 |
| `--full-pose --rpy R P Y` | 主 CLI 使用采样坐标系中的固定 TCP 姿态；三个角度均为弧度 |
| `--ik-max-iterations` | 每次 IK 尝试的迭代上限 |
| `--ik-restarts / --ik-rescue-restarts / --ik-rescue-rounds` | 初始尝试与失败重试预算 |

参考连杆轴可能与世界轴不同，reference 中的 XY 平面不一定是世界水平面。
这些坐标系不会随关节滑块改变，捕获 FK 参考配置只改变后续求解初值。
**Center plane on current TCP** 使用当前滑块的 FK，而非上一次捕获的参考位置。
**Solver settings** 分别提供 **Sampling seed** 与 **IK random seed**：前者控制目标点，
后者控制 IK 随机重启。CLI 的 `--seed` 初始设置两者；API 可分别配置。
网格不使用采样种子，因此该控件在 grid 模式禁用。

主 CLI 和 demo 统一默认 XY/base/仅位置、中心 `[0,0,0]`、RPY `[0,0,0]`，
不再隐式覆盖法向坐标。`--full-pose` 才会约束姿态，`--position-only` 可显式选择位置任务。
主 CLI 支持 `workspace-analyzer`、`python -m workspace_analyzer` 和
`python -m workspace_analyzer.cli`。指定 `--plane` 时自动选择 plane 模式；
与显式的其他 `--mode` 冲突会报错。

规则网格用 `--resolution R`；任意数量采样用 `--samples N`，默认 random，
可附加 `--strategy random|uniform|halton|sobol|lhs|gaussian`。
`--strategy` 在平面模式下要求 `--samples`，两者均不能与 `--resolution` 混用。
Halton/Sobol/LHS 需要安装 `.[sampling]`；random/uniform/gaussian 只依赖 NumPy。
`uniform` 使用二维网格的前 N 个点，N 非平方数时可能不覆盖网格所有行；
需要完整对称网格时用 `--resolution`。
指定目标文件的 `targets` 模式保留所有输入行，不接受 `--samples` 或 `--strategy`。

```bash
python -m workspace_analyzer tests/fixtures/cartesian_stage.urdf \
  --plane xz --center 0 0.2 0 --span 3 --samples 1000 --strategy random \
  --seed 42 --backend numpy --ik-restarts 1 --ik-rescue-restarts 0 \
  --output outputs/sampled-plane.npz --report outputs/sampled-plane.json \
  --plot outputs/sampled-plane.svg
```

主 CLI 的质量门槛也可用于平面任务，例如 `--min-joint-limit-margin 0.1`。

## 倾斜平面

用 `--frame-rpy R P Y` 或页面 **Sampling frame RPY (rad)** 旋转采样坐标系，
即可在任意方向的平面上采样。三个角均为弧度，旋转为 `Rz(yaw) Ry(pitch) Rx(roll)`。
它在所选 `--frame` 的原点复合旋转：`base_from_frame @ rotation(frame_rpy)`。
中心、法向常数、平面轴及 TCP RPY 都在这个旋转后的坐标系中表达。
它不是绕已经指定的平面中心旋转；需要保持某个物理中心时，应换算中心坐标，
或旋转后点击 **Center plane on current TCP**。
JSON 的 `frame_rpy_rad` 和 `base_from_plane_frame` 记录实际变换。

以下可直接运行，解析平移机构应得到 **15/25** 个有解点：

```bash
python -m workspace_analyzer tests/fixtures/cartesian_stage.urdf \
  --backend numpy --plane xy --frame-rpy 0.7853981633974483 0 0 \
  --span 2.4 --resolution 5 --ik-restarts 1 --ik-rescue-restarts 0 \
  --progress --min-success-rate 0.6 \
  --output outputs/tilted.npz --report outputs/tilted.json --plot outputs/tilted.svg
```

此处仍为仅位置任务。加 `--full-pose` 将要求滑台完成倾斜后的 TCP 姿态，因此无解；
同时加 `--rpy -0.7853981633974483 0 0` 抵消旋转后，仍应有 15/25 个有解点。

## 输出与 Python API

JSON 包括坐标变换、网格行列轴和坐标、IK 预算、任务类型、成功率及质量统计。
NPZ 包括目标位姿、关节解、残差、质量测量和相同的平面元数据，均采用求解器基座
坐标表达。`grid` 的行对应平面名称第一个轴、列对应第二个轴。
Demo 的 `--output plane.json` 或 `--output plane.npz` 都会写出 JSON、NPZ、SVG；
主 CLI 分别用 `--output`、`--report`、`--plot` 指定。
输出路径会在计算前检查：NPZ 后缀归一化后的路径、JSON 和 SVG 必须互不冲突，
也不能覆盖输入 URDF 或目标文件；指向相同文件的链接同样报错。
每个文件独立保存，页面会列出成功路径与失败原因。失败文件可能仍是上一次结果，
不能仅凭“计算完成”判断所有文件都已更新。JSON、SVG、NPZ 均先完整写入再替换，
单个文件写入失败时保留旧文件。
CLI 会尝试全部平面输出；部分失败时返回退出码 2，并列出已经成功保存的文件。
任意数量采样的 `grid`、`resolution`、行列坐标为 null；报告记录实际策略、采样种子、
局部 bounds，逐点坐标保存在 NPZ。Python API 在 `PlaneConfig` 中传
`sampling=SamplingConfig(strategy="random", num_samples=1000, seed=42)` 即可。
二维图在采样坐标系表达，绿色表示通过质量门槛、黄色表示 IK 成功但质量未通过、
红色表示 IK 未找到解。图只标出实际样本，不对未采样区域插值。

```python
from workspace_analyzer import PlaneConfig, ReachabilityConfig, analyze_plane, create_solver

solver = create_solver("tests/fixtures/cartesian_stage.urdf", backend="numpy")
result = analyze_plane(
    solver,
    PlaneConfig(plane="xy", frame="base", constant=0.2, span=3, resolution=5),
    ReachabilityConfig(position_only=True, restarts=1, rescue_restarts=0),
)
result.save("outputs/plane.npz")
print(result.metadata["plane_scan"]["success_rate"])
```

同一配置、初值、后端、精度和批大小下可复现采样与求解。不同预算可能得到不同
分类；红点只表示本次 IK 未找到解。网格是有限采样，未验证点不具有可达性结论，
结果不包含碰撞检查或实机执行验证。

## 命令验收与离线复评

`--progress` 将进度写入 stderr，支持四种无界面分析模式；Ctrl-C 返回 130，
未完成的计算不会替换已有产物；平面兼容 demo 使用相同的中断处理。浏览器已有进度，因此不与 `--viser` 混用。
`--min-success-rate` 和 `--min-quality-pass-rate` 是显式验收门槛，取值 0..1：

| 退出码 | 含义 |
| --- | --- |
| 0 | 正常完成，且满足所有指定验收门槛 |
| 1 | 完成测量，但整体成功比例或质量通过比例不足；仍保存诊断产物 |
| 2 | 参数、输入或保存错误 |
| 130 | 用户中断 |

质量比例以全部目标为分母，单点质量门槛仍用 `--min-isotropy`、
`--min-singular-value`、`--min-joint-limit-margin`、`--require-full-rank`。
质量验收适用于 targets/plane，成功比例验收也支持 Cartesian 模式；
验收参数不用于 joint 模式或浏览器。空结果不能通过验收，即使最低比例设为零。

已保存的结果不需要重新加载 URDF、运行 IK 或 Jacobian，可以直接查看和复评：

```bash
workspace-analyzer inspect outputs/tilted.npz \
  --min-joint-limit-margin 0.2 --min-quality-pass-rate 0.2 \
  --report outputs/tilted-review.json --plot outputs/tilted-review.svg \
  --output outputs/tilted-review.npz
```

`inspect` 标准输出为 JSON 摘要。默认保留原有质量门槛，只覆盖明确传入的项；
`--reset-quality-gates` 先清空旧门槛，`--allow-rank-deficient` 可取消满秩要求。
原文件不会被修改，输出路径不能与输入冲突。每次查看都会根据逐点结果重新计算
已保存要求下的验收结论，并继续决定退出码，不沿用文件中的旧 PASS/FAIL。
指定新的验收参数只覆盖对应项，例如只修改质量通过率不会清除 IK 成功率要求。
用 `--reset-acceptance` 清除全部旧验收要求，再按需传入新要求；它不清除单点质量门槛，
`--reset-quality-gates` 也不清除整体验收要求。已保存的验收要求格式无效时退出 2，
可用 `--reset-acceptance` 显式丢弃后重新设定。
NPZ 保存逐点数据，JSON 保存完整元数据摘要；SVG 仅支持 plane 结果。
单个导出发生写入错误时仍尝试其他导出，退出 2 并列出成功路径。

大规模平面在浏览器的二维图最多显示 5,000 个样本，按分类保留代表点，
防止稀少的失败或质量不通过点消失；图中明确标注显示数量。
下载及文件 SVG 仍包含全部样本。三维平面使用离散点云，绿／黄／红含义与二维图一致；
透明度控件仅用于非平面可达性体积图。三维显示超过点数上限时也标注显示数量。

一小时优化的压力测试、浏览器验证及性能对照见
[本次验证记录](optimization-2026-09-13-hour.md)。
