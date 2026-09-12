# 第二十四轮：离线覆盖与扰动报告取消

日期：2026-09-08。

姿态覆盖和扰动统计原先没有取消入口。现在以下接口接受可选的 `cancel_event`：

- `AnalysisResult.orientation_coverage(...)`
- `summarize_orientation_coverage(...)`
- `PosePerturbations.summarize(...)`

新增检查点覆盖输入读取前、分组/变体处理、指标聚合和返回前。已取消任务不读取
输入；中途取消抛出 `AnalysisCancelled`，不会返回部分报告或修改源测量。
不传事件时保持原有行为。此机制是阶段间协作取消，不会强制中断正在执行的
单次 NumPy 运算，也没有将汇总改成流式处理。

新增 8 项回归：三个入口的预取消、两类报告的中途与最终取消，以及覆盖分组
完成后触发取消。验证取消前后源坐标与质量标记不变，再次正常生成的完整报告一致。

全量 pytest：703 passed、4 skipped。Ruff、52 个 Python 文件格式、完整解析
工作流、sdist/wheel 构建及 `git diff --check` 通过。

本轮没有运行 CUDA 或远程 CI；没有提交或推送。
