# Progressive Mainline Assets

生成时间：2026-05-12

本文档记录 codex1 为 progressive shielding / conservativeness curriculum 主线整理的材料资产。

## 1. Final Main Table CSV

合并表路径：

- `codex1_workspace/progressive_final_main_table.csv`

合并表只包含三行：

- `non_progressive`
- `threshold_only_progressive`
- `safeearly_progressive`

### 1.1 列来源

任务/安全/gate 指标来自：

- `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`

对应列映射：

- `search_rate` <- `search_rate_mean`
- `coverage_ratio` <- `coverage_ratio_mean`
- `collision_count` <- `collision_count_mean`
- `guarantee_broken_rate` <- `guarantee_broken_rate_mean`
- `dead_end_rec_rate` <- `dead_end_rec_rate_mean`
- `recursive_gate_rate` <- `recursive_gate_rate_mean`

runtime 指标来自：

- `runs/progressive_mechanism_20260428/summary_metrics.csv`

对应列映射：

- `perf_shield_time_ms` <- `perf_shield_time_ms_mean`
- `perf_recursive_time_ms` <- `perf_recursive_time_ms_mean`

来源列：

- `task_safety_source` 记录任务/安全/gate 指标来源。
- `runtime_source` 记录 runtime 指标来源。

### 1.2 主表可写结论

- `threshold_only_progressive` 是当前最稳的主正结果候选。
- 相比 `non_progressive`，`threshold_only_progressive` 在 `guarantee_broken_rate` 与 `dead_end_rec_rate` 上更低，`search_rate` 基本持平并略高。
- 该结果支持 progressive conservativeness curriculum 的审慎表述：hard-safe 始终保留时，threshold curriculum 可以带来有限但可观察的安全/未来可行性改善。
- `safeearly_progressive` 可作为 late-stage stronger layer 的对照或消融。

### 1.3 不能写太满的结论

- 不能写 `threshold_only_progressive` 全面优于 `non_progressive`，因为它的 `collision_count` 更高，runtime 也更高。
- 不能写 `safeearly_progressive` 是更强成功版本，因为它没有稳定优于 `threshold_only_progressive`。
- 不能用本主表证明 H2 或 dual scheduling 是正文主成功分支。
- 不能把 `episode_return` 作为跨全部目录的主排序指标；合并 CSV 中也未包含该字段。
- 需要在 caption 或正文中说明：任务/安全/gate 指标与 runtime 指标来自不同聚合口径。

## 2. Progressive Stage-Level Mechanism

图路径：

- `codex1_workspace/progressive_stage_mechanism.png`

附表路径：

- `codex1_workspace/progressive_stage_mechanism_table.md`

生成脚本：

- `scripts/plot_progressive_stage_mechanism.py`

数据来源：

- `runs/progressive_mechanism_20260428/stage_metrics.csv`

使用口径：

- 仅使用 `row_type=aggregate` 且 `split=eval` 的 stage-level 聚合行。

图中展示：

- `recursive_gate_rate_mean`
- `dead_end_rec_rate_mean`
- `perf_shield_time_ms_mean`

### 2.1 可支撑的机制表述

- `non_progressive` 固定运行在 `recursive / H=1 / eta=0.35`。
- `threshold_only_progressive` 的 early 阶段为 `safe / H=1 / eta=0.90`，`recursive_gate_rate=0`，说明 early 阶段主要停留在 hard-safe / safe 层。
- `threshold_only_progressive` 的 mid/late 阶段切入 `recursive / H=1 / eta=0.35`，其 gate rate 与 `non_progressive` fixed 阶段接近。
- `safeearly_progressive` 的 late 阶段切入 `recursive / H=2 / eta=0.55`，stage-level 上 gate rate 和 recursive dead-end rate 明显降低，但这不能自动推出 final learned policy 更优。
- 该图支持当前主线写法：progressive 调节的是 stronger layer 的介入时机和保守性强度，而不是打开/关闭 hard-safe。

### 2.2 不能写太满的地方

- 不能只凭 stage-level 图写 `threshold_only_progressive` 的收益完全来自某个唯一因果机制。
- 不能把 `safeearly_progressive` late 阶段的 H2 机制指标改善写成最终策略稳定更优。
- 不能把该机制图替代 final main table；它服务解释 early/mid/late 阶段差异。
- 不能把 H2 由此升级为正文主成功分支。

## 3. Matched Analysis Evidence Boundary

说明文件路径：

- `codex1_workspace/progressive_matched_analysis_note.md`

主数据源：

- `runs/progressive_mechanism_20260428/matched_analysis_summary.csv`

直接依赖目录：

- `runs/progressive_mechanism_20260428/matched_formal_nonprog_eta25_5x5/`
- `runs/progressive_mechanism_20260428/matched_pilot_nonprog_threshold_scan_3x3/`

### 3.1 核心证据

- Matched gate-rate：`threshold_only_progressive` 与 `non_progressive_default_eta035` 的 recursive gate-rate gap 约 `0.0076`；前者 `guarantee_broken_rate` 与 `dead_end_rec_rate` 更低，但 collision 和 runtime 更高。
- Matched compute-budget：`threshold_only_progressive` 与 `non_progressive_eta025` 的 shield-time gap 约 `7.14 ms`；`non_progressive_eta025` gate rate 更高，但 `collision_count`、`guarantee_broken_rate` 和 `dead_end_rec_rate` 都更差。

### 3.2 可支撑的机制表述

- 已有 matched 证据不支持把 `threshold_only_progressive` 的收益简单归因于 gate more 或 compute more。
- 当前更稳妥的解释是：progressive conservativeness curriculum 改变了 stronger layer 介入的训练阶段和分布，而不仅是提高平均 gate rate。

### 3.3 不能写太满的地方

- matched analysis 仍不是完整 threshold frontier sweep。
- 不能写已经完全消除所有 gate / compute confound。
- 不能写 `threshold_only_progressive` 全面支配 non-progressive。
- 当前 draft 不需要新实验；如需加强机制结论，后续只需补小规模 eval frontier，不需要新训练。

## 4. Appendix Evidence Index

索引文件路径：

- `codex1_workspace/progressive_appendix_evidence_index.md`

覆盖内容：

- H2 边界结果
- dual 边界结果
- exact/projected `A_hard` 诊断

### 4.1 H2 边界材料

可引用目录：

- `runs/final_formal_h2_vs_h1_multiseed3x3/`
- `runs/h1_h2_cross_eval_multiseed3x3/`
- `runs/h1_h2_fixedpoint_compare_stable3x3/`
- `runs/h1_h2_fixedpoint_compare_refine3x3/`

可支撑：

- H2 是 runtime stronger-layer 候选，但不是 learned policy 主成功。

### 4.2 Dual 边界材料

可引用目录：

- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/`

可支撑：

- dual 可以改变 runtime，但当前主安全指标没有稳定打赢 `threshold_only_progressive`。

### 4.3 Exact/Projected `A_hard` 支撑材料

可引用目录：

- `runs/model_compare_exact_hard_solver_fast2x1/`
- `runs/model_compare_exact_hard_solver_diag_medium2x2/`

可支撑：

- exact/projected `A_hard` 可作为理论参照。
- sequential 是工程近似。
- dead-end 可以拆成 true dead-end 与 approximation-induced dead-end。

### 4.4 使用边界

- 这些材料只能作为 appendix / discussion / theory support。
- 不要用它们改写正文主线。
- 不要把 H2、dual、exact solver 诊断升级成正文主结果。
- 当前 draft 不需要新实验。

## 5. Submission-Ready Tables and Figures

本轮新增投稿级图表与附录材料。

### 5.1 LaTeX Tables

主结果表：

- `Paper/tables/progressive_main_table.tex`

Stage-level 机制表：

- `Paper/tables/progressive_stage_mechanism_table.tex`

Appendix 边界表：

- `Paper/tables/appendix_h2_boundary_table.tex`
- `Paper/tables/appendix_dual_boundary_table.tex`
- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`

使用边界：

- `progressive_main_table.tex` 可作为正文主表。
- `progressive_stage_mechanism_table.tex` 可作为机制分析表或 appendix 表。
- appendix 三张表只用于 boundary / appendix / theory-support，不应写成正文主成功证据。

### 5.2 Paper-Style Stage Figure

投稿版图：

- `Paper/figures/progressive_stage_mechanism.png`
- `Paper/figures/progressive_stage_mechanism.pdf`

图注：

- `Paper/figures/progressive_stage_mechanism_caption.md`

生成脚本：

- `scripts/plot_progressive_stage_mechanism_paper.py`

使用边界：

- 图可以支持 early / mid / late stage 机制解释。
- 图注已明确 safeearly late 的 H=2 机制变化不能写成 final learned policy uniformly better。

### 5.3 Appendix-Ready Evidence Note

附录材料说明：

- `Paper/appendix_evidence_note.md`

结构：

- Appendix A. H2 Boundary Results
- Appendix B. Dual Scheduling Boundary Results
- Appendix C. Exact/Projected `A_hard` Diagnostics

使用边界：

- 供 codex2 直接摘取附录材料和推荐表述。
- 不用于改写 progressive 主线。
- 当前 draft 不需要新实验。
