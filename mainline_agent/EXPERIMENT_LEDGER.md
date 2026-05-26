# Experiment Ledger

更新时间：`2026-05-21`

## 最新输入与产物

已读取并吸收的日结：

- `mainline_agent/agent_reports/20260520_codex1_daily.md`

本轮未发现新的 `20260521_*_daily.md` 日结，也未发现 `20260520_codex2_daily.md` 文件。以下判断来自可核验产物直接检查：

- `codex1_workspace/submission_asset_checklist.md`
- `Paper/paper_draft_en_v1.md`
- `Paper/citation_todo_list.md`
- `Paper/references_seed.bib`

结论：

- codex1 已完成投稿级主表、stage-level 图表、appendix 边界表、appendix evidence note 和 submission asset checklist。
- codex2 已将英文稿推进为 tighter draft，并生成 conservative citation TODO list。
- 当前未发现新训练或新 eval expansion。
- 当前无需新增实验。

## 已足够支撑正文主表

### Final Main Table

状态：`足够`

主要文件：

- `codex1_workspace/progressive_final_main_table.csv`
- `Paper/tables/progressive_main_table.tex`

任务/安全/gate 来源：

- `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`

runtime 来源：

- `runs/progressive_mechanism_20260428/summary_metrics.csv`

用途：

- 正文主结果表。

注意：

- caption 必须说明 task/safety/gate 与 runtime 来自不同聚合口径。
- 不使用 `episode_return` 做跨目录主排序。

## 已足够支撑机制图/机制表

### Stage-Level Mechanism

状态：`足够`

主要文件：

- `codex1_workspace/progressive_stage_mechanism_table.md`
- `codex1_workspace/progressive_stage_mechanism.png`
- `Paper/tables/progressive_stage_mechanism_table.tex`
- `Paper/figures/progressive_stage_mechanism.png`
- `Paper/figures/progressive_stage_mechanism.pdf`
- `Paper/figures/progressive_stage_mechanism_caption.md`

原始来源：

- `runs/progressive_mechanism_20260428/stage_metrics.csv`

用途：

- 解释 early/mid/late 阶段 stronger layer 介入差异。

注意：

- 不能把 stage-level H2 指标改善写成 final learned policy 稳定更优。

## 已足够支撑审慎 matched 说法

### Matched Analysis

状态：`部分但足够`

主要文件：

- `codex1_workspace/progressive_matched_analysis_note.md`

原始来源：

- `runs/progressive_mechanism_20260428/matched_analysis_summary.csv`
- `runs/progressive_mechanism_20260428/matched_formal_nonprog_eta25_5x5/`
- `runs/progressive_mechanism_20260428/matched_pilot_nonprog_threshold_scan_3x3/`

用途：

- 支持“收益不宜简单归因于 gate more / compute more”。

注意：

- 不是完整 threshold frontier sweep。
- 当前 draft 不需要新实验。

## 已足够支撑 appendix/boundary

### H2 Boundary

状态：`足够作为边界材料`

主要文件：

- `codex1_workspace/progressive_appendix_evidence_index.md`
- `Paper/tables/appendix_h2_boundary_table.tex`
- `Paper/appendix_evidence_note.md`

主要目录：

- `runs/final_formal_h2_vs_h1_multiseed3x3/`
- `runs/h1_h2_cross_eval_multiseed3x3/`
- `runs/h1_h2_fixedpoint_compare_stable3x3/`
- `runs/h1_h2_fixedpoint_compare_refine3x3/`

用途：

- 支持 stronger runtime filtering 与 better learned policy 不单调。

### Dual Boundary

状态：`足够作为边界材料`

主要文件：

- `Paper/tables/appendix_dual_boundary_table.tex`
- `Paper/appendix_evidence_note.md`

主要目录：

- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/`

用途：

- 说明 dual 改变 runtime，但当前主安全指标未稳定优于 threshold-only。

### Exact/Projected `A_hard` Diagnostics

状态：`足够作为理论/诊断支撑`

主要文件：

- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`
- `Paper/appendix_evidence_note.md`

主要目录：

- `runs/model_compare_exact_hard_solver_fast2x1/`
- `runs/model_compare_exact_hard_solver_diag_medium2x2/`

用途：

- 支撑 exact/projected `A_hard` 语义、sequential approximation、true dead-end vs approximation-induced dead-end。

## 不建议重复的工作

- 不重复跑大规模 progressive 主比较训练。
- 不重新盘点已有实验目录，除非出现数字冲突。
- 不把 H2、dual、exact `A_hard` 扩成正文主线。
- 不为当前 draft 补完整 matched frontier，除非后续决定把机制因果 claim 写得更强。

## 投稿前材料风险

- `Paper/tables/*.tex` 使用 `booktabs` 命令，最终 LaTeX preamble 需要包含 `\usepackage{booktabs}`。
- 主表中 task/safety/gate 与 runtime 来自不同聚合口径，caption 和正文必须保留说明。
- 英文稿已经不是空 skeleton，但仍不是最终 polished submission。
- `Paper/citation_todo_list.md` 记录了剩余引用缺口，后续必须核验真实引用或保留 TODO，不得臆造。
- stage 图若用于双栏论文，可能需要 full-width figure 或缩短 x-axis labels。
