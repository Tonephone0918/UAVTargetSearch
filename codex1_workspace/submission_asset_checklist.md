# Submission Asset Checklist

生成时间：2026-05-20

角色：codex1  
主线：progressive shielding / conservativeness curriculum  
正文主比较对象：`non_progressive`、`threshold_only_progressive`、`safeearly_progressive`

本清单用于投稿前材料核查。未启动新训练，未扩展 eval，未修改 `src/` 方法或训练代码。

## 0. 全局检查结论

- LaTeX 表格均存在，均有 `caption` 和稳定 `label`。
- LaTeX 表格均使用 `booktabs` 命令：`\toprule`、`\midrule`、`\bottomrule`。
- 投稿主文件 preamble 必须包含：`\usepackage{booktabs}`。
- 已修补一个小问题：`Paper/tables/progressive_stage_mechanism_table.tex` 的 `tabular` 列格式从 8 列修正为 9 列。
- `Paper/figures/progressive_stage_mechanism.png` 和 `.pdf` 均存在。
- 图注与 appendix note 均保留 H2 / safeearly / dual / exact hard 的边界表述。
- 当前不需要新实验。

## 1. 正文主表

- 文件路径：`Paper/tables/progressive_main_table.tex`
- 数据来源：
  - task/safety/gate：`runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
  - runtime：`runs/progressive_mechanism_20260428/summary_metrics.csv`
  - 合并表：`codex1_workspace/progressive_final_main_table.csv`
- 使用位置：正文主结果表。
- Label：`tab:progressive-main`
- Caption 检查：
  - 已说明 task/safety/gate 来自 formal comparison。
  - 已说明 runtime 来自 re-aggregated mechanism summary。
  - 已说明 `threshold_only_progressive` 是 mixed but useful improvement，不是全面支配。
- 可写结论：
  - `threshold_only_progressive` 相比 `non_progressive` 降低 `guarantee_broken_rate` 和 `dead_end_rec_rate`。
  - `search_rate` 基本持平并略高。
  - 这是 progressive conservativeness curriculum 的有限正结果。
- 禁止写法：
  - 不能写 `threshold_only_progressive` 全面支配。
  - 不能写所有安全指标都更好。
  - 不能忽略 `collision_count` 和 runtime 不占优。
  - 不能用 `episode_return` 作为跨目录主排序指标。
- 是否需要 codex2 引用：是，建议正文 Results 主表引用。
- 投稿前风险：
  - 必须保留 mixed aggregation caveat。
  - 需要 `booktabs`。

## 2. Stage-Level 机制表

- 文件路径：`Paper/tables/progressive_stage_mechanism_table.tex`
- 数据来源：
  - `runs/progressive_mechanism_20260428/stage_metrics.csv`
  - `codex1_workspace/progressive_stage_mechanism_table.md`
- 使用位置：正文机制分析表或 appendix 机制表。
- Label：`tab:progressive-stage-mechanism`
- Caption 检查：
  - 已说明 early 阶段主要停留在 hard-safe / safe 层。
  - 已说明 threshold-only 在 mid/late 切入 recursive feasible layer。
  - 已说明 safeearly late 切入 H=2 stronger layer。
  - 已说明 H2 late-stage 机制统计不能解释为 final learned policy uniformly better。
- 可写结论：
  - progressive 调节 stronger layer 的介入时机和保守性强度。
  - threshold-only 的 mid/late stage 使用 recursive / H=1 / eta=0.35。
  - safeearly late 使用 recursive / H=2 / eta=0.55，是 stronger-layer 消融。
- 禁止写法：
  - 不能只凭 stage 表写 threshold-only 的收益完全来自唯一因果机制。
  - 不能把 safeearly late 的 H2 机制指标改善写成最终策略更优。
  - 不能把该表替代 final main table。
- 是否需要 codex2 引用：是，建议正文机制分析或 appendix 引用。
- 投稿前风险：
  - 已修补 `tabular` 列数问题。
  - 需要 `booktabs`。

## 3. Stage-Level 机制图

- 文件路径：
  - `Paper/figures/progressive_stage_mechanism.png`
  - `Paper/figures/progressive_stage_mechanism.pdf`
  - `Paper/figures/progressive_stage_mechanism_caption.md`
- 数据来源：
  - `runs/progressive_mechanism_20260428/stage_metrics.csv`
  - 生成脚本：`scripts/plot_progressive_stage_mechanism_paper.py`
- 使用位置：正文机制图。
- 文件存在性：
  - PNG 存在：是，`1800 x 1440`。
  - PDF 存在：是，1 page PDF。
- Caption 检查：
  - 英文/中文 caption 均存在。
  - 没有把 H2 或 `safeearly_progressive` 写成 final learned policy 更优。
  - 能服务 codex2 的正文机制引用。
- 可写结论：
  - early stage 保持 hard-safe / safe 层。
  - threshold-only 在 mid/late 切入 recursive feasible layer。
  - safeearly late 切入 H=2，并改变 gate/dead-end/runtime 统计。
- 禁止写法：
  - 不能写 safeearly final learned policy 更优。
  - 不能写 H2 是正文主成功。
  - 不能把 stage 图作为完整因果证明。
- 是否需要 codex2 引用：是，建议正文 Figure 引用。
- 投稿前风险：
  - 图中横轴标签较密，若目标期刊双栏版面较窄，可能需要用 full-width figure 或缩短 x-label。

## 4. H2 Appendix 边界表

- 文件路径：`Paper/tables/appendix_h2_boundary_table.tex`
- 数据来源：
  - `runs/final_formal_h2_vs_h1_multiseed3x3/summary_metrics.csv`
  - `runs/h1_h2_cross_eval_multiseed3x3/summary_metrics.csv`
  - `runs/h1_h2_fixedpoint_compare_stable3x3/`
  - `runs/h1_h2_fixedpoint_compare_refine3x3/`
  - 索引：`codex1_workspace/progressive_appendix_evidence_index.md`
- 使用位置：appendix / boundary discussion。
- Label：`tab:appendix-h2-boundary`
- Caption 检查：
  - 已说明 H2 是 runtime stronger-layer candidate。
  - 已说明不是 main learned-policy success。
  - 已说明 cross-eval 信号更像 runtime H2 shield，而不是 H2-trained checkpoint uniformly dominating。
- 可写结论：
  - H2 可作为 runtime stronger-layer 候选。
  - H2 在部分 fixed-checkpoint / cross-eval 设置降低 recursive dead-end 或部分安全指标。
  - H2 证据支持 stronger runtime filtering 与 learned policy improvement 非单调。
- 禁止写法：
  - 不能写 H2 稳定优于 H1。
  - 不能把 H2 放进正文主结果作为成功分支。
  - 不能与 progressive formal compare 混成同一证据强度。
- 是否需要 codex2 引用：是，但只在 appendix 或 boundary discussion。
- 投稿前风险：
  - 需要 `booktabs`。
  - 表名和正文措辞必须避免 “H2 wins”。

## 5. Dual Appendix 边界表

- 文件路径：`Paper/tables/appendix_dual_boundary_table.tex`
- 数据来源：
  - `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/summary_metrics.csv`
  - 索引：`codex1_workspace/progressive_appendix_evidence_index.md`
- 使用位置：appendix / boundary discussion / future work。
- Label：`tab:appendix-dual-boundary`
- Caption 检查：
  - 已说明 dual changes runtime behavior and reduces shield time。
  - 已说明当前不改善 main safety metrics。
  - 已说明不是 main success result。
- 可写结论：
  - dual scheduling 改变 runtime，降低 shield time。
  - 当前主安全指标没有稳定优于 threshold-only progressive。
- 禁止写法：
  - 不能写 dual 是第二条成熟主创新。
  - 不能写 dual 稳定优于 threshold-only progressive。
  - 不能把 dual 表作为正文主胜利表。
- 是否需要 codex2 引用：可选；建议 appendix 或 discussion 引用。
- 投稿前风险：
  - 需要 `booktabs`。
  - 需避免 runtime 改善被误读为总体方法胜利。

## 6. Exact/Projected `A_hard` Appendix 诊断表

- 文件路径：`Paper/tables/appendix_exact_hard_diagnostic_table.tex`
- 数据来源：
  - `runs/model_compare_exact_hard_solver_fast2x1/`
  - `runs/model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`
  - 索引：`codex1_workspace/progressive_appendix_evidence_index.md`
- 使用位置：appendix / theory-support / diagnostic evidence。
- Label：`tab:appendix-exact-hard`
- Caption 检查：
  - 已说明 exact/projected view 是 semantic reference。
  - 已说明 sequential construction 是 online engineering approximation。
  - 已说明 diagnostic rates 不是 separate main result。
- 可写结论：
  - exact/projected `A_hard` 可作为 joint feasibility 语义参照。
  - sequential `A_hard` 是工程近似，会出现 false-empty / false-nonempty。
  - dead-end 可拆成 true dead-end 与 approximation-induced dead-end。
- 禁止写法：
  - 不能写 online 主路径每步 exact solving。
  - 不能把 exact solver diagnostics 写成主 empirical contribution。
  - 不能用 exact/projected 诊断替代 progressive curriculum 比较。
- 是否需要 codex2 引用：建议 appendix 方法语义或诊断段落引用。
- 投稿前风险：
  - 需要 `booktabs`。
  - `A_hard` 在 LaTeX caption 中写作 `\texttt{A\_hard}`，若正文偏数学记号，可后续统一成 `$A_{\mathrm{hard}}$`，但当前不是阻塞问题。

## 7. Appendix Evidence Note

- 文件路径：`Paper/appendix_evidence_note.md`
- 数据来源：
  - `codex1_workspace/progressive_appendix_evidence_index.md`
  - H2 / dual / exact diagnostic result files listed inside the note.
- 使用位置：appendix drafting material for codex2。
- Caption / wording 检查：
  - H2 仅作为 runtime stronger-layer candidate。
  - dual 仅作为 runtime / boundary result。
  - exact/projected `A_hard` 仅作为 semantic reference 和 diagnostic support。
  - 明确写了 appendix materials 不应升级为 main contribution。
- 可写结论：
  - stronger runtime filtering and better learned policy are not monotonically equivalent。
  - H2 / dual / exact 共同支撑边界讨论和理论支撑。
- 禁止写法：
  - 不能让 appendix note 抢走 progressive 主线。
  - 不能把 suggested wording 改成强 claim。
- 是否需要 codex2 引用：是，作为 appendix drafting source。
- 投稿前风险：
  - note 是 Markdown，不是最终 LaTeX appendix；codex2 需要摘取/转写。

## 8. For codex2

### 8.1 应进入正文的表/图

- 正文主表：`Paper/tables/progressive_main_table.tex`
- 正文机制图：`Paper/figures/progressive_stage_mechanism.png` 或 `Paper/figures/progressive_stage_mechanism.pdf`
- 可选正文/appendix 机制表：`Paper/tables/progressive_stage_mechanism_table.tex`

### 8.2 应进入 appendix 的表

- `Paper/tables/appendix_h2_boundary_table.tex`
- `Paper/tables/appendix_dual_boundary_table.tex`
- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`

### 8.3 必须保留的 caveat

- 主表 task/safety/gate 和 runtime 来自不同聚合口径。
- `threshold_only_progressive` 是 mixed but useful improvement，不是全面支配。
- `safeearly_progressive` 是 late-stage stronger-layer 消融，不是更强成功版本。
- matched analysis 只支持“不宜简单归因于 gate more / compute more”，不是完整 frontier proof。
- H2 / dual / exact `A_hard` 都是 appendix / boundary / theory-support 材料。

### 8.4 不能写的 claim

- 不能写 `threshold_only_progressive` dominates `non_progressive`。
- 不能写 progressive shielding universally improves all safety metrics。
- 不能写 `safeearly_progressive` is the best learned policy。
- 不能写 H2 is stably better than H1。
- 不能写 dual scheduling is a second mature main innovation。
- 不能写 online system solves exact `A_hard` every step。
- 不能写 matched analysis fully eliminates gate / compute confounds。

### 8.5 投稿前技术提醒

- LaTeX preamble 需要 `\usepackage{booktabs}`。
- 如果 stage 图用于双栏论文，优先考虑 full-width figure 或缩短 x-axis labels。
- Cross-reference 建议：
  - `tab:progressive-main`
  - `tab:progressive-stage-mechanism`
  - `tab:appendix-h2-boundary`
  - `tab:appendix-dual-boundary`
  - `tab:appendix-exact-hard`

## 9. 是否需要新实验

不需要。

当前材料足以支持 progressive shielding / conservativeness curriculum 的审慎主线。新实验只在未来决定写更强 causal/frontier claim 时才有必要；当前投稿收敛阶段不建议启动。
