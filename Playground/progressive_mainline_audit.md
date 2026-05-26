# Progressive Mainline Audit

生成时间：2026-05-10

本文档只盘点现有结果，不引入新训练。当前论文主线固定为 progressive shielding / conservativeness curriculum，正文主比较对象固定为：

- `non_progressive`
- `threshold_only_progressive`
- `safeearly_progressive`

`H=2`、dual scheduling、exact/projected `A_hard` 只作为边界结果或理论支撑材料。

## 1. 已有实验盘点

### 1.1 已阅读的主线文档

- `Playground/progressive_shield_plan_v1.md`
  - 已明确当前 progressive 语义不是 shield off/on warmup，而是 hard-safe always-on 下的保守性层级调度。
  - 已提醒 `H=2`、dual scheduling 不能升级为正文主线。
- `Playground/risk_modification_log.md`
  - 已记录 risk gate 的语义位置为 post-`A_hard` / pre-`A_rec`。
  - 已显示 risk gate 有工程价值，但判别力不能写太满。
- `Paper/paper_result_notes_cn_v2.md`
  - 已形成当前结果口径：任务/安全指标优先来自 formal compare，runtime 优先来自 re-aggregated mechanism summary。
  - 已明确 `threshold_only_progressive` 只能写 mixed improvement。
- `Paper/paper_outline_cn_v2.md`
  - 已把正文主表限定在三组 progressive 设置。
  - 已把 H2/dual 放到边界结果或 appendix。
- `Paper/paper_draft_cn_v2.md`
  - 初稿已经基本沿着 conservativeness curriculum 写，但仍有若干 TODO 需要最终数据表固化。

### 1.2 指定结果目录盘点

- `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`、`per_checkpoint_metrics.csv`、`per_checkpoint_render_metrics.csv`
  - 用途：progressive 正文主表的任务/安全/gate 指标。
  - 口径：`3 training seeds x 5 eval seeds x 5 episodes`。
- `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5_raw/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`
  - 用途：原始/备查口径，不建议优先作为正文主表来源。
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`、`per_checkpoint_metrics.csv`、`per_checkpoint_render_metrics.csv`、`compare_page.json`、`compare_page.html`
  - 用途：dual 边界结果。
- `runs/progressive_mechanism_20260428/`
  - 存在：`summary_metrics.csv`、`stage_metrics.csv`、`matched_analysis_summary.csv`、`training_curves.csv`、`training_curves_aggregate.csv`、`asset_inventory.md`、`notes.md`、`overview.html`
  - 另有 matched scan 子目录：`matched_pilot_nonprog_threshold_scan_3x3/`、`matched_formal_nonprog_eta25_5x5/` 等。
  - 用途：re-aggregated runtime、stage-level 机制解释、matched gate-rate / compute-budget 分析。
- `runs/final_formal_h2_vs_h1_multiseed3x3/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`
  - 用途：H2 vs H1 边界结果，不能作为 progressive 主线主表。
- `runs/h1_h2_cross_eval_multiseed3x3/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`
  - 用途：H1/H2 checkpoint x shield cross-eval，支持 stronger filtering 与 learned policy 不必然一致。
- `runs/h1_h2_fixedpoint_compare_stable3x3/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`、`matched_gate_rate.csv`、`matched_compute_budget.csv`
  - 用途：H1/H2 matched 边界分析。
- `runs/h1_h2_fixedpoint_compare_refine3x3/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`、`matched_gate_rate.csv`、`matched_compute_budget.csv`
  - 用途：H1/H2 refined matched 边界分析。
- `runs/model_compare_exact_hard_solver_fast2x1/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`
  - 用途：exact/projected `A_hard` 快速诊断。
- `runs/model_compare_exact_hard_solver_diag_medium2x2/`
  - 存在：`summary_metrics.csv`、`per_seed_metrics.csv`
  - 用途：exact/projected `A_hard` 稍稳诊断，含 sequential/exact/rescue 的 false-empty / false-nonempty 统计。

## 2. 主线证据满足度判断

### Q1. 是否已有 final main table 所需数据？

结论：Yes。

证据：

- 任务/安全/gate 指标：`runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
  - 覆盖 `search_rate`、`coverage_ratio`、`collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate`、`recursive_gate_rate`。
- runtime 指标：`runs/progressive_mechanism_20260428/summary_metrics.csv`
  - 覆盖 `perf_shield_time_ms`、`perf_recursive_time_ms`。

注意：

- formal compare 的 `summary_metrics.csv` 中也有 `perf_*` 字段，但当前更推荐正文 runtime 使用 `progressive_mechanism_20260428/summary_metrics.csv` 的 re-aggregated 口径。
- `episode_return` 不应作为跨目录主排序指标。

### Q2. 是否已有 stage-level progressive 机制数据？

结论：Yes。

证据：

- `runs/progressive_mechanism_20260428/stage_metrics.csv`
  - 覆盖 `progressive_stage`、`effective_shield_mode`、`effective_lookahead_horizon`、`effective_risk_threshold`、`recursive_gate_rate`、`dead_end_rec_rate`、`perf_shield_time_ms`、`perf_recursive_time_ms`。

可支撑的机制表述：

- `threshold_only_progressive` early 阶段为 `safe / H=1 / eta=0.9`，`recursive_gate_rate=0`。
- `threshold_only_progressive` mid/late 阶段为 `recursive / H=1 / eta=0.35`。
- `safeearly_progressive` early 阶段同样为 safe，mid 为 `recursive / H=1 / eta=0.35`，late 切入 `recursive / H=2 / eta=0.55`。

### Q3. 是否已有 matched analysis 回答 threshold_only_progressive 的收益是否只是 gate more / compute more？

结论：Partial，但已经足够支持审慎写法。

证据：

- `runs/progressive_mechanism_20260428/matched_analysis_summary.csv`
  - `formal_matched_gate_rate`：`threshold_only_progressive` 对 `non_progressive_default_eta035`，gate-rate gap 约 `0.0076`。
  - `formal_matched_compute_budget`：`threshold_only_progressive` 对 `non_progressive_eta025`，shield-time gap 约 `7.14 ms`。
- `runs/progressive_mechanism_20260428/matched_formal_nonprog_eta25_5x5/summary_metrics.csv`
  - matched compute-budget 的候选 non-progressive eta=0.25 来源。

当前可写：

- `threshold_only_progressive` 的收益不能简单归因于平均 gate rate 更高。
- 在 matched compute-budget 附近，non-progressive eta=0.25 仍比 threshold-only 在 `collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate` 上更差。

当前不能写太满：

- 不能写成已经完整证明“收益完全来自训练期时序分布而非 compute/gate”。
- matched 分析足以做机制支持，但还不是一个完整 frontier sweep。

### Q4. 是否已有足够 H=2 / dual 边界结果支持 stronger runtime filtering 不必然带来 better learned policy？

结论：Yes，作为边界结果足够。

证据：

- H2 fixed checkpoint 边界：`runs/final_formal_h2_vs_h1_multiseed3x3/summary_metrics.csv`
  - `recursive_risk_rescue_h2_eta055` 相比 H1 降低 `guarantee_broken_rate` 与 `dead_end_rec_rate`，但 `search_rate` 更低、`collision_count` 更高。
- H1/H2 matched 边界：`runs/h1_h2_fixedpoint_compare_stable3x3/matched_gate_rate.csv`、`runs/h1_h2_fixedpoint_compare_stable3x3/matched_compute_budget.csv`
  - 支持 H2 优势并非全面支配。
- H1/H2 refined 边界：`runs/h1_h2_fixedpoint_compare_refine3x3/summary_metrics.csv`、`matched_gate_rate.csv`、`matched_compute_budget.csv`
  - 支持 H2 是 runtime stronger-layer 候选，但不是 learned policy 主成功。
- Cross-eval：`runs/h1_h2_cross_eval_multiseed3x3/summary_metrics.csv`
  - 支持 checkpoint 与 runtime shield 的收益不完全同向。
- Dual 边界：`runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/summary_metrics.csv`
  - `threshold_only_dual_progressive` runtime 更低，但 `collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate` 更差。

写作定位：

- H2/dual 可用来支持 filtering-learning mismatch。
- 不应把 H2/dual 写成正文主成功分支。

### Q5. 是否已有足够 A_hard exact/projected 诊断作为理论底座支撑？

结论：Yes，作为支撑材料足够；正文主结果不宜过度依赖。

证据：

- `runs/model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`
  - 对 sequential / exact / rescue 版本给出 `exact_hard_query_count`、`exact_hard_false_empty_count`、`exact_hard_false_empty_rate`、`seq_empty_exact_nonempty_rate`、`seq_nonempty_exact_empty_rate`。
  - 例如 `safe_sequential` 诊断中 `seq_empty_exact_nonempty_rate≈0.139`、`seq_nonempty_exact_empty_rate≈0.417`，显示顺序近似确实存在 projected exact 语义下的错配。
  - `recursive_risk_sequential` 诊断中 `seq_empty_exact_nonempty_rate≈0.171`、`seq_nonempty_exact_empty_rate≈0.419`。
- `runs/model_compare_exact_hard_solver_fast2x1/summary_metrics.csv`
  - 快速小样本结果与 medium 诊断方向一致。

写作定位：

- 可支撑 exact/projected `A_hard` view、true dead-end vs approximation-induced dead-end 的理论底座。
- 不应把 exact solver 对比升级为正文主实验。

## 3. 当前论文主线可写到什么强度

可以较稳地写：

- 本文采用 hard-safe always-on 的 layered allowed-action shield，progressive 调节的是 stronger layer 介入强度，而不是安全底线开关。
- `threshold_only_progressive` 是当前最稳主正结果候选，相比 `non_progressive` 在 `guarantee_broken_rate` 与 `dead_end_rec_rate` 上更好，`search_rate` 基本持平。
- `threshold_only_progressive` 的改善是 mixed improvement，并伴随一定 runtime 代价。
- `safeearly_progressive` 是 late-stage stronger layer 的消融/对照，不是更强成功版本。
- H2/dual 结果支持 stronger runtime filtering 与 better learned policy 之间不存在简单单调关系。
- exact/projected `A_hard` 诊断可以支撑方法语义和 dead-end 诊断，不应抢正文主线。

不能写太满：

- 不能写 `threshold_only_progressive` 全面优于 `non_progressive`。
- 不能写 `safeearly_progressive` 是更强 progressive 成功版本。
- 不能写 H2 已稳定优于 H1。
- 不能写 dual scheduling 已形成第二条成熟主创新。
- 不能把 episode return 作为跨全部目录统一主排序指标。
- 不能把 matched analysis 写成完整消除了 compute/gate confound；目前更适合写“已有证据不支持仅由 gate more / compute more 解释”。

## 4. 仍缺什么

当前没有阻塞正文主表的硬缺口。

较小缺口：

- 若要把 matched 机制结论写得更强，需要更系统的 threshold frontier sweep；但这不是当前主线写作的必要条件。
- 若要让 stage-level 图更漂亮，可能需要从 `stage_metrics.csv` 生成一张 early/mid/late 小图；这属于整理产物，不是新训练。
- 若审稿压力要求 runtime 完全同口径，可补一个只读重聚合脚本，把 formal compare 的 task/safety 与 mechanism runtime 明确合并到单一 table CSV；不需要重训。

## 5. 最小下一步建议

1. 优先使用 `Playground/progressive_final_table_candidate.md` 的表作为正文主表草稿。
2. 从 `runs/progressive_mechanism_20260428/stage_metrics.csv` 生成一张 stage-level mechanism 图或表。
3. 暂不启动新训练。
4. 如确需补实验，只补 matched threshold frontier 的小规模 eval，不补大规模训练；H2/dual/exact 只作为边界或 appendix。
