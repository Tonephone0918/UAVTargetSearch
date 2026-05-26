# Results / Discussion 骨架 v1（中文）

## 1. 主结果：progressive 主线比较

正文主结果应围绕 `non_progressive`、`threshold_only_progressive` 和 `safeearly_progressive` 三个设置展开。当前任务、安全和 gate 指标来自 `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`，formal compare 口径为 `3 training seeds x 5 eval seeds x 5 episodes`。runtime 指标来自 `runs/progressive_mechanism_20260428/summary_metrics.csv` 的 re-aggregated mechanism summary，用于避免旧 profiling 聚合口径差异。

| model | search_rate | coverage_ratio | collision_count | guarantee_broken_rate | dead_end_rec_rate | recursive_gate_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `non_progressive` | 0.9973 | 0.9989 | 90.79 | 0.3433 | 0.4640 | 0.2684 | 197.84 | 167.40 |
| `threshold_only_progressive` | 0.9987 | 0.9979 | 92.57 | 0.3324 | 0.4437 | 0.2760 | 238.13 | 202.96 |
| `safeearly_progressive` | 1.0000 | 0.9984 | 94.73 | 0.3543 | 0.4670 | 0.2657 | 192.30 | 162.81 |

在这一口径下，`threshold_only_progressive` 是当前最稳的主正结果候选，但其收益必须写成 mixed improvement，而不是全面支配。

与 `non_progressive` 相比，`threshold_only_progressive` 的 `search_rate` 基本持平并略高，约为 `0.9987` 对 `0.9973`；`guarantee_broken_rate` 更低，约为 `0.332` 对 `0.343`；`dead_end_rec_rate` 也更低，约为 `0.444` 对 `0.464`。这些结果支持一个保守结论：在 hard-safe 始终保留的前提下，threshold curriculum 可以在部分安全与 future-feasibility 指标上带来相对稳定的改善。

但同一组结果也显示，该分支不应被写成全面胜利。`threshold_only_progressive` 的 `collision_count` 并未低于 `non_progressive`，约为 `92.57` 对 `90.79`；runtime 也更高，re-aggregated mechanism summary 中 `perf_shield_time_ms` 约为 `238.13ms` 对 `197.84ms`。因此，更稳妥的写法是：threshold curriculum 改善了部分安全/可行性指标，但没有同时支配任务表现、碰撞数和在线开销。

`safeearly_progressive` 当前更适合作为主线对照或消融。它在 formal compare 中达到 `search_rate = 1.0`，但 `collision_count`、`guarantee_broken_rate` 与 `dead_end_rec_rate` 均未稳定优于 `threshold_only_progressive`。由于该分支与 `threshold_only_progressive` 的主要差异在于 late stage 切入 `H=2/eta=0.55`，当前结果更自然地说明：late-stage stronger layer 并不会自动转化为更优 learned policy。

## 2. 机制分析：threshold curriculum 不只是简单 gate more

当前证据不宜把 `threshold_only_progressive` 的收益简单归因于“递归检查开得更多”。从 aggregate 指标看，`threshold_only_progressive` 的 `recursive_gate_rate` 相对 `non_progressive` 只有小幅变化，约为 `0.276` 对 `0.268`；而 `safeearly_progressive` 的 gate rate 约为 `0.266`，并没有因此带来更好的主安全指标。stage-level 统计进一步显示，early 阶段的 `safe` 模式几乎不触发递归检查，mid/late 阶段才进入 `recursive` 模式。

图表资产可引用 `Paper/figures/progressive_stage_mechanism.png`。该图来自 `runs/progressive_mechanism_20260428/stage_metrics.csv` 中 `row_type=aggregate` 且 `split=eval` 的 stage-level 聚合行，展示 `recursive_gate_rate`、`dead_end_rec_rate` 与 runtime 指标。推荐图注：Progressive conservativeness curriculum 的 stage-level 机制统计。early 阶段停留在 hard-safe / safe 层，`threshold_only_progressive` 在 mid/late 阶段切入 `A_rec`；`safeearly_progressive` 在 late 阶段启用 `H=2` stronger layer，但这种 runtime filtering pattern 没有转化为 uniformly better final learned policy。

| model | stage | shield mode | horizon | threshold | recursive_gate_rate | dead_end_rec_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `non_progressive` | fixed | recursive | 1 | 0.35 | 0.2449 | 0.4549 | 175.87 | 147.49 |
| `threshold_only_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 31.41 | 0.00 |
| `threshold_only_progressive` | mid | recursive | 1 | 0.35 | 0.2477 | 0.4495 | 177.11 | 149.05 |
| `threshold_only_progressive` | late | recursive | 1 | 0.35 | 0.2473 | 0.4494 | 179.65 | 148.72 |
| `safeearly_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 30.91 | 0.00 |
| `safeearly_progressive` | mid | recursive | 1 | 0.35 | 0.2457 | 0.4397 | 178.46 | 150.32 |
| `safeearly_progressive` | late | recursive | 2 | 0.55 | 0.0507 | 0.1441 | 97.22 | 67.18 |

因此，当前可保守表述为：threshold curriculum 的作用不是单纯增加 gate 次数，而是改变训练过程中 stronger layer 介入的时机和分布。它让训练早期主要在 `A_hard` 底座上学习，随后再逐步引入 `A_rec` 约束。已有 matched analysis 也只支持审慎说法：当前证据不支持“收益仅由 gate more / compute more 解释”，但还不能写成已经完全消除了 gate-rate 或 compute-budget confound，因为这还不是完整 frontier sweep。

## 3. 机制分析：stronger filtering 与 learned policy mismatch

本文结果中最值得讨论的现象是，stronger runtime safety filtering 与 better learned policy improvement 并不天然等价。这个结论不是对 H2 或 dual 的否定，而是对其当前证据边界的更准确定位。

从理论-证据对应关系看，主表回答的是 progressive conservativeness curriculum 的 learned-policy 结果；stage-level 图表回答的是 early/mid/late 阶段 stronger layer 是否按预期介入；matched analysis 回答的是 threshold-only 的收益是否能被简单归因于 gate more 或 compute more；H2 和 dual 边界结果回答的是更强或更复杂的 runtime filtering 是否自然带来更优 learned policy；exact/projected `A_hard` 诊断则支撑 dead-end 语义和顺序近似误差讨论。上述证据共同支撑 filtering-learning mismatch，但不构成单一因果机制的完整证明。

H2 的边界结果显示，更强的小视界检查可以改善部分 future-feasibility 指标，但训练闭环并未稳定吸收这种优势。在 `final_formal_h2_vs_h1_multiseed3x3` 中，`recursive_risk_rescue_h2_eta055` 相比 `recursive_risk_rescue_h1` 具有更低的 `guarantee_broken_rate` 和 `dead_end_rec_rate`，同时 runtime 也更低；但它的 `search_rate` 更低，`collision_count` 更高。因此，当前不能写成“H2 已经优于 H1”。更可防守的说法是：H2 作为 runtime stronger layer 有候选价值，但 learned checkpoint 层面的结果仍是 mixed。主表中的 `safeearly_progressive` 也支持这一边界定位：late-stage 切入 `H=2` 没有稳定优于 threshold-only。

dual scheduling 进一步支持这一谨慎结论。在 `formal_compare_with_dual_multiseed5x5` 中，`threshold_only_dual_progressive` 的 `perf_shield_time_ms` 低于 `threshold_only_progressive`，但 `collision_count`、`guarantee_broken_rate` 和 `dead_end_rec_rate` 均更差。也就是说，更复杂的运行时阈值调度可以改变计算开销，却没有稳定带来更好的主安全指标。当前更适合把 dual 写成 discussion / future work，而不是第二条主成功结果。

exact/projected `A_hard` 诊断则应作为理论底座和 appendix 支撑材料。它的作用是定义 grounded `A_hard` 参照语义，并帮助区分 true dead-end 与 approximation-induced dead-end；但它不应替代 progressive conservativeness curriculum 成为正文主线。

## 4. 边界结果：H2 的当前位置

`H=2` 在方法上是 `A_H^{viable}` 的自然扩展。它把递归可行性从一步检查推广到小视界 future-feasibility 检查，因此从机制上具有合理性。当前结果也显示，H2 并非没有价值：它在部分设置中能降低 guarantee-broken 或 dead-end 相关指标，并减少某些递归检查开销。

然而，正文必须明确 H2 当前是边界结果。它还没有在 learned policy 层面稳定优于 H1，也不能替代 `threshold_only_progressive` 成为主线正结果。当前可写为：H2 揭示了 stronger layer 的潜力和风险，也说明更强 runtime filter 需要与训练分布、risk threshold 和 actor 适应过程共同调节。在当前版本中，它应放在 results/discussion 后半部分或 appendix。

## 5. 边界结果：dual 的当前位置

dual scheduling 的目标是根据运行时风险状态调整阈值，从而在保守性和开销之间实现更细粒度折中。这个方向在方法上是自然的，但当前结果不支持把它提升为主方法贡献。

目前 dual 的主要信号是 runtime 下降，而不是主安全指标提升。相对 `threshold_only_progressive`，`threshold_only_dual_progressive` 的 `perf_shield_time_ms` 更低，但 `collision_count`、`guarantee_broken_rate` 和 `dead_end_rec_rate` 更高。因此，dual 当前最稳妥的定位是 appendix 或 discussion：它说明更复杂调度不一定更好，也为后续研究提供一个需要重新设计和校准的方向。

## 6. 局限性与不能写太满的地方

当前初稿需要主动承认若干局限。第一，`threshold_only_progressive` 是当前主正结果候选，但它不是全面支配；它改善的是部分安全/可行性指标，而不是所有指标。第二，`safeearly_progressive`、H2 和 dual 当前都是 mixed / boundary，不能写成稳定成功。第三，runtime 指标采用 re-aggregated mechanism summary 口径，正文需要明确任务/安全/gate 指标与 runtime 指标分别引用的来源。第四，`episode_return` 由于 reward normalization 口径差异，不适合作为跨全部目录的统一主指标。

此外，本文关于 filtering-learning mismatch 的机制解释仍需要后续完整 frontier sweep 进一步支撑。当前可保守写成：已有结果提示 stronger runtime safety filtering 与 better learned policy improvement 之间不存在简单单调关系；已有 matched analysis 不支持收益仅由 gate more / compute more 解释，但还不能写成完全消除了这些 confound。exact/projected `A_hard` 诊断目前主要作为理论参照和 appendix 支撑材料，用于解释 sequential 近似、rescue 边界纠偏，以及 true dead-end 与 approximation-induced dead-end 的区别；它不是主环境大规模 exact proof，也不能写成在线主路径每步依赖 exact solver。
