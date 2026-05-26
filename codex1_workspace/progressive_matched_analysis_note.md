# Progressive Matched Analysis Note

生成时间：2026-05-12

本文档整理 `threshold_only_progressive` 的 matched gate-rate / matched compute-budget 证据边界。当前只做证据整理，不启动新训练。

## 1. 数据来源

主来源：

- `runs/progressive_mechanism_20260428/matched_analysis_summary.csv`

直接依赖目录：

- `runs/progressive_mechanism_20260428/matched_formal_nonprog_eta25_5x5/`
- `runs/progressive_mechanism_20260428/matched_pilot_nonprog_threshold_scan_3x3/`

口径说明：

- formal matched 行使用 `3_training_seeds_x_5_eval_seeds_x_5_episodes`。
- pilot scan 使用 `3_eval_seeds_x_3_episodes`，只能作为探索性阈值扫描，不应和 formal matched 等强度引用。

## 2. Matched Gate-Rate 核心对比

Formal matched gate-rate：

- reference: `threshold_only_progressive`
- candidate: `non_progressive_default_eta035`
- budget: `3_training_seeds_x_5_eval_seeds_x_5_episodes`
- recursive gate-rate gap: `0.0076`

| model | recursive_gate_rate | collision_count | guarantee_broken_rate | dead_end_rec_rate | perf_shield_time_ms |
|---|---:|---:|---:|---:|---:|
| `threshold_only_progressive` | 0.2760 | 92.57 | 0.3324 | 0.4437 | 238.13 |
| `non_progressive_default_eta035` | 0.2684 | 90.79 | 0.3433 | 0.4640 | 197.84 |

解读：

- 两者 gate rate 很接近，说明 `threshold_only_progressive` 的 `guarantee_broken_rate` 和 `dead_end_rec_rate` 改善不能简单写成“只是 gate 开得更多”。
- 但 `threshold_only_progressive` 的 collision 更高，runtime 也更高，所以仍然是 mixed improvement，不是全面胜利。

## 3. Matched Compute-Budget 核心对比

Formal matched compute-budget：

- reference: `threshold_only_progressive`
- candidate: `non_progressive_eta025`
- budget: `3_training_seeds_x_5_eval_seeds_x_5_episodes`
- shield-time gap: `7.14 ms`

| model | perf_shield_time_ms | recursive_gate_rate | collision_count | guarantee_broken_rate | dead_end_rec_rate |
|---|---:|---:|---:|---:|---:|
| `threshold_only_progressive` | 238.13 | 0.2760 | 92.57 | 0.3324 | 0.4437 |
| `non_progressive_eta025` | 230.99 | 0.3336 | 101.47 | 0.3651 | 0.5114 |

解读：

- 在近似 compute-budget 下，`non_progressive_eta025` 的 gate rate 反而更高，但 `collision_count`、`guarantee_broken_rate` 和 `dead_end_rec_rate` 都更差。
- 这支持审慎结论：`threshold_only_progressive` 的收益不应被简化成 compute more 或 gate more。

## 4. Pilot Scan 边界

Pilot scan 文件：

- `runs/progressive_mechanism_20260428/matched_pilot_nonprog_threshold_scan_3x3/summary_metrics.csv`

观察：

- pilot scan 覆盖 `nonprog_eta20/25/30/35/40` 和 `threshold_only_ref`。
- 该口径只有 `3_eval_seeds_x_3_episodes`，且不是最终 formal 5x5 聚合。
- pilot 中不同 eta 的 task/safety/runtime 排序并不单调，说明 matched frontier 仍有探索空间。

写作使用：

- pilot scan 可以放在 appendix 或 internal note 中说明为什么 matched 结论要保守。
- 不建议用 pilot scan 替代 formal matched 结论。

## 5. 当前可以写的审慎结论

可以写：

- 已有 matched 证据不支持把 `threshold_only_progressive` 的收益简单归因于 gate more 或 compute more。
- 在 matched gate-rate 附近，`threshold_only_progressive` 的 `guarantee_broken_rate` 和 `dead_end_rec_rate` 更低。
- 在 matched compute-budget 附近，`non_progressive_eta025` gate rate 更高但安全/可行性指标更差。
- 因此，当前更稳妥的解释是：progressive conservativeness curriculum 改变了 stronger layer 介入的训练阶段和分布，而不仅是增加平均 gate 次数。

## 6. 当前不能写太满的结论

不能写：

- 不能写已经完全消除了所有 gate / compute confound。
- 不能写 `threshold_only_progressive` 全面支配 non-progressive。
- 不能写 matched analysis 已经构成完整 threshold frontier sweep。
- 不能写收益已经被证明完全来自训练期时序分布这一单一因果机制。

## 7. 是否需要补小规模 frontier eval

当前 draft 不需要新实验。

明确结论：

```text
No new experiment is required for the current draft.
```

如果后续想把机制结论写得更强，可以考虑最小补充：

- 只做 eval，不做新训练。
- 对 `non_progressive` 在更多 eta 上补 formal 5x5 frontier。
- 输出位置建议：`runs/progressive_mechanism_20260428/matched_frontier_nonprog_eta_grid_5x5/`
- 服务问题：进一步判断 `threshold_only_progressive` 的优势是否在更完整 gate/compute frontier 上仍成立。

但该补充不是当前 progressive 主线写作的必要条件。
