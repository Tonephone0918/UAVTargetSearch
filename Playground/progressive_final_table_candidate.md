# Progressive Final Table Candidate

生成时间：2026-05-10

本文档给出 progressive conservativeness curriculum 主线的 final main table 草表。比较对象固定为：

- `non_progressive`
- `threshold_only_progressive`
- `safeearly_progressive`

`episode_return` 不进入主表排序。H2/dual/exact `A_hard` 不进入本主表。

## 1. 推荐主表

任务/安全/gate 指标来自：

- `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`

runtime 指标来自：

- `runs/progressive_mechanism_20260428/summary_metrics.csv`

| model | search_rate | coverage_ratio | collision_count | guarantee_broken_rate | dead_end_rec_rate | recursive_gate_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `non_progressive` | 0.9973 | 0.9989 | 90.79 | 0.3433 | 0.4640 | 0.2684 | 197.84 | 167.40 |
| `threshold_only_progressive` | 0.9987 | 0.9979 | 92.57 | 0.3324 | 0.4437 | 0.2760 | 238.13 | 202.96 |
| `safeearly_progressive` | 1.0000 | 0.9984 | 94.73 | 0.3543 | 0.4670 | 0.2657 | 192.30 | 162.81 |

## 2. 每个数字的来源

### `non_progressive`

- `search_rate=0.9973`
  - `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
  - column: `search_rate_mean`
- `coverage_ratio=0.9989`
  - same CSV
  - column: `coverage_ratio_mean`
- `collision_count=90.79`
  - same CSV
  - column: `collision_count_mean`
- `guarantee_broken_rate=0.3433`
  - same CSV
  - column: `guarantee_broken_rate_mean`
- `dead_end_rec_rate=0.4640`
  - same CSV
  - column: `dead_end_rec_rate_mean`
- `recursive_gate_rate=0.2684`
  - same CSV
  - column: `recursive_gate_rate_mean`
- `perf_shield_time_ms=197.84`
  - `runs/progressive_mechanism_20260428/summary_metrics.csv`
  - column: `perf_shield_time_ms_mean`
- `perf_recursive_time_ms=167.40`
  - same runtime CSV
  - column: `perf_recursive_time_ms_mean`

### `threshold_only_progressive`

- `search_rate=0.9987`
  - `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
  - column: `search_rate_mean`
- `coverage_ratio=0.9979`
  - same CSV
  - column: `coverage_ratio_mean`
- `collision_count=92.57`
  - same CSV
  - column: `collision_count_mean`
- `guarantee_broken_rate=0.3324`
  - same CSV
  - column: `guarantee_broken_rate_mean`
- `dead_end_rec_rate=0.4437`
  - same CSV
  - column: `dead_end_rec_rate_mean`
- `recursive_gate_rate=0.2760`
  - same CSV
  - column: `recursive_gate_rate_mean`
- `perf_shield_time_ms=238.13`
  - `runs/progressive_mechanism_20260428/summary_metrics.csv`
  - column: `perf_shield_time_ms_mean`
- `perf_recursive_time_ms=202.96`
  - same runtime CSV
  - column: `perf_recursive_time_ms_mean`

### `safeearly_progressive`

- `search_rate=1.0000`
  - `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
  - column: `search_rate_mean`
- `coverage_ratio=0.9984`
  - same CSV
  - column: `coverage_ratio_mean`
- `collision_count=94.73`
  - same CSV
  - column: `collision_count_mean`
- `guarantee_broken_rate=0.3543`
  - same CSV
  - column: `guarantee_broken_rate_mean`
- `dead_end_rec_rate=0.4670`
  - same CSV
  - column: `dead_end_rec_rate_mean`
- `recursive_gate_rate=0.2657`
  - same CSV
  - column: `recursive_gate_rate_mean`
- `perf_shield_time_ms=192.30`
  - `runs/progressive_mechanism_20260428/summary_metrics.csv`
  - column: `perf_shield_time_ms_mean`
- `perf_recursive_time_ms=162.81`
  - same runtime CSV
  - column: `perf_recursive_time_ms_mean`

## 3. 聚合口径说明

主表采用混合来源，但不是混合实验结论：

- 任务/安全/gate 指标来自 formal compare：
  - `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
  - 口径：`3 training seeds x 5 eval seeds x 5 episodes`
  - 推荐引用字段：`search_rate_mean`、`coverage_ratio_mean`、`collision_count_mean`、`guarantee_broken_rate_mean`、`dead_end_rec_rate_mean`、`recursive_gate_rate_mean`
- runtime 指标来自 re-aggregated mechanism summary：
  - `runs/progressive_mechanism_20260428/summary_metrics.csv`
  - 推荐引用字段：`perf_shield_time_ms_mean`、`perf_recursive_time_ms_mean`
  - 原因：旧 formal compare 中也有 `perf_*`，但 runtime 聚合口径与后续 mechanism summary 不完全一致；正文 runtime 优先使用 re-aggregated 口径更稳。

formal compare 中的 runtime 备查值如下，不建议作为主表默认 runtime：

| model | formal perf_shield_time_ms | formal perf_recursive_time_ms |
|---|---:|---:|
| `non_progressive` | 330.92 | 275.70 |
| `threshold_only_progressive` | 350.55 | 296.11 |
| `safeearly_progressive` | 281.67 | 235.93 |

## 4. 表格解读建议

可以写：

- `threshold_only_progressive` 相比 `non_progressive` 在 `guarantee_broken_rate` 和 `dead_end_rec_rate` 上更低，`search_rate` 基本持平。
- `threshold_only_progressive` 不是全面胜利，因为 `collision_count` 更高，runtime 也更高。
- `safeearly_progressive` 达到最高 `search_rate`，但 `collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate` 均不优于 `threshold_only_progressive`，因此更适合作为 stronger late-stage layer 的消融/对照。

不能写：

- 不能写 threshold-only 全面支配 non-progressive。
- 不能写 safeearly 是更强成功版本。
- 不能用这张表证明 H2 或 dual 是主成功分支。

## 5. Caption 建议

建议 caption：

> Main comparison of progressive conservativeness curricula under an always-on hard-safe shield. Task and safety metrics are aggregated over 3 training seeds, 5 evaluation seeds per checkpoint, and 5 episodes per evaluation seed from the formal comparison. Runtime metrics use the re-aggregated mechanism summary to keep profiling statistics on a consistent runtime aggregation. `threshold_only_progressive` shows mixed but useful improvements in guarantee violation and recursive dead-end rates, while `safeearly_progressive` serves as a late-stage stronger-layer ablation rather than a uniformly better variant.

中文 caption 备选：

> Progressive conservativeness curriculum 的主结果比较。任务与安全指标来自 formal compare 口径，即 3 个训练 seed、每个 checkpoint 5 个 eval seed、每个 eval seed 5 个 episode；runtime 指标来自 re-aggregated mechanism summary，以避免旧 profiling 聚合口径差异。`threshold_only_progressive` 在 guarantee violation 与 recursive dead-end 上呈现有限但稳定的改善，但并未在 collision 或 runtime 上全面支配；`safeearly_progressive` 应解释为 late-stage stronger-layer 消融，而不是更强成功版本。

## 6. 可选 stage-level 附表

stage-level 机制数据来自：

- `runs/progressive_mechanism_20260428/stage_metrics.csv`

| model | stage | shield mode | horizon | threshold | recursive_gate_rate | dead_end_rec_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `non_progressive` | fixed | recursive | 1 | 0.35 | 0.2449 | 0.4549 | 175.87 | 147.49 |
| `threshold_only_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 31.41 | 0.00 |
| `threshold_only_progressive` | mid | recursive | 1 | 0.35 | 0.2477 | 0.4495 | 177.11 | 149.05 |
| `threshold_only_progressive` | late | recursive | 1 | 0.35 | 0.2473 | 0.4494 | 179.65 | 148.72 |
| `safeearly_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 30.91 | 0.00 |
| `safeearly_progressive` | mid | recursive | 1 | 0.35 | 0.2457 | 0.4397 | 178.46 | 150.32 |
| `safeearly_progressive` | late | recursive | 2 | 0.55 | 0.0507 | 0.1441 | 97.22 | 67.18 |

这个附表适合放在 mechanism analysis 或 appendix，而不是替代主表。
