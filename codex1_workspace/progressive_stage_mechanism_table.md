# Progressive Stage Mechanism Table

数据来源：`runs/progressive_mechanism_20260428/stage_metrics.csv`，仅使用 `row_type=aggregate` 且 `split=eval` 的行。

| model | stage | shield mode | horizon | threshold | recursive_gate_rate | dead_end_rec_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `non_progressive` | fixed | recursive | 1 | 0.35 | 0.2449 | 0.4549 | 175.87 | 147.49 |
| `threshold_only_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 31.41 | 0.00 |
| `threshold_only_progressive` | mid | recursive | 1 | 0.35 | 0.2477 | 0.4495 | 177.11 | 149.05 |
| `threshold_only_progressive` | late | recursive | 1 | 0.35 | 0.2473 | 0.4494 | 179.65 | 148.72 |
| `safeearly_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 30.91 | 0.00 |
| `safeearly_progressive` | mid | recursive | 1 | 0.35 | 0.2457 | 0.4397 | 178.46 | 150.32 |
| `safeearly_progressive` | late | recursive | 2 | 0.55 | 0.0507 | 0.1441 | 97.22 | 67.18 |

可支撑表述：early 阶段主要停留在 safe / hard-safe 层，`threshold_only_progressive` 在 mid/late 切入 H=1 recursive layer；`safeearly_progressive` late 切入 H=2，但这不应被写成 final learned policy 稳定更优。

不能写太满：该表说明 stage 机制差异和 runtime/gate 分布，不单独证明 threshold-only 的收益完全来自某个唯一因果机制。
