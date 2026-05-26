# Progressive Appendix Evidence Index

生成时间：2026-05-12

本文档是给 codex2 使用的 appendix 证据包索引。它只整理 H2、dual、exact/projected `A_hard` 的边界与支撑材料，不改写正文主线。

正文主线仍固定为：

- `non_progressive`
- `threshold_only_progressive`
- `safeearly_progressive`

## 1. H2 边界结果

### 1.1 可引用文件

- `runs/final_formal_h2_vs_h1_multiseed3x3/summary_metrics.csv`
- `runs/final_formal_h2_vs_h1_multiseed3x3/per_seed_metrics.csv`
- `runs/h1_h2_cross_eval_multiseed3x3/summary_metrics.csv`
- `runs/h1_h2_cross_eval_multiseed3x3/per_seed_metrics.csv`
- `runs/h1_h2_fixedpoint_compare_stable3x3/summary_metrics.csv`
- `runs/h1_h2_fixedpoint_compare_stable3x3/matched_gate_rate.csv`
- `runs/h1_h2_fixedpoint_compare_stable3x3/matched_compute_budget.csv`
- `runs/h1_h2_fixedpoint_compare_refine3x3/summary_metrics.csv`
- `runs/h1_h2_fixedpoint_compare_refine3x3/matched_gate_rate.csv`
- `runs/h1_h2_fixedpoint_compare_refine3x3/matched_compute_budget.csv`

### 1.2 关键观察

`final_formal_h2_vs_h1_multiseed3x3/summary_metrics.csv`：

| model | search_rate | collision_count | guarantee_broken_rate | dead_end_rec_rate | recursive_gate_rate | perf_shield_time_ms |
|---|---:|---:|---:|---:|---:|---:|
| `recursive_risk_rescue_h1` | 1.0000 | 86.89 | 0.3741 | 0.4444 | 0.2402 | 181.62 |
| `recursive_risk_rescue_h2_eta055` | 0.9778 | 96.11 | 0.3500 | 0.1500 | 0.0489 | 99.94 |

`h1_h2_cross_eval_multiseed3x3/summary_metrics.csv`：

| model | search_rate | collision_count | guarantee_broken_rate | dead_end_rec_rate | perf_shield_time_ms |
|---|---:|---:|---:|---:|---:|
| `h1_ckpt_h1_shield` | 1.0000 | 86.89 | 0.3741 | 0.4444 | 181.62 |
| `h1_ckpt_h2_shield` | 1.0000 | 76.44 | 0.3157 | 0.1287 | 92.19 |
| `h2_ckpt_h1_shield` | 1.0000 | 90.67 | 0.3630 | 0.4824 | 206.22 |
| `h2_ckpt_h2_shield` | 0.9778 | 96.11 | 0.3500 | 0.1500 | 99.94 |

`h1_h2_fixedpoint_compare_refine3x3/matched_compute_budget.csv`：

- H2 reference `recursive_risk_rescue_h2_eta55_ref` vs H1 candidate `recursive_risk_rescue_h1_eta50` has near matched shield time: `90.20 ms` vs `91.18 ms`.
- Under this comparison H2 has lower `collision_count`, `guarantee_broken_rate`, and `dead_end_rec_rate`.

### 1.3 可支撑结论

可以写：

- H2 是 runtime stronger-layer 候选。
- H2 在部分 fixed-checkpoint 或 cross-eval 设置下能降低 recursive dead-end 和部分安全指标。
- H1 checkpoint + H2 shield 的 cross-eval 表现强于 H2 checkpoint + H2 shield，说明收益更像 runtime layer，而不是 H2 训练闭环已经稳定学出更优 policy。

不能写：

- 不能写 H2 已经稳定优于 H1。
- 不能把 H2 升级为正文主成功分支。
- 不能把 H2 边界结果和 progressive 主线 formal compare 混成同一证据强度。

推荐一句话：

```text
H2 is a useful runtime stronger-layer candidate, but the current evidence does not establish it as a better learned-policy training regime.
```

## 2. Dual 边界结果

### 2.1 可引用文件

- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/summary_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_seed_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_checkpoint_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_checkpoint_render_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/compare_page.json`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/compare_page.html`

### 2.2 关键观察

`formal_compare_with_dual_multiseed5x5/summary_metrics.csv`：

| model | search_rate | coverage_ratio | collision_count | guarantee_broken_rate | dead_end_rec_rate | recursive_gate_rate | perf_shield_time_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `threshold_only_progressive` | 0.9987 | 0.9979 | 92.57 | 0.3324 | 0.4437 | 0.2760 | 238.13 |
| `threshold_only_dual_progressive` | 0.9987 | 0.9995 | 108.49 | 0.3864 | 0.4708 | 0.2462 | 181.29 |

### 2.3 可支撑结论

可以写：

- dual scheduling 改变了 runtime 行为，并降低了 `perf_shield_time_ms`。
- 当前 dual 的主安全指标没有稳定打赢 `threshold_only_progressive`。
- dual 更适合作为 discussion / appendix / future work 边界材料。

不能写：

- 不能把 dual 写成第二条成熟主创新。
- 不能写 dual 稳定优于 threshold-only progressive。
- 不能把 dual 放进正文主表作为核心胜利。

推荐一句话：

```text
Dual scheduling reduces some runtime cost, but it does not improve the main safety metrics over threshold-only progressive in the current evidence.
```

## 3. Exact/Projected `A_hard` 诊断

### 3.1 可引用文件

- `runs/model_compare_exact_hard_solver_fast2x1/summary_metrics.csv`
- `runs/model_compare_exact_hard_solver_fast2x1/per_seed_metrics.csv`
- `runs/model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`
- `runs/model_compare_exact_hard_solver_diag_medium2x2/per_seed_metrics.csv`

### 3.2 关键观察

`model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`：

| model | seq_empty_exact_nonempty_rate | seq_nonempty_exact_empty_rate | exact_hard_rescue_count | guarantee_broken_rate | dead_end_rec_rate |
|---|---:|---:|---:|---:|---:|
| `safe_sequential` | 0.1390 | 0.4166 | 0.00 | 0.5604 | 0.0000 |
| `safe_rescue` | 0.2093 | 0.3808 | 28.25 | 0.4292 | 0.0000 |
| `recursive_risk_sequential` | 0.1706 | 0.4186 | 0.00 | 0.5646 | 0.4167 |
| `recursive_risk_rescue` | 0.2606 | 0.3206 | 33.25 | 0.3688 | 0.4479 |

### 3.3 可支撑结论

可以写：

- exact/projected `A_hard` 可作为理论参照。
- sequential `A_hard` 是工程近似，会出现 projected exact 语义下的 false-empty 和 false-nonempty。
- dead-end 诊断可以拆成 true dead-end 与 approximation-induced dead-end。
- rescue / exact diagnostic 有助于解释 `A_hard` 语义，但不是当前 progressive 主结果。

不能写：

- 不能把 exact solver 对比升级成正文主实验。
- 不能写在线主路径每步都依赖 exact solver。
- 不能用 exact/projected `A_hard` 诊断替代 progressive 主线比较。

推荐一句话：

```text
The exact/projected `A_hard` view serves as a semantic reference for diagnosing approximation-induced dead-ends, while the online shield remains an engineered approximation.
```

## 4. 总体使用边界

这些 appendix 证据支持一个共同的边界结论：

```text
Stronger runtime filtering and better learned policy are not monotonically equivalent.
```

但正文主线仍应保持：

```text
progressive shielding / conservativeness curriculum
```

正文主结果仍只围绕：

- `non_progressive`
- `threshold_only_progressive`
- `safeearly_progressive`

不建议新增实验：

```text
No new experiment is required for the current draft.
```
