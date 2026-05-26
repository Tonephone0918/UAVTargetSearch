# Appendix Evidence Note

This note collects appendix-ready evidence for the progressive shielding / conservativeness curriculum paper. It is not main-text prose and should not be used to upgrade H=2, dual scheduling, or exact/projected `A_hard` diagnostics into the main contribution.

## Appendix A. H2 Boundary Results

### Citable Result Files

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

### Recommended Table

- `Paper/tables/appendix_h2_boundary_table.tex`

### What Can Be Written

- H=2 is a runtime stronger-layer candidate.
- H=2 can reduce recursive dead-end rates and some safety indicators in fixed-checkpoint or cross-evaluation settings.
- The cross-evaluation pattern suggests that the strongest H=2 signal comes from runtime filtering, not from a uniformly better H=2-trained learned policy.

### What Should Not Be Overstated

- Do not claim that H=2 is stably better than H=1.
- Do not move H=2 into the main result table as a primary success.
- Do not mix H2 boundary evidence with the progressive main formal comparison as if they had the same evidential role.

### Suggested Wording

> H=2 provides a useful runtime stronger-layer candidate, but the current evidence does not establish it as a better learned-policy training regime.

## Appendix B. Dual Scheduling Boundary Results

### Citable Result Files

- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/summary_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_seed_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_checkpoint_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_checkpoint_render_metrics.csv`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/compare_page.json`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/compare_page.html`

### Recommended Table

- `Paper/tables/appendix_dual_boundary_table.tex`

### What Can Be Written

- Dual scheduling changes runtime behavior.
- In the current evidence, dual scheduling reduces shield time relative to threshold-only progressive.
- Its main safety metrics do not stably improve over threshold-only progressive.

### What Should Not Be Overstated

- Do not present dual scheduling as a second mature method contribution.
- Do not claim that dual scheduling stably outperforms threshold-only progressive.
- Do not use the dual table as a main success table.

### Suggested Wording

> Dual scheduling reduces some runtime cost, but it does not improve the main safety metrics over threshold-only progressive in the current evidence.

## Appendix C. Exact/Projected `A_hard` Diagnostics

### Citable Result Files

- `runs/model_compare_exact_hard_solver_fast2x1/summary_metrics.csv`
- `runs/model_compare_exact_hard_solver_fast2x1/per_seed_metrics.csv`
- `runs/model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`
- `runs/model_compare_exact_hard_solver_diag_medium2x2/per_seed_metrics.csv`

### Recommended Table

- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`

### What Can Be Written

- The exact/projected `A_hard` view is a semantic reference for diagnosing allowed-action construction.
- Sequential `A_hard` construction is an online engineering approximation.
- Dead-end cases can be separated into true dead-ends and approximation-induced dead-ends.
- Exact/rescue diagnostics support the method semantics, but they are not the progressive main result.

### What Should Not Be Overstated

- Do not claim that the online main path performs exact solving at every step.
- Do not turn exact solver diagnostics into the main empirical contribution.
- Do not use exact/projected `A_hard` diagnostics to replace the progressive curriculum comparison.

### Suggested Wording

> The exact/projected `A_hard` view serves as a semantic reference for diagnosing approximation-induced dead-ends, while the online shield remains an engineered approximation.

## Overall Boundary

These appendix materials support the broader boundary claim that stronger runtime filtering and better learned policy are not monotonically equivalent. The main paper should still center on the progressive shielding / conservativeness curriculum comparison among `non_progressive`, `threshold_only_progressive`, and `safeearly_progressive`.

No new experiment is required for the current draft.
