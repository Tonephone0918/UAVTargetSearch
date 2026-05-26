# Claim Evidence Map

更新时间：`2026-05-21`

## Claim 1: Progressive conservativeness curriculum has a mixed but useful benefit

状态：`Supported`

证据：

- `codex1_workspace/progressive_final_main_table.csv`
- `Paper/tables/progressive_main_table.tex`
- `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
- `runs/progressive_mechanism_20260428/summary_metrics.csv`

允许表述：

- `threshold_only_progressive` improves selected safety/feasibility indicators.
- It reduces `guarantee_broken_rate` and `dead_end_rec_rate` relative to `non_progressive`.
- It maintains comparable search performance.
- The gain is mixed because collision and runtime do not improve.

禁止表述：

- `threshold_only_progressive` dominates `non_progressive`.
- Progressive shielding universally improves safety.
- It is a clean SOTA-style win on all metrics.

## Claim 2: Progressive scheduling changes when stronger layers intervene

状态：`Supported`

证据：

- `codex1_workspace/progressive_stage_mechanism_table.md`
- `Paper/tables/progressive_stage_mechanism_table.tex`
- `Paper/figures/progressive_stage_mechanism.png`
- `Paper/figures/progressive_stage_mechanism.pdf`
- `runs/progressive_mechanism_20260428/stage_metrics.csv`

允许表述：

- early stage mainly stays at hard-safe / safe layer.
- threshold-only introduces recursive filtering in mid/late stages.
- safeearly switches to H2 in late stage, but this is a stronger-layer ablation.

禁止表述：

- safeearly is the best final learned policy.
- H2 stage-level metric improvements prove H2 training superiority.
- The stage plot alone proves a single causal mechanism.

## Claim 3: The benefit is not simply gate more or compute more

状态：`Partially supported`

证据：

- `codex1_workspace/progressive_matched_analysis_note.md`
- `runs/progressive_mechanism_20260428/matched_analysis_summary.csv`

允许表述：

- Existing matched evidence does not support reducing the effect to gate more or compute more.
- The safer explanation is that the curriculum changes the stage distribution and timing of stronger-layer intervention.

禁止表述：

- All gate/compute confounds are eliminated.
- A complete frontier proof has been established.
- The exact causal mechanism is fully identified.

## Claim 4: Stronger runtime filtering is not monotonically equivalent to better learned policy

状态：`Supported as boundary/discussion`

证据：

- `codex1_workspace/progressive_appendix_evidence_index.md`
- `Paper/appendix_evidence_note.md`
- `Paper/tables/appendix_h2_boundary_table.tex`
- `Paper/tables/appendix_dual_boundary_table.tex`

允许表述：

- H2 and dual results show that stronger runtime filtering or altered runtime scheduling does not automatically yield a better learned policy.
- H2 is a useful runtime stronger-layer candidate.
- dual reduces some runtime cost but does not improve the main safety metrics over threshold-only in current evidence.

禁止表述：

- H2 is stable better than H1.
- dual is a second mature main innovation.
- boundary evidence should replace the progressive main table.

## Claim 5: Exact/projected `A_hard` is a semantic reference, not the online main path

状态：`Supported`

证据：

- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`
- `Paper/appendix_evidence_note.md`
- `runs/model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`
- `runs/model_compare_exact_hard_solver_fast2x1/summary_metrics.csv`

允许表述：

- exact/projected `A_hard` provides a semantic reference for joint feasibility.
- sequential construction is an engineering approximation.
- exact diagnostics help separate true dead-end from approximation-induced dead-end.

禁止表述：

- the online system solves exact `A_hard` every step.
- exact/projected diagnostics are the main empirical result.
- exact solver comparison should replace the progressive mainline.

## Claim 6: Current manuscript is ready for writing convergence, not new experiments

状态：`Supported`

证据：

- `mainline_agent/agent_reports/20260520_codex1_daily.md`
- `codex1_workspace/submission_asset_checklist.md`
- `Paper/tables/progressive_main_table.tex`
- `Paper/figures/progressive_stage_mechanism.png`
- `Paper/appendix_evidence_note.md`
- `Paper/paper_draft_en_v1.md`
- `Paper/citation_todo_list.md`

允许表述：

- The evidence package is sufficient for the current progressive mainline.
- The next priority is manuscript convergence, citation completion, and cross-reference consistency.
- New experiments are unnecessary unless the paper intentionally adds a stronger causal or frontier claim.

禁止表述：

- The paper is already submission-final.
- Missing citations can remain as placeholders in a submitted manuscript.
- More training should be launched by default.

## 当前最强 Claim

在 always-on `A_hard` 底座上，progressive conservativeness curriculum 可以带来有限但可观察的安全/未来可行性改善；同时，H2/dual 等边界结果说明 stronger runtime filtering 与 better learned policy 并不单调等价。

## 当前最弱 Claim

matched analysis 只能支持“不是简单 gate more / compute more”的审慎说法，不能支持完整因果识别或完整 frontier 证明。

## 当前主要写作风险

- 英文稿已有 tighter draft，但还不是最终 polished submission。
- `Paper/citation_todo_list.md` 中的真实引用缺口是投稿前硬问题。
- 如果正文把 H2、dual、exact `A_hard` 写得过重，会冲淡 progressive conservativeness curriculum 主线。
