# Daily Report: codex1

Date: 2026-05-20
Role: analysis / experiment-materials
Session Goal: Consolidate existing progressive shielding evidence into curator-ready tables, figures, appendix materials, and a daily handoff without running new training.

## 1. Files Read

- `mainline_agent/skill/daily-report/SKILL.md`: read to follow the required daily-report template.
- `codex1.md`: read updated codex1 work orders and constraints.
- `codex1_workspace/progressive_final_main_table.csv`: used as the source for the submission main table.
- `codex1_workspace/progressive_stage_mechanism_table.md`: used as the source/check for stage-level mechanism table values.
- `codex1_workspace/progressive_matched_analysis_note.md`: used for matched gate-rate / compute-budget evidence boundaries.
- `codex1_workspace/progressive_appendix_evidence_index.md`: used for H2, dual, and exact/projected `A_hard` appendix evidence.
- `codex1_workspace/progressive_mainline_assets.md`: read and updated as the consolidated asset index.
- `runs/progressive_mechanism_20260428/stage_metrics.csv`: used by plotting scripts to generate stage-level mechanism figures.
- `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`: source of task/safety/gate metrics already consolidated into the main table.
- `runs/progressive_mechanism_20260428/summary_metrics.csv`: source of re-aggregated runtime metrics already consolidated into the main table.
- `runs/progressive_mechanism_20260428/matched_analysis_summary.csv`: source for matched analysis note.
- `runs/final_formal_h2_vs_h1_multiseed3x3/summary_metrics.csv`: source for H2 boundary evidence.
- `runs/h1_h2_cross_eval_multiseed3x3/summary_metrics.csv`: source for H1/H2 checkpoint-shield cross-eval boundary evidence.
- `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/summary_metrics.csv`: source for dual boundary evidence.
- `runs/model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`: source for exact/projected `A_hard` diagnostic evidence.

## 2. Experiments Or Commands Run

- No new training experiments were run.
- `./.venv/bin/python scripts/plot_progressive_stage_mechanism.py`: generated internal stage-level mechanism PNG/table in `codex1_workspace/`.
- `./.venv/bin/python scripts/plot_progressive_stage_mechanism_paper.py`: generated paper-style stage-level mechanism PNG in `Paper/figures/`.
- `convert Paper/figures/progressive_stage_mechanism.png Paper/figures/progressive_stage_mechanism.pdf`: attempted PDF conversion; failed due to ImageMagick PDF security policy.
- `./.venv/bin/python - ... PIL PDF export`: successfully generated `Paper/figures/progressive_stage_mechanism.pdf`.
- CSV/header inspection commands: used to verify source fields and extracted metrics; no data generation beyond tables/notes.

## 3. Results Added Or Modified

- `codex1_workspace/progressive_final_main_table.csv`: added machine-readable merged main table.
- `codex1_workspace/progressive_stage_mechanism.png`: added internal stage-level mechanism figure.
- `codex1_workspace/progressive_stage_mechanism_table.md`: added internal stage-level mechanism table.
- `codex1_workspace/progressive_matched_analysis_note.md`: added matched analysis evidence-boundary note.
- `codex1_workspace/progressive_appendix_evidence_index.md`: added appendix evidence index for H2, dual, exact/projected `A_hard`.
- `codex1_workspace/progressive_mainline_assets.md`: updated as consolidated asset index.
- `scripts/plot_progressive_stage_mechanism.py`: added internal stage mechanism plotting script.
- `scripts/plot_progressive_stage_mechanism_paper.py`: added paper-style stage mechanism plotting script.
- `Paper/tables/progressive_main_table.tex`: added submission-ready main LaTeX table.
- `Paper/tables/progressive_stage_mechanism_table.tex`: added submission-ready stage-level LaTeX table.
- `Paper/tables/appendix_h2_boundary_table.tex`: added appendix H2 boundary LaTeX table.
- `Paper/tables/appendix_dual_boundary_table.tex`: added appendix dual boundary LaTeX table.
- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`: added appendix exact/projected `A_hard` diagnostic LaTeX table.
- `Paper/figures/progressive_stage_mechanism.png`: added paper-style stage mechanism PNG.
- `Paper/figures/progressive_stage_mechanism.pdf`: added paper-style stage mechanism PDF.
- `Paper/figures/progressive_stage_mechanism_caption.md`: added bilingual cautious figure captions.
- `Paper/appendix_evidence_note.md`: added appendix-ready evidence note.
- `mainline_agent/agent_reports/20260520_codex1_daily.md`: added this daily report.

## 4. Evidence Supporting Mainline

- Final main table supports the progressive conservativeness curriculum comparison among `non_progressive`, `threshold_only_progressive`, and `safeearly_progressive`.
  Evidence: `codex1_workspace/progressive_final_main_table.csv`, `Paper/tables/progressive_main_table.tex`
- `threshold_only_progressive` is the strongest main positive candidate, with lower `guarantee_broken_rate` and `dead_end_rec_rate` than `non_progressive`, while remaining a mixed improvement rather than a full win.
  Evidence: `Paper/tables/progressive_main_table.tex`
- Stage-level mechanism assets support the interpretation that early stages stay at hard-safe / safe behavior, threshold-only enters recursive feasible filtering in mid/late stages, and safeearly enters H=2 in late stage.
  Evidence: `Paper/figures/progressive_stage_mechanism.png`, `Paper/tables/progressive_stage_mechanism_table.tex`
- Matched analysis supports the cautious claim that threshold-only gains are not simply explained by gate more or compute more.
  Evidence: `codex1_workspace/progressive_matched_analysis_note.md`

## 5. Appendix / Boundary / Negative Evidence

- H2 results show a runtime stronger-layer candidate but not a stable learned-policy main success.
  Placement: appendix / boundary
  Evidence: `Paper/tables/appendix_h2_boundary_table.tex`, `Paper/appendix_evidence_note.md`
- Dual scheduling changes runtime and lowers shield time in the boundary comparison, but main safety metrics do not beat threshold-only progressive.
  Placement: appendix / boundary / negative
  Evidence: `Paper/tables/appendix_dual_boundary_table.tex`, `Paper/appendix_evidence_note.md`
- Exact/projected `A_hard` diagnostics support theory/diagnosis of sequential approximation, false empty/nonempty behavior, and approximation-induced dead-end.
  Placement: appendix / theory-support
  Evidence: `Paper/tables/appendix_exact_hard_diagnostic_table.tex`, `Paper/appendix_evidence_note.md`
- ImageMagick PDF conversion failure was environmental; PIL PDF export succeeded.
  Placement: internal
  Evidence: `Paper/figures/progressive_stage_mechanism.pdf`

## 6. Claims Still Not Allowed

- Do not claim `threshold_only_progressive` fully dominates `non_progressive`.
- Do not claim `safeearly_progressive` is a stronger successful version.
- Do not claim H2 is stably better than H1 or should become the main result.
- Do not claim dual scheduling is a second mature method contribution.
- Do not claim exact/projected `A_hard` diagnostics are the main empirical result.
- Do not claim matched analysis is a complete frontier proof or fully removes all gate/compute confounds.
- Do not use `episode_return` as the cross-directory main ranking metric.

## 7. Recommended Next Work

- codex2 should update manuscript references to use `Paper/tables/progressive_main_table.tex` and `Paper/figures/progressive_stage_mechanism.*`.
- codex2 should place H2, dual, and exact/projected `A_hard` tables in appendix or boundary discussion only.
- Curator should update `mainline_agent/EXPERIMENT_LEDGER.md`, `CLAIM_EVIDENCE_MAP.md`, and `NEXT_ACTIONS.md` from this report.
- No new experiment is required for the current draft.

## 8. Important Paths

- `codex1_workspace/progressive_mainline_assets.md`
- `codex1_workspace/progressive_final_main_table.csv`
- `codex1_workspace/progressive_stage_mechanism_table.md`
- `codex1_workspace/progressive_matched_analysis_note.md`
- `codex1_workspace/progressive_appendix_evidence_index.md`
- `Paper/tables/progressive_main_table.tex`
- `Paper/tables/progressive_stage_mechanism_table.tex`
- `Paper/tables/appendix_h2_boundary_table.tex`
- `Paper/tables/appendix_dual_boundary_table.tex`
- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`
- `Paper/figures/progressive_stage_mechanism.png`
- `Paper/figures/progressive_stage_mechanism.pdf`
- `Paper/figures/progressive_stage_mechanism_caption.md`
- `Paper/appendix_evidence_note.md`
- `scripts/plot_progressive_stage_mechanism.py`
- `scripts/plot_progressive_stage_mechanism_paper.py`

## 9. Open Risks Or Questions

- LaTeX tables use `booktabs` commands (`\toprule`, `\midrule`, `\bottomrule`); manuscript preamble should include `\usepackage{booktabs}`.
- Main table mixes formal task/safety/gate metrics with re-aggregated runtime metrics; captions already state this, but codex2 should preserve that caveat.
- Appendix evidence supports boundary claims only; manuscript structure must prevent H2/dual/exact diagnostics from crowding the progressive mainline.

Ready for curator update.
