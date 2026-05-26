# Codex1 下一阶段科研材料工作单

生成时间：`2026-05-21`

任务来源：

- `mainline_agent/NEXT_ACTIONS.md`
- `codex1_workspace/submission_asset_checklist.md`
- `Paper/paper_draft_en_v1.md`
- `Paper/citation_todo_list.md`

## 0. 角色定位

你是 `codex1`，负责科研材料、证据资产、图表一致性和投稿前最终 QA。

请注意和 `codex2` 区分：

- `codex1`：维护实验/图表/附录证据资产，检查表格、图、caption、label、来源、正文引用和可写结论。
- `codex2`：推进英文稿、related work、引用补齐和投稿文本 polish。

当前不要启动新训练，不要扩展 eval，不要修改训练代码，不要新增方法分支。当前论文主线固定为：

```text
progressive shielding / conservativeness curriculum
```

正文主比较对象固定为：

1. `non_progressive`
2. `threshold_only_progressive`
3. `safeearly_progressive`

`H=2`、dual scheduling、exact/projected `A_hard` 诊断只作为 boundary / appendix / theory-support 材料。

---

## 1. 当前状态

上一轮你已经完成：

```text
codex1_workspace/submission_asset_checklist.md
```

并确认投稿资产整体可用。codex2 已将英文稿推进为 tighter draft：

```text
Paper/paper_draft_en_v1.md
```

本轮不再生成新的主表或机制图。目标是做最终资产 QA：检查英文稿是否正确使用了现有图表、appendix 和 caveat。

---

## 2. 必读文件

请先阅读：

- `mainline_agent/PROJECT_MAINLINE.md`
- `mainline_agent/EXPERIMENT_LEDGER.md`
- `mainline_agent/CLAIM_EVIDENCE_MAP.md`
- `mainline_agent/NEXT_ACTIONS.md`
- `codex1_workspace/submission_asset_checklist.md`
- `Paper/paper_draft_en_v1.md`
- `Paper/citation_todo_list.md`
- `Paper/references_seed.bib`
- `Paper/figures/progressive_stage_mechanism_caption.md`
- `Paper/appendix_evidence_note.md`

并核查：

- `Paper/tables/progressive_main_table.tex`
- `Paper/tables/progressive_stage_mechanism_table.tex`
- `Paper/tables/appendix_h2_boundary_table.tex`
- `Paper/tables/appendix_dual_boundary_table.tex`
- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`
- `Paper/figures/progressive_stage_mechanism.png`
- `Paper/figures/progressive_stage_mechanism.pdf`

---

## 3. 本轮目标

生成最终资产 QA 文件：

```text
codex1_workspace/final_asset_qa.md
```

该文件需要回答：

1. 英文稿是否正确引用正文主表；
2. 英文稿是否正确引用 stage-level 机制图；
3. 英文稿是否正确安排 stage-level 机制表；
4. H2 / dual / exact hard appendix 表是否只作为 boundary / appendix / theory-support；
5. caption、label、booktabs、数值格式是否仍存在风险；
6. stage 图双栏显示是否可能有风险；
7. 哪些地方需要 codex2 修改正文；
8. 是否仍然不需要新实验。

---

## 4. 具体任务

### 4.1 对照英文稿检查引用

检查 `Paper/paper_draft_en_v1.md` 中是否合理使用：

- `tab:progressive-main`
- `fig:progressive-stage-mechanism`
- `tab:progressive-stage-mechanism`
- `tab:appendix-h2-boundary`
- `tab:appendix-dual-boundary`
- `tab:appendix-exact-hard`

注意：

- Markdown 草稿可以使用路径占位，但最终 LaTeX 需要稳定 label；
- 如果英文稿引用方式不清楚，在 `final_asset_qa.md` 中给 codex2 明确修改建议；
- 不要直接大改英文稿，除非只是修一个明显路径或 label 小错误。

### 4.2 检查证据边界是否被保持

对照 `mainline_agent/CLAIM_EVIDENCE_MAP.md` 检查：

- `threshold_only_progressive` 是否仍是 mixed but useful improvement；
- `safeearly_progressive` 是否仍是 stronger-layer ablation；
- H2 是否仍是 runtime stronger-layer candidate；
- dual 是否仍是 runtime / boundary result；
- exact/projected `A_hard` 是否仍是 semantic reference / diagnostic support；
- matched analysis 是否没有被写成完整 frontier proof。

如果发现过强表述，在 `final_asset_qa.md` 中指出具体段落和推荐改法。

### 4.3 检查投稿技术风险

重点检查并记录：

- `booktabs` 是否需要在最终 LaTeX preamble 中声明；
- stage 图若用于双栏，是否需要 full-width figure 或缩短 x-axis labels；
- `A_hard` 在 caption 和正文中是否需要统一为数学记号；
- 表格 caption 是否保留 task/safety/gate 与 runtime 来源不同的 caveat。

### 4.4 小范围修补

如果发现以下小问题，可以直接修补：

- `.tex` 表格 label/caption 拼写；
- 图注中明显过强或不一致措辞；
- appendix note 中明显边界不清的句子。

不要做：

- 不要重写英文稿；
- 不要重新生成整套图表；
- 不要改实验代码；
- 不要补实验。

---

## 5. 禁止事项

本轮不要做以下事情：

- 不要启动新训练；
- 不要扩展 eval；
- 不要修改 `src/` 下训练或 shield 代码；
- 不要新增 method；
- 不要重写论文正文；
- 不要把 H2、dual、exact `A_hard` 诊断升级为主线；
- 不要把 `threshold_only_progressive` 写成全面胜利；
- 不要把 `safeearly_progressive` 写成更强成功版本；
- 不要把 matched analysis 写成完整 frontier proof。

---

## 6. 最终回复格式

完成后请汇报：

1. 是否生成 `codex1_workspace/final_asset_qa.md`；
2. 英文稿中的表/图/appendix 引用是否一致；
3. 是否修补了任何表格、图注或 appendix note；
4. 是否发现 `booktabs`、caption、label、双栏图等投稿前风险；
5. 给 codex2 的最重要修改建议是什么；
6. 是否确认不需要新实验。

---

## 7. 固定结束要求：日结汇报开关

默认不生成日结。只有当下面开关被手动改为 `true` 时，任务结束才必须生成日结：

```text
DAILY_REPORT_ENABLED=false
```

当 `DAILY_REPORT_ENABLED=true` 时，请使用 `daily-report（日结汇报）` skill，在以下目录生成当天日结：

```text
mainline_agent/agent_reports/
```

文件名规则：

```text
YYYYMMDD_codex1_daily.md
```

日期使用当前系统日期。例如：

```text
20260521_codex1_daily.md
```

日结必须按 `daily-report` skill 模板填写，至少包含：

- 今天读了哪些文件/跑了哪些实验；
- 新增或修改了哪些结果；
- 哪些结果可以支持主线；
- 哪些结果只能作为 appendix / boundary / negative evidence；
- 哪些 claim 仍然不能写；
- 明天建议做什么；
- 重要文件路径。

当 `DAILY_REPORT_ENABLED=false` 时，不需要生成日结，只需按本文件的“最终回复格式”正常汇报。
