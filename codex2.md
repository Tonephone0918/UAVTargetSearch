# Codex2 下一阶段论文写作工作单

生成时间：`2026-05-21`

任务来源：

- `mainline_agent/NEXT_ACTIONS.md`
- `codex1_workspace/submission_asset_checklist.md`
- `Paper/paper_draft_en_v1.md`
- `Paper/citation_todo_list.md`

## 0. 角色定位

你是 `codex2`，负责论文正文、英文稿、related work、引用补齐和投稿文本 polish。

请注意和 `codex1` 区分：

- `codex1`：维护实验/图表/附录证据资产，检查表格、图、caption、label、来源、正文引用和可写结论。
- `codex2`：把论文写顺、写准、写成可投稿版本。

当前不要启动新训练，不要扩展 eval，不要修改训练代码，不要新增方法分支。如果缺图、缺表、缺 appendix 证据，请记录给 codex1，不要自己扩实验。

当前论文主线固定为：

```text
progressive shielding / conservativeness curriculum
```

正文主比较对象固定为：

1. `non_progressive`
2. `threshold_only_progressive`
3. `safeearly_progressive`

当前主结果强度固定为：

```text
threshold_only_progressive = mixed but useful improvement
```

不能写成全面胜利。

---

## 1. 当前状态

你已经将英文稿推进为 tighter draft：

```text
Paper/paper_draft_en_v1.md
```

并生成了保守的引用缺口清单：

```text
Paper/citation_todo_list.md
```

codex1 已生成投稿资产检查清单：

```text
codex1_workspace/submission_asset_checklist.md
```

本轮目标是从 tighter draft 继续推进到 polished submission draft，并处理 citation TODO。

---

## 2. 必读文件

请先阅读：

- `mainline_agent/PROJECT_MAINLINE.md`
- `mainline_agent/EXPERIMENT_LEDGER.md`
- `mainline_agent/CLAIM_EVIDENCE_MAP.md`
- `mainline_agent/NEXT_ACTIONS.md`
- `Paper/paper_draft_en_v1.md`
- `Paper/paper_draft_cn_v2.md`
- `Paper/paper_related_work_cn_v1.md`
- `Paper/citation_todo_list.md`
- `Paper/references_seed.bib`
- `codex1_workspace/submission_asset_checklist.md`

以及投稿资产：

- `Paper/tables/progressive_main_table.tex`
- `Paper/tables/progressive_stage_mechanism_table.tex`
- `Paper/tables/appendix_h2_boundary_table.tex`
- `Paper/tables/appendix_dual_boundary_table.tex`
- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`
- `Paper/figures/progressive_stage_mechanism.png`
- `Paper/figures/progressive_stage_mechanism.pdf`
- `Paper/figures/progressive_stage_mechanism_caption.md`
- `Paper/appendix_evidence_note.md`

如果 codex1 新生成：

```text
codex1_workspace/final_asset_qa.md
```

请优先读取并遵守其中的修改建议。

---

## 3. 本轮目标

本轮目标是投稿前文本 polish：

1. 将 `Paper/paper_draft_en_v1.md` 从 tighter draft 收敛为 polished submission draft；
2. 处理 `Paper/citation_todo_list.md` 中的剩余 citation gap；
3. 统一正文、图表、附录之间的 cross-reference；
4. 保持所有 claim 与 `mainline_agent/CLAIM_EVIDENCE_MAP.md` 一致；
5. 不新增实验，不扩展方法。

---

## 4. 任务 1：Polish 英文稿

目标文件：

```text
Paper/paper_draft_en_v1.md
```

重点处理：

- Abstract：更像投稿摘要，清楚交代问题、方法、结果强度和边界；
- Introduction：增强动机、贡献和 filtering-learning mismatch 主线；
- Method：保持 allowed-action filtering、`A_hard -> A_rec -> A_H^{viable}` 和 exact/projected reference 的清楚边界；
- Results and Discussion：减少草稿痕迹，明确 Table/Figure/Appendix 的引用；
- Limitations：保留 mixed result、runtime source split、matched frontier 不完整、exact solver 非在线主路径等 caveat；
- Conclusion：总结审慎，不写成全面胜利。

要求：

- 使用正式、审慎、可投稿的学术英语；
- 不使用 `first`、`novel`、`groundbreaking` 等高风险措辞；
- 主结果必须写成 `mixed but useful improvement`；
- 明确 progressive 是 conservativeness curriculum，不是 hard-safe off/on warmup；
- 明确 shield 是 allowed-action filtering，不是 planner takeover；
- 明确 H2、dual、exact/projected `A_hard` 是 boundary / appendix / theory-support。

---

## 5. 任务 2：处理 Citation TODO

目标文件：

```text
Paper/citation_todo_list.md
Paper/references_seed.bib
Paper/paper_draft_en_v1.md
Paper/paper_related_work_cn_v1.md
```

要求：

- 优先处理 `Paper/citation_todo_list.md` 中的剩余缺口；
- 只加入能可靠确认 author / title / venue / year / claim 的真实文献；
- 如果不能联网或不能可靠核验，就保留 TODO，不要编造；
- 新增引用必须同步写入 `Paper/references_seed.bib`；
- 在 `Paper/citation_todo_list.md` 中标记哪些 gap 已解决、哪些仍待检索。

特别注意：

- 不要把 dynamic shielding 文献写成已经覆盖本文 exact progressive conservativeness curriculum；
- 不要把内部 H2 / dual / exact `A_hard` 结果写成外部 prior work；
- 不要用不确定 UAV 文献支撑具体 claim。

---

## 6. 任务 3：统一图表和附录引用

请在英文稿中合理引用：

- `Paper/tables/progressive_main_table.tex`：正文主表，label `tab:progressive-main`；
- `Paper/figures/progressive_stage_mechanism.png` 或 `.pdf`：正文机制图，label `fig:progressive-stage-mechanism`；
- `Paper/tables/progressive_stage_mechanism_table.tex`：机制表，label `tab:progressive-stage-mechanism`，正文或 appendix 视篇幅决定；
- `Paper/tables/appendix_h2_boundary_table.tex`：appendix / boundary，label `tab:appendix-h2-boundary`；
- `Paper/tables/appendix_dual_boundary_table.tex`：appendix / boundary，label `tab:appendix-dual-boundary`；
- `Paper/tables/appendix_exact_hard_diagnostic_table.tex`：appendix / theory support，label `tab:appendix-exact-hard`；
- `Paper/appendix_evidence_note.md`：附录证据说明。

要求：

- 主表 caveat 必须保留：task/safety/gate 与 runtime 来源不同；
- stage 图不能被写成 H2 或 safeearly 最终更优；
- appendix 表不能抢正文主线；
- 如果仍是 Markdown 草稿，可以用清晰路径和 label 占位。

---

## 7. 任务 4：最终 Claim 一致性检查

请对照：

```text
mainline_agent/CLAIM_EVIDENCE_MAP.md
codex1_workspace/submission_asset_checklist.md
```

检查英文稿是否存在过强表述，尤其是：

- `threshold_only_progressive` 不能写成全面支配；
- `safeearly_progressive` 不能写成更强成功版本；
- H2 不能写成稳定优于 H1；
- dual 不能写成第二条成熟主创新；
- matched analysis 不能写成完整消除 confound；
- exact/projected `A_hard` 不能写成在线每步 exact solver；
- `episode_return` 不能作为跨目录主排序指标。

如发现过强表述，请直接修正。

---

## 8. 禁止事项

本轮不要做以下事情：

- 不要启动新训练；
- 不要扩展 eval；
- 不要修改训练代码；
- 不要新增方法；
- 不要把 `threshold_only_progressive` 写成全面支配；
- 不要把 `safeearly_progressive` 写成更强成功版本；
- 不要把 H2 写成稳定优于 H1；
- 不要把 dual 写成第二条成熟主创新；
- 不要把 matched analysis 写成完整消除 confound；
- 不要把 `episode_return` 作为跨全部目录主排序指标；
- 不要臆造真实文献引用。

---

## 9. 最终回复格式

完成后请汇报：

1. 英文稿 `Paper/paper_draft_en_v1.md` 做了哪些 polish；
2. `Paper/citation_todo_list.md` 哪些 gap 已解决、哪些仍保留；
3. 是否更新 `Paper/references_seed.bib` 或 `Paper/paper_related_work_cn_v1.md`；
4. 正文引用了哪些主表、图和 appendix 表；
5. 哪些 claim 被修正为更审慎；
6. 是否还需要 codex1 小范围补图表/图注；
7. 是否确认不需要新实验。

---

## 10. 固定结束要求：日结汇报开关

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
YYYYMMDD_codex2_daily.md
```

日期使用当前系统日期。例如：

```text
20260521_codex2_daily.md
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
