# Next Actions

更新时间：`2026-05-21`

## 总体判断

当前实验资产已经足够支撑 progressive shielding / conservativeness curriculum 主线。codex1 已完成投稿资产检查清单，codex2 已将英文稿推进为 tighter draft 并生成 citation TODO list。下一步不补训练，转入投稿前最终收敛。

## Codex1 下一步

角色：科研材料与证据资产最终 QA。

优先任务：

1. 基于 `codex1_workspace/submission_asset_checklist.md` 和 `Paper/paper_draft_en_v1.md`，检查英文稿中的表/图/appendix 引用是否与资产 label、caption、使用边界一致。
2. 生成 `codex1_workspace/final_asset_qa.md`，列出每个资产是否被正确引用、是否存在 label/caption/booktabs/双栏图风险、是否需要 codex2 修改正文。
3. 若发现小范围表格、图注或 appendix note 问题，可以修补；不要重写整套资产。
4. 若 codex2 后续发现正文引用缺口，只做小范围表格/图注支持，不新增实验。

不要做：

- 不启动新训练。
- 不修改 `src/` 方法代码。
- 不新增 H2/dual/exact `A_hard` 主线。
- 不重复跑已有 formal compare。

## Codex2 下一步

角色：论文写作与投稿稿推进。

优先任务：

1. 将 `Paper/paper_draft_en_v1.md` 从 tighter draft 继续收敛为 polished submission draft，重点压实 abstract、introduction、results/discussion、limitations 和 conclusion。
2. 处理 `Paper/citation_todo_list.md`：能可靠核验的真实引用补入 `Paper/references_seed.bib` 和英文稿；不能核验的继续保留 TODO，不得臆造。
3. 根据 `codex1_workspace/submission_asset_checklist.md`，统一正文主表、stage 图、appendix 表和 `Paper/appendix_evidence_note.md` 的引用位置与 caveat。
4. 检查中文稿、英文稿、图注和 appendix 是否都保持同一 claim 强度：`threshold_only_progressive` 是 mixed but useful improvement，H2/dual/exact `A_hard` 是边界/支撑材料。

不要做：

- 不扩实验。
- 不改代码。
- 不把中文草稿中的审慎结论翻译成英文强 claim。

## Mainline Agent 下一步

每次收到 agent 日结后执行：

1. 更新 `PROJECT_MAINLINE.md`。
2. 更新 `EXPERIMENT_LEDGER.md`。
3. 更新 `CLAIM_EVIDENCE_MAP.md`。
4. 更新 `NEXT_ACTIONS.md`。
5. 判断是否存在重复实验、主线漂移、证据不足或 claim 过强。

## 当前最小必要工作

当前最小必要工作不是新实验，而是：

```text
英文投稿稿 polish + citation TODO 处置 + 图表/附录 cross-reference 最终 QA + claim-evidence 最终审查
```

## 本次更新吸收的日结

- `mainline_agent/agent_reports/20260520_codex1_daily.md`

## 本次直接核验的新增产物

- `codex1_workspace/submission_asset_checklist.md`
- `Paper/paper_draft_en_v1.md`
- `Paper/citation_todo_list.md`
- `Paper/references_seed.bib`
