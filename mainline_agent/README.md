# Mainline Agent Workspace

这个目录是 curator / mainline agent 的固定工作区，用来维护项目主线、证据地图、实验台账和下一步行动。

## 使用方式

每次 `codex1`、`codex2` 或其他工作 agent 完成阶段性任务后，把它们的日结或最终汇报放到：

```text
mainline_agent/agent_reports/
```

推荐命名：

```text
YYYYMMDD_codex1_daily.md
YYYYMMDD_codex2_daily.md
YYYYMMDD_codex3_daily.md
```

然后让我读取这些日结，并更新：

- `PROJECT_MAINLINE.md`
- `EXPERIMENT_LEDGER.md`
- `CLAIM_EVIDENCE_MAP.md`
- `NEXT_ACTIONS.md`

## Curator 原则

- 主线优先：先判断结果是否服务当前论文主线。
- 证据对齐：每个 claim 都要对应具体文件、表、图或实验目录。
- 结论克制：区分正文主结果、附录边界结果、内部探索结果。
- 避免重复：已经满足要求的实验不重复跑。
- 最小下一步：优先安排能消除最大不确定性的最小工作。

