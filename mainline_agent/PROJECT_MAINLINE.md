# Project Mainline

更新时间：`2026-05-21`

## 当前论文主线

当前主线固定为：

```text
progressive shielding / conservativeness curriculum
```

更准确地说，本文不是讲“hard-safe shield 从关闭到开启”，而是在 always-on `A_hard` 底座上，研究 stronger layer 的介入时机和保守性强度如何影响 learned policy。

## 核心定位

本文将多无人机协同搜索中的安全模块组织为分层 allowed-action shield：

- `A_hard`：一步硬安全底座，always-on。
- `A_rec`：在 `A_hard` 基础上加入一步递归可行性。
- `A_H^{viable}`：有限小视界 future-feasibility。

理想语义下：

```text
A_H^{viable} subseteq A_rec subseteq A_hard
```

shield 的语义是 allowed-action filtering，不是 planner takeover。

## 正文主比较

正文主比较只围绕三组：

1. `non_progressive`
2. `threshold_only_progressive`
3. `safeearly_progressive`

## 当前主结果强度

`threshold_only_progressive` 是当前最稳主正结果候选，但只能写成：

```text
mixed but useful improvement
```

可写：

- 相比 `non_progressive`，它降低 `guarantee_broken_rate` 和 `dead_end_rec_rate`。
- `search_rate` 基本持平并略高。
- 它支持 progressive conservativeness curriculum 的有限收益。

不能写：

- 不能写成全面支配。
- 不能写成所有安全指标都更好。
- 不能忽略 `collision_count` 和 runtime 不占优。

## 边界材料定位

- `safeearly_progressive`：late-stage stronger-layer 消融/对照，不是更强成功版本。
- H2：runtime stronger-layer candidate，不是 learned-policy 主成功。
- dual scheduling：runtime 边界材料，不是第二条成熟主创新。
- exact/projected `A_hard`：理论参照和诊断支撑，不是正文主实验，也不是在线主路径每步 exact solver。
- matched analysis：支持“不是简单 gate more / compute more”的审慎说法，不是完整 frontier 证明。

## 当前论文阶段

当前不需要重复跑大规模训练。codex1 已生成正文主表、stage-level 图表、appendix 边界表、appendix evidence note，并完成投稿资产检查清单；codex2 已将英文稿从 skeleton 推进到可继续收敛的投稿草稿，并生成 citation TODO list。

下一阶段从“材料生成”进入“投稿稿收敛”：

- 英文稿需要从 tighter draft 继续进入 polished submission draft。
- related work 的剩余 citation gap 需要核验补齐，不能臆造文献。
- 图表、附录和正文 cross-reference 需要最终一致性检查。
- 最终 claim 仍需保持 mixed but useful improvement 的审慎强度。
