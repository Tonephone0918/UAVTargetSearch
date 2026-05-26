# 结果备忘 v2（中文）

## 0. 本轮使用原则

1. 本轮写作只锁定当前已经核实的主线目录，不把结果备忘录扩展成新的实验计划。
2. 正文主排名优先使用任务、安全与 shield-behavior 指标：
   `search_rate`、`collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate`、`recursive_gate_rate`、`perf_shield_time_ms`
3. `episode_return` 不是本轮主排序指标。
   原因：旧 baseline 与后续 `risk_base` 系列存在 reward normalization 口径差异，不能在所有目录间无条件横比。
4. 只要结果口径不同，就必须单独标记：
   - `formal compare`
   - `training-seed compare`
   - `re-aggregated mechanism summary`
   - `single-checkpoint boundary compare`

## 1. 本轮写作当前引用的结果目录

### A. progressive 主线正式比较

1. `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
   - 口径：`formal compare`
   - 聚合方式：`3 training seeds x 5 eval seeds x 5 episodes`
   - 当前用途：正文主表的任务/安全指标基线
   - 备注：部分 `perf_*` 指标不宜直接作为最终 runtime 引用

2. `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/per_checkpoint_metrics.csv`
   - 口径：`training-seed compare`
   - 当前用途：观察不同 training seed 的波动与不确定性
   - 备注：不与 `summary_metrics.csv` 混写为同一层级结论

### B. progressive 机制与 runtime 汇总

1. `runs/progressive_mechanism_20260428/summary_metrics.csv`
   - 口径：`re-aggregated mechanism summary`
   - 当前用途：引用 `perf_shield_time_ms`、`perf_recursive_time_ms` 以及 stage-aware 机制描述
   - 备注：这一文件优先于旧 `formal_compare_multiseed5x5/summary_metrics.csv` 的 runtime 汇总

2. `runs/progressive_mechanism_20260428/stage_metrics.csv`
   - 口径：按 `early / mid / late / fixed` 切分的阶段统计
   - 当前用途：解释 `threshold_only_progressive` 与 `safeearly_progressive` 的课程差异
   - 备注：适合支撑方法/机制讨论，不适合作为唯一主表来源

### C. dual 边界比较

1. `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/summary_metrics.csv`
   - 口径：`formal compare`
   - 比较对象：`non_progressive`、`threshold_only_progressive`、`safeearly_progressive`、`threshold_only_dual_progressive`
   - 当前用途：dual 的边界结果与 discussion 材料

2. `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_checkpoint_metrics.csv`
   - 口径：`training-seed compare`
   - 当前用途：避免 dual 只凭个别 seed 的偶然优势被误写为稳定结果

### D. H1/H2 边界比较

1. `runs/final_formal_h2_vs_h1_multiseed3x3/summary_metrics.csv`
   - 口径：`single-checkpoint boundary compare`
   - 聚合方式：固定 checkpoint，`3 eval seeds x 3 episodes`
   - 当前用途：H2 的当前边界定位
   - 备注：这是边界结果，不应与 progressive 主线 formal compare 混成同一强度的结论

## 2. 本轮已经确认的结论

1. `threshold_only_progressive` 是当前最稳的主正结果候选。
   - 可保守写法：它相对 `non_progressive` 在 `guarantee_broken_rate` 与 `dead_end_rec_rate` 上更稳，`search_rate` 基本持平。
   - 不可写法：它“全面优于” `non_progressive`。

2. `safeearly_progressive` 当前更适合作为主线对照或消融。
   - 可保守写法：它说明 late-stage 切入更强 layer 不会自动转化为更优 learned outcome。
   - 不可写法：它代表“更强版本的 progressive 已经成功”。

3. dual 当前没有形成可直接放进正文主成功故事的稳定收益。
   - 可保守写法：它在部分 runtime 维度上更轻，但主安全指标未打赢 `threshold_only_progressive`。
   - 不可写法：它构成第二条成熟主创新。

4. H2 当前只能写成 stronger layer 的候选方向或边界结果。
   - 可保守写法：H2 在部分安全/可行性指标上显示出值得研究的信号，但 learned checkpoint 层面的正式证据仍是 mixed。
   - 不可写法：H2 已经稳定优于 H1。

5. 当前主文关于 stronger filtering 与 better learned policy 不天然等价的说法可以成立，但应保持机制性、审慎性表述。
   - 可保守写法：已有 progressive 主线、H2 边界结果与 dual 边界结果共同提示二者并非单调关系。
   - `TODO`：若后续补齐 matched frontier 或 cross-eval 证据，可进一步加强这一段。

## 3. 本轮属于 mixed 的结论

1. `threshold_only_progressive` 是否“整体优于” `non_progressive`
   - 当前不能这么写。
   - 更准确的说法是：它在部分安全/未来可行性指标上更优，但 `collision_count` 和 runtime 不占优。

2. `safeearly_progressive` 是否说明 late-stage H2 curriculum 更好
   - 当前不能这么写。
   - 更准确的说法是：它没有稳定优于 `threshold_only_progressive`，因此更像一个对照分支。

3. H2 是否已经证明更强前瞻过滤能稳定提升训练结果
   - 当前不能这么写。
   - 更准确的说法是：H2 作为 runtime layer 有候选价值，但 learned checkpoint 结果仍 mixed。

4. dual 是否已经证明动态阈值调度优于单阈值 threshold curriculum
   - 当前不能这么写。
   - 更准确的说法是：dual 更像降低介入成本的尝试，目前主安全指标未形成优势。

## 4. 当前不宜写太满的地方

1. 不要把 `threshold_only_progressive` 写成“全面支配”。
2. 不要把 `safeearly_progressive` 写成“更强更优版本”。
3. 不要把 `H=2` 写成“已经完成闭环验证的主创新”。
4. 不要把 dual 写成“完整成功的第二方法”。
5. 不要把 `episode_return` 当作跨全部目录统一可比的主指标。
6. 不要把当前 mixed 结果包装成单调的“过滤越强越好”故事。

## 5. 各分支当前在文中的建议定位

### `threshold_only_progressive`

- 建议定位：正文主结果 / 当前最稳的主正分支
- 当前可保守表述为：
  - “在 hard-safe 始终保留的前提下，threshold curriculum 在部分安全与 future-feasibility 指标上带来相对稳定的改善。”
- 若后续 codex1 的 matched frontier 结果继续支持，则可加强为：
  - “其收益不仅来自 gate 更常开，还来自更合适的训练期保守性注入方式。”

### `safeearly_progressive`

- 建议定位：主线对照 / 消融
- 当前可保守表述为：
  - “它用于检验 late-stage 切入更强 layer 是否自然受益，而当前证据并不支持这一点。”

### `H=2`

- 建议定位：边界结果 / 机制材料 / appendix 候选
- 当前可保守表述为：
  - “H2 展示了 stronger runtime layer 的方向，但当前 learned checkpoint 证据仍是 mixed。”
- 若后续有更公平、更稳的 frontier 小补实验继续支持，则可加强为：
  - “H2 的价值更可能体现在 runtime stronger layer，而非当前训练闭环已经稳定吸收。”

### dual

- 建议定位：discussion / appendix / future work 过渡
- 当前可保守表述为：
  - “dual 更像一种运行时阈值调度尝试，目前主要体现为介入成本侧的变化，而非主安全指标的稳定优势。”

## 6. 当前可直接引用的几组数字

### progressive 主线 formal compare

- `non_progressive`
  - `search_rate ≈ 0.9973`
  - `collision_count ≈ 90.79`
  - `guarantee_broken_rate ≈ 0.343`
  - `dead_end_rec_rate ≈ 0.464`

- `threshold_only_progressive`
  - `search_rate ≈ 0.9987`
  - `coverage_ratio ≈ 0.9979`
  - `collision_count ≈ 92.57`
  - `guarantee_broken_rate ≈ 0.332`
  - `dead_end_rec_rate ≈ 0.444`

- `safeearly_progressive`
  - `search_rate = 1.0`
  - `collision_count ≈ 94.73`
  - `guarantee_broken_rate ≈ 0.354`
  - `dead_end_rec_rate ≈ 0.467`

### progressive mechanism summary 中更适合引用的 runtime

- `non_progressive`
  - `perf_shield_time_ms ≈ 197.84`
  - `perf_recursive_time_ms ≈ 167.40`

- `threshold_only_progressive`
  - `perf_shield_time_ms ≈ 238.13`
  - `perf_recursive_time_ms ≈ 202.96`

- `safeearly_progressive`
  - `perf_shield_time_ms ≈ 192.30`
  - `perf_recursive_time_ms ≈ 162.81`

### dual 边界结果

- `threshold_only_dual_progressive`
  - `search_rate ≈ 0.9987`
  - `coverage_ratio ≈ 0.9995`
  - `collision_count ≈ 108.49`
  - `guarantee_broken_rate ≈ 0.386`
  - `dead_end_rec_rate ≈ 0.471`

### H2 边界结果

- `recursive_risk_rescue_h2_eta055`
  - `search_rate ≈ 0.9778`
  - `collision_count ≈ 96.11`
  - `guarantee_broken_rate ≈ 0.350`
  - `dead_end_rec_rate ≈ 0.150`

## 7. 当前阻塞点与口径提醒

1. `formal_compare_multiseed5x5/summary_metrics.csv` 的部分 `perf_*` 聚合与后续 re-aggregated 机制汇总不完全一致。
   - 当前处理：任务/安全指标可以继续引用 formal compare；runtime 优先引用 `runs/progressive_mechanism_20260428/summary_metrics.csv`。

2. H2 当前只有 `final_formal_h2_vs_h1_multiseed3x3/` 被纳入这轮备忘录。
   - 当前处理：只把它写成边界结果，不把它上升为更强结论。
   - `TODO`：若后续需要加强 H2 讨论，再单独补充更大口径 compare，并明确与主线 formal compare 的层级差别。

3. 本轮结果备忘录没有把新的 frontier 小补实验当成前提。
   - 当前处理：所有正文表述都基于“现在已核实证据”。
   - 若 codex1 后续补来 matched frontier 结果，应另开一版 v3 备忘录，不要直接口头覆盖 v2。
