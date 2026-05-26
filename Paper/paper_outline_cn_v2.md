# 论文提纲 v2（中文）

## 1. 题目候选

1. 面向多无人机协同搜索的分层 Allowed-Action Shield：基于精确 `A_hard` 底座的保守性课程与机制分析
2. 多无人机协同搜索中的分层安全动作过滤：从精确 `A_hard` 到渐进式保守性调度
3. 基于精确硬安全底座的多无人机协同搜索 Shield 框架：Allowed-Action Filtering 与学习机制分析
4. 多无人机协同搜索中的分层 Shield 语义：`A_hard`、递归可行性与渐进保守性课程
5. 多无人机协同搜索中的精确 `A_hard` 与分层安全过滤：为何更强运行时过滤不必然带来更优学习策略

## 2. 当前最推荐题目

当前最推荐使用题目 1：

> 面向多无人机协同搜索的分层 Allowed-Action Shield：基于精确 `A_hard` 底座的保守性课程与机制分析

推荐理由如下：

- 题目把重心放在“精确 `A_hard` 底座 + 分层 allowed-action 框架 + 机制分析”，与当前最稳的证据一致。
- 它保留 progressive / threshold curriculum 的位置，但不会把 progressive 误写成 hard-safe 的开关。
- 它自然允许把 `H=2` 和 dual 放在边界结果或 appendix，而不需要把它们包装成已经成熟的主成功层。

## 3. 当前最可防守的核心贡献

1. 提出一个以 exact / grounded `A_hard` 为底座的分层 shield 视角，将多无人机协同搜索中的安全控制统一表述为 allowed-action set 的构造、投影与收缩问题。
2. 明确给出 `A_hard`、`A_rec` 与 `A_H^{viable}` 的层级关系，并强调 shield 的语义是 allowed-action filtering，而不是外部 planner 对 actor 的动作接管。
3. 区分 exact/projected `A_hard` 与顺序近似 `A_hard`，并用 true dead-end 与 approximation-induced dead-end 的诊断视角解释 dead-end 的不同来源。
4. 基于当前已完成的 progressive formal compare 与边界结果，指出 progressive / threshold curriculum 调节的是保守性注入强度；同时说明 stronger runtime safety filtering 与 better learned policy improvement 并不天然等价。

## 4. 当前最危险的创新性风险

当前最大的风险不是“结果不够强”，而是“把已有 shielding、look-ahead feasibility、dynamic scheduling 文献中已经出现过的思想误写成本文独有创新”。具体风险包括：

- 把 shielding 本身写成新发明。
- 把 progressive 写成泛泛的“前弱后强”口号，而没有说明 hard-safe 一直保留。
- 在 mixed 证据下把 `H=2` 或 dual 写成稳定主结果。
- 把更强运行时过滤直接等同于更优 learned policy。
- 把 reward normalization 不完全一致的结果写成统一回报胜利。

## 5. 规避这些风险的写法建议

1. 创新表述聚焦在“grounded `A_hard` + layered allowed-action framework + dead-end diagnosis + filtering-learning mismatch analysis”。
2. 不使用“首个”“首次”“开创性”等高风险措辞，也不把 shield 本身写成发明对象。
3. 正文主线只围绕 `non_progressive`、`threshold_only_progressive` 和 `safeearly_progressive` 展开，其中 `threshold_only_progressive` 写成最稳的主正结果候选，而不是全面支配者。
4. `H=2` 与 dual 在当前稿件中只承担边界结果、机制材料或 appendix 候选的角色。
5. `episode_return` 只作辅助描述；正文主排名优先依赖 `search_rate`、`collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate` 与 runtime 指标。
6. 对尚待 codex1 后续核对的地方，用“当前可保守表述为……”和“若后续 matched frontier 结果继续支持，则可加强为……”来控制表述强度。

## 6. 正文主表推荐结构

### 主表 1：progressive 主线正式比较

- 比较对象：`non_progressive`、`threshold_only_progressive`、`safeearly_progressive`
- 主要口径：`runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/`
- 主指标：`search_rate`、`collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate`
- 次指标：`recursive_gate_rate`、`perf_shield_time_ms`
- 写法重点：把 `threshold_only_progressive` 写成“在部分安全/可行性指标上更稳”，而不是“全面更优”

### 主表 2：可选的边界结果摘要表

- 仅在版面允许时保留
- 比较对象：`threshold_only_progressive`、`threshold_only_dual_progressive`、`recursive_risk_rescue_h2_eta055`
- 主要口径：
  - `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/`
  - `runs/final_formal_h2_vs_h1_multiseed3x3/`
- 写法重点：证明结论 mixed，而不是证明 H2/dual 全面更强

## 7. 正文主图推荐结构

### 图 1：方法总览图

- actor 输出原始动作偏好
- shield 计算 allowed action set
- 层级结构：`A_hard -> A_rec -> A_H^{viable}`
- 若原动作不允许，则 actor 在 allowed set 内重选

### 图 2：`A_hard` 的 exact/projected 语义示意图

- 联合可行集
- 单 agent 投影可行动作
- 顺序近似与 exact projected set 的差异
- false empty 与 rescue 的概念位置

### 图 3：progressive / threshold curriculum 时间轴

- `early`: `safe`, `H=1`, high threshold
- `mid`: `recursive`, `H=1`, tighter threshold
- `late`:
  - `threshold_only_progressive`: 维持 `H=1`
  - `safeearly_progressive`: 切入 `H=2`

### 图 4：边界结果示意图（可选）

- 用一张小图说明 H2/dual 当前并非稳定主正结果
- 可画成“更强 filtering”与“更好 learned policy”之间非单调关系的概念图

## 8. Appendix 推荐结构

1. progressive 实验口径说明
   - `formal compare`
   - `training-seed compare`
   - `summary_metrics` 与 re-aggregated runtime 的区别

2. stage-level curriculum 统计
   - `runs/progressive_mechanism_20260428/stage_metrics.csv`
   - 说明 `threshold_only_progressive` 与 `safeearly_progressive` 的实际 stage 差异

3. H2 与 dual 的边界结果细表
   - H2 当前为何只能保守写成 stronger layer 候选
   - dual 当前为何更适合 discussion / future work

4. `A_hard` exact/projected 语义补充
   - exact feasibility 定义
   - sequential approximation 的误差来源
   - `TODO`：若后续补齐主环境同口径 exact 诊断，可把更多结果移入正文
