# 论文提纲 v1（中文）

## 1. 题目候选

1. 面向多无人机协同搜索的分层 Allowed-Action Shield：基于精确 `A_hard` 底座的渐进保守性课程与机制分析
2. 多无人机协同搜索中的分层安全动作过滤：从精确 `A_hard` 到渐进式保守性调度
3. 基于精确硬安全底座的多无人机协同搜索 Shield 框架：Allowed-Action Filtering 与学习机制分析
4. A Grounded Layered Shield for Safe Multi-UAV Cooperative Search: Allowed-Action Filtering, Curriculum, and Mechanism Analysis
5. 多无人机协同搜索中的精确 `A_hard` 与分层 Shield：强运行时过滤为何不必然带来更优学习策略

## 2. 当前最推荐题目方向

最推荐当前使用题目 1 或其英文对应版本。

推荐理由：

- 它把创新落点放在“分层 allowed-action shield + 精确 `A_hard` 底座 + 机制分析”，而不是把 `H=2` 或 dual 硬包装成主成功结果。
- 它保留了 progressive / curriculum 的位置，但没有把 progressive 写成“前期不安全、后期开安全”。
- 它天然允许把 `H=2`、dual、exact rescue 写成边界结果、机制证据或 appendix。

## 3. 核心研究问题

1. 在多无人机协同搜索（multi-UAV cooperative search）中，如何把 shield 明确为一种 `allowed-action filtering` 机制，而不是外部 planner 直接接管动作选择？
2. 如何以 exact / grounded 的一步硬安全集合 `A_hard` 为底座，构造 `A_hard -> A_rec -> A_H^{viable}` 的分层安全过滤框架？
3. progressive / threshold curriculum 应如何被解释为“保守性课程”（conservativeness curriculum），而不是 hard-safe 的开/关切换？
4. 更强的运行时安全过滤（stronger runtime filtering）在什么条件下会或不会转化为更好的 learned policy？

## 4. 当前最可防守的贡献点

1. 给出一个以 exact / grounded `A_hard` 为底座的分层 shield 视角，将多无人机协同搜索中的安全约束统一为 `allowed-action set` 的计算与收缩问题。
2. 明确区分 exact projected `A_hard`、顺序近似 `A_hard` 与 `sequential_with_exact_rescue`，并提出“true dead-end vs approximation-induced dead-end”的诊断视角。
3. 将 progressive 重新表述为 hard-safe 始终保留前提下的保守性课程，强调训练中调节的是 `A_hard` 到更强层级的升级强度，而不是是否允许不安全动作通过。
4. 基于 formal compare、cross-eval、matched gate-rate / compute-budget 诊断指出：更强的 runtime filtering 与更优的 learned policy 并不天然等价，这一错配本身构成了机制层面的经验贡献。

## 5. 全文结构提纲

### 第 1 节：引言

- 问题背景：多无人机协同搜索同时面临任务性能、安全约束和在线计算三重张力。
- 现有问题：shield 并不新，但很多表述会把它写成 planner takeover，或把 progressive 误写成 hard-safe on/off。
- 本文主线：`A_hard` always-on，shield 是 allowed-action filtering，progressive 是 conservativeness curriculum。
- 主要发现：最稳的收益来自 `threshold_only_progressive`；`H=2` 与 dual 当前是 mixed / 边界结果。

### 第 2 节：相关工作

- Safe RL / shielded RL
- Multi-agent shielding / dynamic shielding
- 多无人机协同搜索中的安全增强方法
- 本文区别：不宣称发明 shielding，本工作关注 grounded `A_hard`、layered filtering、dead-end diagnosis、以及 filtering-learning mismatch

### 第 3 节：问题定义与方法总览

- 环境、动作空间、硬安全约束
- actor-shield 交互语义
- 分层结构：`A_hard`、`A_rec`、`A_H^{viable}`
- 主算法骨架：MAPPO + centralized shield

### 第 4 节：精确 `A_hard` 语义与 dead-end 诊断

- exact joint feasibility 与 projected admissible action
- 顺序近似的来源与限制
- `sequential_with_exact_rescue`
- true dead-end vs approximation-induced dead-end

### 第 5 节：progressive / threshold curriculum

- progressive 不是 hard-safe on/off
- 课程只调节保守层级或触发阈值
- `threshold_only_progressive` 与 `safeearly_progressive` 的 schedule 差异
- `FIXME`：风险函数是写成 `risk_base` 还是先保留更抽象的 threshold curriculum 叙事

### 第 6 节：实验设置

- 主训练/评测口径
- 指标体系
- 主比较对象与边界比较对象
- 结果引用优先级与可比性说明

### 第 7 节：主结果

- `non_progressive` / `threshold_only_progressive` / `safeearly_progressive`
- 强调 `threshold_only_progressive` 是 mixed 但最可防守的主正结果
- 不把 `episode_return` 作为跨口径主排名指标

### 第 8 节：机制与边界结果

- exact-solver 诊断：顺序近似与 exact projected `A_hard` 的差异
- H1/H2 matched gate-rate / compute-budget
- H1/H2 `2x2` cross-eval
- dual scheduling 当前表现

### 第 9 节：讨论与局限性

- stronger filtering != better learned policy
- `H=2` 和 dual 目前为何不宜升格为主创新
- 当前证据边界与尚缺实验

### 第 10 节：结论

- 回到 grounded `A_hard`、layered shield、conservativeness curriculum 和 mismatch analysis

## 6. 正文主表 / 主图建议

### 主表建议

1. 主表 1：`non_progressive`、`threshold_only_progressive`、`safeearly_progressive` 的正式 `3 training seeds x 5 eval seeds x 5 episodes` 对比
   - 主指标：`search_rate`、`collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate`
   - 次指标：`recursive_gate_rate`、`perf_shield_time_ms`
   - 备注：`episode_return` 不作为主排序依据

2. 主表 2：H1/H2 的边界比较摘要
   - 可压缩展示 `final_formal_h2_vs_h1_*` 与 matched gate-rate / compute-budget 的代表项
   - 目的不是证明 H2 全面更优，而是证明结论 mixed

### 主图建议

1. 图 1：方法框架图
   - actor proposal
   - `A_hard`
   - `A_rec`
   - `A_H^{viable}`
   - actor re-selection

2. 图 2：`A_hard` exact projected 语义示意图
   - exact joint feasible set
   - per-agent projection
   - sequential approximation
   - false empty / rescue

3. 图 3：progressive curriculum 时间轴
   - `early: safe/H=1/eta=0.9`
   - `mid: recursive/H=1/eta=0.35`
   - `late`
     - threshold-only: `H=1/eta=0.35`
     - safeearly: `H=2/eta=0.55`

4. 图 4：H1/H2 `2x2` cross-eval 矩阵图
   - 直接展示 `H1 ckpt + H2 shield` 与其他三格的关系

## 7. Appendix 建议放什么

1. exact hard-solver 诊断表
   - `model_compare_exact_hard_solver_fast2x1`
   - `model_compare_exact_hard_solver_diag_medium2x2`

2. H1/H2 的 matched gate-rate / matched compute-budget 细表

3. H2 runtime ablation
   - `refine_only`
   - `exact_empty_only`
   - `risk_trigger_only`
   - `viability_only`

4. progressive stage-level 统计
   - `runs/progressive_mechanism_20260428/stage_metrics.csv`

5. 结果口径说明
   - single-checkpoint compare
   - training-seed compare
   - formal compare
   - raw vs re-aggregated summary

## 8. 当前最危险的创新性风险

最危险的风险不是“结果不够漂亮”，而是“叙事把已有 shielding 文献、dynamic shielding 文献和 look-ahead 文献已经做过的东西误写成本文独有创新”。

具体包括：

- 把 shielding 本身写成创新
- 把 progressive 写成一个空泛口号
- 把 `H=2` 或 dual 在 mixed 证据下写成稳定主正结果
- 把 runtime stronger filtering 与 learned policy improvement 混为一谈
- 把 reward 不完全可比的结果写成统一回报胜利

## 9. 建议的规避方式

1. 创新表述只聚焦在“grounded `A_hard` + layered allowed-action framework + dead-end diagnosis + mismatch analysis”。
2. 不使用“首个”“首次”“开创性”等高风险措辞。
3. 正文主表只放最可防守的 progressive 主线；`H=2`、dual 主要放机制/边界结果。
4. 明确写出 `threshold_only_progressive` 的收益是 mixed improvement，而不是全面支配。
5. 对 `episode_return` 统一加可比性说明，主排名依赖任务、安全与 shield-behavior 指标。
6. 单独设一节讨论 stronger runtime filtering 与 learned policy mismatch，这样 mixed 结果会转化为机制贡献，而不是叙事漏洞。
