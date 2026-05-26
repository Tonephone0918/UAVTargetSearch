# 中文论文 Master Draft v2

## 题目

面向多无人机协同搜索的分层 Allowed-Action Shield：基于 Grounded `A_hard` 语义的保守性课程与机制分析

`TODO`：最终题目可在“方法框架导向”和“机制分析导向”之间再收敛。当前版本避免使用“每步精确求解”的暗示，强调 grounded `A_hard` semantics 与 exact/projected view。

## 摘要（保守版）

多无人机协同搜索要求多个智能体在动态威胁与机间耦合约束下协同发现目标，同时保持持续可执行的安全行为。本文将该问题中的安全模块组织为一个分层 allowed-action shield 框架。与把 shield 描述成外部 planner 直接接管动作选择不同，本文强调 shield 的语义是 allowed-action filtering：actor 先提出动作偏好，shield 构造允许动作集合，若原动作不可接受，actor 仅在允许集合内部重新选择。该语义保留了策略学习的主体地位，也使不同强度的安全逻辑可以统一表述为 allowed set 的逐层收缩。

本文以 grounded `A_hard` semantics 为底座，并用 exact/projected `A_hard` view 作为参照，组织出 `A_hard`、`A_rec` 与 `A_H^{viable}` 三层结构。`A_hard` 负责一步硬安全底线，`A_rec` 进一步约束下一步递归可行性，`A_H^{viable}` 则表示有限小视界的 future-feasibility。基于这一层级关系，progressive / threshold curriculum 被解释为 conservativeness curriculum：hard-safe 层始终存在，训练过程调节的是 stronger layer 的介入时机与保守性强度，而不是打开或关闭 hard-safe。

在当前 formal comparison 中，`threshold_only_progressive` 是最稳的主正结果候选：它相对 `non_progressive` 降低了 guarantee violation 与 recursive dead-end 指标，并保持接近的 search performance；但它在 collision count 和 runtime 上不占优，因此只能写成 mixed but useful improvement。`safeearly_progressive`、H2 和 dual scheduling 当前均呈 mixed 结果，更适合作为对照、边界结果或 appendix 材料。本文据此强调一个机制结论：stronger runtime safety filtering 与 better learned policy improvement 并不天然等价。

## 1. 引言

多无人机协同搜索（multi-UAV cooperative search）要求多个智能体在受限时间和受限观测条件下协同发现目标、完成区域覆盖，并同时规避威胁区域与机间碰撞。与单智能体安全强化学习相比，这类任务的困难不只在于避免一次危险动作，还在于如何在多机耦合约束下持续保留后续可行动作，避免系统被推进到局部拥挤、交换冲突或下一步无安全续行动作的状态。

现有关于 shielding 的研究已经表明，在执行前对动作进行安全过滤是一条重要路线；然而，针对多智能体协同搜索任务，仍有两个问题经常没有被讲清。第一，一部分工作在叙事上容易把 shield 描述成一个外部控制器，仿佛它直接替 actor 选择“最优安全动作”。这种表述会模糊策略学习与安全过滤之间的职责边界。第二，关于 progressive shielding 的讨论常被简化为“训练前期弱安全、训练后期强安全”的 warmup 方案，但如果这种“弱安全”被理解为允许不安全动作通过，那么 hard-safe 约束在语义上就已经被放松。

本文采用一条更保守也更可防守的主线。我们不把 shielding 本身写成新发明，而是把多无人机协同搜索中的安全机制重新组织为一个以 grounded `A_hard` semantics 为底座的分层 shield 框架。这里的关键不在于由 shield 直接输出一个动作，而在于由 shield 构造允许动作集合（allowed action set），并约束 actor 只能从该集合中执行动作。换言之，shield 的语义是 allowed-action filtering，而不是 planner takeover。

基于这一视角，本文把安全过滤组织为三层：一步硬安全集合 `A_hard`、递归可行集合 `A_rec`，以及有限小视界可行集合 `A_H^{viable}`。`A_hard` 对应一步层面的 hard-safe 底线；`A_rec` 在 `A_hard` 基础上进一步排除那些会把系统立即推向下一步 dead-end 的动作；`A_H^{viable}` 则在更短的前瞻视界上维持 future-feasibility。由此，shield 的作用不再只是普通 action mask，也不是临时拼接的 heuristic shield，而是有明确层级语义的 allowed-set framework。

这一框架也改变了我们对 progressive / threshold curriculum 的理解。本文当前主线并不把 progressive 解释为“shield off -> on warmup”，而是把它解释为 conservativeness curriculum。训练过程中始终保留 `A_hard` 这一 hard-safe 底座，progressive 调节的是是否、以及以多大强度，从 `A_hard` 升级到更保守的 `A_rec` 或 `A_H^{viable}` 层。风险阈值的作用是决定 stronger layer 何时介入，而不是决定 hard-safe 是否存在。

值得强调的是，本文并不试图讲述一个“过滤越强，策略就一定学得越好”的简单故事。当前已有结果提示，这种单调叙事并不稳固。`threshold_only_progressive` 是当前最稳的主正结果候选，但其收益主要体现为部分安全/可行性指标的改善，而不是对所有任务指标、风险指标和在线成本的全面支配。H2 stronger layer 与 dual scheduling 虽然在部分设置下显示出候选价值，却尚未形成稳定、无争议的 learned policy 优势。

本文当前最可防守的贡献可以概括为四点。第一，本文提出并分析了一个以 grounded `A_hard` semantics 为底座的分层 allowed-action shield 框架。第二，本文明确 `A_hard`、`A_rec` 与 `A_H^{viable}` 的关系，并强调 shield 的语义是 filtering 而不是 takeover。第三，本文使用 exact/projected `A_hard` view 区分 true dead-end 与 approximation-induced dead-end。第四，本文基于当前实验结果，保守指出 threshold curriculum 的有限收益，并分析 stronger runtime filtering 与 learned policy improvement 之间的错配。

## 2. Related Work

### 2.1 Shielding 与 Safe RL

安全强化学习通常试图在策略优化过程中同时考虑任务收益与约束满足。常见路线包括约束马尔可夫决策过程、拉格朗日惩罚、风险敏感目标、控制屏障函数，以及执行时的 shielding。与只在奖励中加入安全惩罚的方法相比，shielding 可以在动作执行前对策略输出进行约束，从而把一部分安全责任从训练目标转移到运行时过滤机制上。`TODO: citation`

本文与这类工作共享执行前过滤的基本思想，但不把 shielding 本身作为新发明。本文关注的是在多无人机协同搜索中如何把 shield 的语义写清楚：shield 不是一个替 actor 直接选择最优动作的外部 planner，而是一个 allowed-action filtering 模块。

### 2.2 Multi-agent Shielding 与 Safe MARL

多智能体安全强化学习进一步引入了智能体间耦合约束。单智能体中可局部判断的安全动作，在多智能体场景下可能取决于其他 agent 的同步动作。例如，多 UAV 系统不仅需要规避威胁区域，还需要避免机间碰撞、交换位置冲突，以及局部拥挤导致的下一步无安全动作。`TODO: citation`

本文不主要解决 shield 的去中心化可实现性，也不声称提出一般形式的 multi-agent shielding 算法。本文聚焦于多无人机协同搜索任务，并把其安全约束组织为以 grounded `A_hard` semantics 为底座的 allowed-action framework。

### 2.3 Look-ahead、MPC 与 Viability Style Shielding

look-ahead、model-predictive control（MPC）和 viability kernel 相关方法通常不只检查当前动作是否立即安全，还会评估动作执行后是否存在未来安全 continuation。这与本文的 `A_rec` 和 `A_H^{viable}` 有直接关联。`A_rec` 可以理解为一步递归可行性检查，`A_H^{viable}` 则对应有限小视界的 future-feasibility 约束。`TODO: citation`

本文需要强调的区别是，当前方法不应被写成标准 MPC controller。标准 MPC 往往通过滚动优化返回单个动作，而本文的 shield 返回 allowed action set。即使在 `A_H^{viable}` 层使用小视界可行性推理，本文仍保留 actor-shield 的分工：shield 收缩可执行动作空间，actor 在允许集合内部表达偏好并完成重选。

### 2.4 Dynamic、Adaptive 与 Progressive Shielding

动态或自适应 shielding 试图根据训练阶段、风险水平或运行状态调节安全模块的介入强度。本文采用 progressive / threshold curriculum，但其含义需要与普通 warmup 区分。本文当前主线不是“训练前期 shield off、后期 shield on”，而是在 `A_hard` 始终保留的前提下调节更强过滤层的介入强度。`TODO: citation`

这个定位对结果解释同样重要。当前证据并不支持“更强过滤必然带来更优 learned policy”的单调叙事。`threshold_only_progressive` 是当前最稳的主正结果候选，但其收益是 mixed improvement；H2 与 dual 当前更适合作为边界结果或机制材料。

### 2.5 UAV Cooperative Search、MAPPO 与 Action Masking

多无人机协同搜索任务通常结合多智能体协同、部分可观测、动态威胁、目标发现和覆盖控制等因素。MAPPO 等 centralized training with decentralized execution 风格的方法为这类任务提供了常用学习基座。另一方面，在离散动作空间中使用 action masking 也很常见。`TODO: citation`

本文与普通 action masking 的区别在于，本文不只关心哪些动作当前非法，还关心 allowed set 的语义来源和层级关系。普通 action mask 往往只对应局部不可执行规则；本文则把 mask 组织为 `A_hard -> A_rec -> A_H^{viable}` 的分层 allowed-action set，并把 `A_hard` 放在 exact/projected view 的参照下理解。

### 2.6 本文定位

本文不声称发明 shielding，也不把 look-ahead feasibility 或 action masking 本身写成新概念。本文当前更可防守的差异在于 grounded `A_hard` semantics、layered allowed-action framework、dead-end diagnosis，以及 filtering-learning mismatch analysis。

## 3. Method

### 3.1 问题定义

考虑一个多无人机协同搜索环境。设时刻 `t` 的全局状态为 `s_t`，每个 UAV 基于局部观测 `o_t^i` 由策略网络输出动作偏好，联合动作记为 `a_t = (a_t^1,\dots,a_t^n)`。环境转移由 `f(s_t, a_t)` 给出，状态空间中满足边界、威胁规避、机间安全距离以及交换冲突约束的集合记为 `\mathcal S_{\mathrm{safe}}`。一步硬安全联合动作集可记为

\[
A^{\mathrm{safe}}(s_t)=\{a_t \in A \mid f(s_t,a_t)\in \mathcal S_{\mathrm{safe}}\}.
\]

对于多无人机协同搜索，仅有一步 hard-safe 往往不足以保证持续可执行性：一个动作虽然当前安全，却可能让下一步所有动作都不可行，进而形成 dead-end。本文因此采用分层 allowed-action 框架，把当前安全和未来仍可持续安全分开建模。

### 3.2 Shield 作为 Allowed-Action Filtering

本文不把 shield 定义为直接替 actor 选择动作的外部规划器，而把它定义为允许动作集过滤器。给定当前状态 `s_t`，shield 返回

\[
\mathcal A_t^{\mathrm{allow}}(s_t)\subseteq A^{\mathrm{safe}}(s_t).
\]

actor 先输出原始策略 `\pi_\theta(a\mid o_t)`；若原始提议动作已经属于 `\mathcal A_t^{\mathrm{allow}}(s_t)`，则该动作直接执行；否则 actor 在允许集合上进行 masked re-selection。于是，最终执行动作始终来自允许集合，但对允许集合内部的相对偏好仍由 actor 决定。

### 3.3 分层允许动作集

本文使用三层 allowed-action set：

\[
A_hard(s_t), \qquad A_{rec}(s_t), \qquad A_H^{viable}(s_t).
\]

`A_hard` 是 always-on 的一步硬安全底座。`A_rec` 在 `A_hard` 基础上加入一步递归可行性约束，只保留那些执行后下一步仍存在至少一个硬安全续行动作的动作。`A_H^{viable}` 将这一思想推广到有限小视界 `H`，要求存在长度为 `H` 的安全 continuation。若忽略近似误差，三者满足

\[
A_H^{viable}(s_t)\subseteq A_{rec}(s_t)\subseteq A_{hard}(s_t).
\]

这条包含链是本文方法的理论骨架。progressive 调节的不是安全与不安全的切换，而是是否进一步收缩到更保守的 allowed set。

### 3.4 `A_hard` 的 Exact/Projected View

虽然 `A_hard` 在实现中常以逐 agent 的局部规则方式在线构造，但从理论上讲，它对应一个联合可行性对象。给定 agent `i` 的候选动作 `a_i`，若存在其他 agent 的一个联合动作 completion，使得整体联合动作满足一步硬安全，则 `a_i` 应被视为对 agent `i` 可接受：

\[
A_{hard,i}^{\star}(s_t)
=
\{a_i \mid \exists a_{-i},\ (a_i,a_{-i}) \in A^{\mathrm{safe}}(s_t)\}.
\]

本文使用这一 exact/projected view 作为参照语义，而不是声称在线主路径每步都进行 exact 求解。当前实现中的顺序式 `A_hard` 构造是该参照语义的工程近似；exact rescue / diagnostics 用于理解和缓解近似误差。

### 3.5 Dead-end 诊断

exact/projected `A_hard` view 使 dead-end 可以被拆分为两类。若 exact projected `A_hard` 本身为空，则系统处于 true dead-end；若 exact projected `A_hard` 非空，但顺序近似或局部裁决过程返回空集，则该状态对应 approximation-induced dead-end。当前实现中的 `sequential_with_exact_rescue` 可以理解为一种边界纠偏机制：在线主路径仍使用顺序近似，只在顺序近似返回空集或极小候选集时调用 exact witness / rescue。

### 3.6 Progressive / Threshold Curriculum

在本文框架中，training-time curriculum 的核心问题不是“何时开启安全”，而是“何时把 `A_hard` 升级为更保守的子集”。因此，本文把 progressive / threshold curriculum 解释为 conservativeness curriculum：`A_hard` 始终存在，训练过程只调节 `A_rec` 或 `A_H^{viable}` 的介入强度。

当前主线比较中，`threshold_only_progressive` 在 early 阶段使用 `safe/H=1/eta=0.9`，mid 和 late 阶段使用 `recursive/H=1/eta=0.35`。`safeearly_progressive` 的 early 和 mid 阶段相同，但 late 阶段切换到 `recursive/H=2/eta=0.55`。因此，`safeearly_progressive` 更适合作为 late-stage stronger layer 的消融，而不是主成功分支。

### 3.7 理论命题与证据边界

本文方法部分依赖四个层级命题。第一，在忽略工程近似误差时，有限视界可行集合、递归可行集合与一步硬安全集合满足

\[
A_H^{viable}(s_t)\subseteq A_{rec}(s_t)\subseteq A_{hard}(s_t).
\]

第二，risk gate 和 progressive schedule 不破坏 hard safety，因为 gate 只决定是否从 `A_hard` 升级到其更保守子集，而不是决定是否允许 `A_hard` 之外的动作通过。第三，exact/projected `A_hard` 是 joint feasibility 在单 agent 动作上的 projection 参照；在线 `A_hard` 则是这一参照对象的工程近似。第四，顺序近似可能产生 false-empty 或 false-nonempty，因此 exact diagnostic / rescue 的作用是解释和缓解 approximation-induced dead-end，而不是把 exact solver 改写成在线主路径。

这些命题与实验结果的对应关系也需要保持边界。主表检验的是 progressive conservativeness curriculum 在 learned policy 层面的结果，stage-level 图表检验 early/mid/late 阶段 stronger layer 的实际介入方式，matched analysis 只支持“收益不宜简单归因于 gate more / compute more”的审慎表述，H2/dual 边界结果支持 stronger filtering 与 better learned policy 之间的非单调关系，exact/projected `A_hard` 诊断则支撑 dead-end 语义与近似误差讨论。它们共同支撑本文主线，但不能被写成超过现有证据的因果证明。

## 4. Experimental Setup

### 4.1 环境与任务

本文实验基于多无人机协同搜索环境。当前代码默认配置为 `20 x 20` 地图、`10` 架 UAV、`10` 个目标、`5` 个动态威胁，最大 episode 长度为 `120` 步；UAV 间安全距离为 `1`，威胁安全距离为 `2`。实验关注在目标发现、区域覆盖、硬安全违反、递归可行性和在线计算成本之间的权衡。

### 4.2 策略学习基座

本文使用 MAPPO 作为策略学习基座。actor 根据局部观测输出动作偏好，critic 使用集中式信息进行训练。本文不把 MAPPO 本身作为方法创新点，而是将其作为稳定的多智能体强化学习骨架，用于评估不同 shield 语义和 curriculum 设置对 learned policy 的影响。

### 4.3 比较对象

本文主线固定为 progressive conservativeness curriculum，即在 `A_hard` 始终保留的前提下，调节 stronger layer 介入的阶段、阈值和视界。正文主比较对象固定为三组：

- `non_progressive`：固定使用 `recursive/H=1/eta=0.35`，表示不随训练阶段改变保守性介入强度的基线。
- `threshold_only_progressive`：early 阶段使用 `safe/H=1/eta=0.90`，主要停留在 `A_hard`；mid 和 late 阶段切换到 `recursive/H=1/eta=0.35`，即逐步引入 `A_rec`。
- `safeearly_progressive`：early 与 mid 阶段与 `threshold_only_progressive` 一致，但 late 阶段切入 `recursive/H=2/eta=0.55`，用于检验 late-stage stronger layer 是否自然带来更优 learned policy。

因此，`threshold_only_progressive` 是当前最稳的主正结果候选；`safeearly_progressive` 是 stronger late-stage layer 的消融/对照，而不是预设的更强成功版本。

H2 与 dual scheduling 当前作为边界结果处理。H2 对应更强的小视界 viability 检查；dual scheduling 进一步调整运行时风险阈值。当前证据不支持将二者写成正文主成功层。

### 4.4 评测口径与指标

正文主结果中的任务、安全和 gate 指标来自 `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`，口径为 `3 training seeds x 5 eval seeds x 5 episodes`。这些指标包括 `search_rate`、`coverage_ratio`、`collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate` 和 `recursive_gate_rate`。

正文主表中的 runtime 指标来自 `runs/progressive_mechanism_20260428/summary_metrics.csv`，包括 `perf_shield_time_ms` 与 `perf_recursive_time_ms`。使用 re-aggregated mechanism summary 的原因是，旧 formal profiling 中的部分 `perf_*` 字段与后续 mechanism summary 的聚合口径不完全一致；因此本文将任务/安全/gate 指标和 runtime 指标分开说明来源，以避免把不同 profiling 聚合口径混写成同一证据层级。

`episode_return` 不作为跨全部目录的主排序指标。原因是旧 baseline 与后续 `risk_base` 系列在 reward normalization 上存在历史口径差异，直接比较可能混入奖励尺度变化。正文主排序优先依赖任务、安全、gate 和 runtime 指标。

H2 边界结果主要来自 `runs/final_formal_h2_vs_h1_multiseed3x3/` 及相关 H1/H2 matched / cross-eval 结果；dual 边界结果来自 `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/`。这些结果用于 discussion 或 appendix，不进入 progressive 主线主表。

## 5. Results / Discussion

### 5.1 Progressive 主结果

表 1 给出 progressive conservativeness curriculum 的主结果。投稿级 LaTeX 表位于 `Paper/tables/progressive_main_table.tex`；下方 Markdown 表保留为中文草稿的可读版本。任务、安全和 gate 指标来自 formal compare summary；runtime 指标来自 re-aggregated mechanism summary。

| model | search_rate | coverage_ratio | collision_count | guarantee_broken_rate | dead_end_rec_rate | recursive_gate_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `non_progressive` | 0.9973 | 0.9989 | 90.79 | 0.3433 | 0.4640 | 0.2684 | 197.84 | 167.40 |
| `threshold_only_progressive` | 0.9987 | 0.9979 | 92.57 | 0.3324 | 0.4437 | 0.2760 | 238.13 | 202.96 |
| `safeearly_progressive` | 1.0000 | 0.9984 | 94.73 | 0.3543 | 0.4670 | 0.2657 | 192.30 | 162.81 |

表 1 支持的最稳妥结论是：`threshold_only_progressive` 是 mixed but useful improvement。与 `non_progressive` 相比，它的 `search_rate` 基本持平并略高，`guarantee_broken_rate` 从 `0.3433` 降至 `0.3324`，`dead_end_rec_rate` 从 `0.4640` 降至 `0.4437`。这些结果支持一个审慎结论：threshold-only progressive 调度在部分安全与 future-feasibility 指标上带来有限但有用的改善，但并不构成对 non-progressive 基线的全面支配。

同时，`threshold_only_progressive` 不是全面支配。它的 `collision_count` 更高，为 `92.57` 对 `90.79`；runtime 也更高，`perf_shield_time_ms` 为 `238.13ms` 对 `197.84ms`，`perf_recursive_time_ms` 为 `202.96ms` 对 `167.40ms`。因此本文不能写成 threshold-only progressive 全面优于 non-progressive，而应写成安全/可行性收益与碰撞、在线开销代价并存。

`safeearly_progressive` 达到最高 `search_rate=1.0000`，但其 `collision_count`、`guarantee_broken_rate` 和 `dead_end_rec_rate` 均不优于 `threshold_only_progressive`。由于该设置的主要差异是 late 阶段切入 `H=2/eta=0.55`，它更适合作为 late-stage stronger-layer 消融/对照，而不是更强成功版本。

### 5.2 Stage-level Mechanism Analysis

stage-level 统计进一步解释了 progressive curriculum 的实际运行方式。图 1 和表 2 来自 `runs/progressive_mechanism_20260428/stage_metrics.csv`，仅使用 `row_type=aggregate` 且 `split=eval` 的 stage-level 聚合行；它们展示不同训练阶段的 shield mode、horizon、threshold、gate 行为、dead-end 行为与 runtime。图 1 对应文件为 `Paper/figures/progressive_stage_mechanism.png` 和 `Paper/figures/progressive_stage_mechanism.pdf`，图注草稿位于 `Paper/figures/progressive_stage_mechanism_caption.md`。投稿级 stage 表位于 `Paper/tables/progressive_stage_mechanism_table.tex`。

图 1：Progressive conservativeness curriculum 的 stage-level 机制统计。early 阶段停留在 hard-safe / safe 层，几乎不触发 recursive gate；`threshold_only_progressive` 在 mid/late 阶段切入 `A_rec`；`safeearly_progressive` 在 late 阶段启用 `H=2` stronger layer，并在当期降低 recursive gate 与 recursive dead-end 指标，但这种 runtime filtering pattern 没有转化为 uniformly better final learned policy。

| model | stage | shield mode | horizon | threshold | recursive_gate_rate | dead_end_rec_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `non_progressive` | fixed | recursive | 1 | 0.35 | 0.2449 | 0.4549 | 175.87 | 147.49 |
| `threshold_only_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 31.41 | 0.00 |
| `threshold_only_progressive` | mid | recursive | 1 | 0.35 | 0.2477 | 0.4495 | 177.11 | 149.05 |
| `threshold_only_progressive` | late | recursive | 1 | 0.35 | 0.2473 | 0.4494 | 179.65 | 148.72 |
| `safeearly_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 30.91 | 0.00 |
| `safeearly_progressive` | mid | recursive | 1 | 0.35 | 0.2457 | 0.4397 | 178.46 | 150.32 |
| `safeearly_progressive` | late | recursive | 2 | 0.55 | 0.0507 | 0.1441 | 97.22 | 67.18 |

表 2 显示，progressive 的 early 阶段主要停留在 `A_hard` 层：`shield mode=safe` 时没有递归检查，`recursive_gate_rate=0`，`perf_recursive_time_ms=0`。`threshold_only_progressive` 在 mid/late 阶段切到 `A_rec`，并保持 `H=1/eta=0.35`；`safeearly_progressive` 则在 late 阶段切入 `H=2/eta=0.55`，运行时的 gate rate 和 dead-end rate 明显下降，但这一 stronger-layer runtime 行为没有稳定转化为更优 final learned policy。

因此，`threshold_only_progressive` 的收益不能简单写成 gate more。aggregate 主表中它的 `recursive_gate_rate` 只比 `non_progressive` 略高，而 `safeearly_progressive` 的 gate rate 更低却没有得到更好的主安全/可行性指标。已有 matched analysis 也只支持审慎说法：当前证据不支持“收益仅由 gate more / compute more 解释”，但还不能写成已经完全消除了 gate-rate 或 compute-budget confound。更准确的机制表述是，threshold curriculum 改变了训练期 stronger layer 的介入时机和分布；这种时序性改变与主表中的 mixed improvement 相容，但仍应被解释为经验机制证据，而不是完整因果证明。

### 5.3 Boundary Results and Supporting Diagnostics

H2 与 dual scheduling 只放在 boundary results / discussion 中。它们的共同作用不是证明一条新的主成功路径，而是支持一个更审慎的机制结论：stronger runtime filtering 与 better learned policy 并不天然等价。H2 是 `A_H^{viable}` 的自然扩展，在部分固定 checkpoint 或 runtime shield 设置下可以降低 guarantee-broken 或 dead-end 指标；但现有 H1/H2 formal、matched 和 cross-eval 结果不支持写成 H2 稳定优于 H1。`safeearly_progressive` 的主表结果也进一步显示，late-stage H2 stronger layer 没有自然变成全面更优的 learned policy。H2 的 appendix-ready 表格位于 `Paper/tables/appendix_h2_boundary_table.tex`。

dual scheduling 同样只能作为 mixed / boundary 结果。相对 `threshold_only_progressive`，`threshold_only_dual_progressive` 可以降低部分 runtime 开销，但在 `collision_count`、`guarantee_broken_rate` 与 `dead_end_rec_rate` 上更差。因此，dual 当前适合写成运行时阈值调度的候选方向或 appendix 对照，不能写成第二条成熟主创新。dual 的 appendix-ready 表格位于 `Paper/tables/appendix_dual_boundary_table.tex`。

exact/projected `A_hard` 诊断则作为理论底座和 appendix 支撑材料。它用于定义 grounded `A_hard` 参照语义，并区分 true dead-end 与 approximation-induced dead-end；但正文主线仍应围绕 progressive conservativeness curriculum 的三组主比较展开，不能让 exact solver 诊断抢走 progressive 主线。exact hard diagnostic 的 appendix-ready 表格位于 `Paper/tables/appendix_exact_hard_diagnostic_table.tex`，证据说明位于 `Paper/appendix_evidence_note.md`。

## 6. Limitations

本文当前结果需要主动承认以下局限。第一，`threshold_only_progressive` 是当前最稳的主正结果候选，但不是全面支配：它改善了 `guarantee_broken_rate` 与 `dead_end_rec_rate`，但 `collision_count` 与 runtime 不占优。

第二，H2 和 dual 都是 mixed / boundary 结果。H2 可以作为 stronger runtime layer 的候选，但不能写成稳定优于 H1；dual 可以降低部分运行时开销，但没有稳定改善主安全指标。因此二者不能升级为正文主成功分支。

第三，runtime 指标采用 re-aggregated mechanism summary 口径。本文主表明确将任务/安全/gate 指标来源于 `formal_compare_multiseed5x5/summary_metrics.csv`，将 runtime 指标来源于 `progressive_mechanism_20260428/summary_metrics.csv`，以避免旧 profiling 聚合口径差异。

第四，`episode_return` 不作为跨全部目录的统一主排序指标。由于历史实验之间 reward normalization 口径不完全一致，主结论应依赖任务、安全、gate 和 runtime 指标。

第五，matched analysis 还不是完整 frontier sweep。当前 matched gate-rate / compute-budget 分析足以支持“已有证据不支持收益仅由 gate more / compute more 解释”的审慎说法，但不能写成已经完全消除了所有 confound。

第六，exact/projected `A_hard` 诊断目前主要作为理论参照和 appendix 支撑材料。它有助于解释 sequential `A_hard` 近似、`sequential_with_exact_rescue` 边界纠偏，以及 true dead-end 与 approximation-induced dead-end 的区别；但这不是主环境大规模 exact proof，也不能写成在线主路径每步依赖 exact solver。

## 7. 结论（保守版）

本文将多无人机协同搜索中的 shield 重新组织为一个以 grounded `A_hard` semantics 为底座的分层 allowed-action framework。该框架强调 shield 的语义是 action-set filtering，而不是 planner takeover；并通过 `A_hard`、`A_rec` 与 `A_H^{viable}` 的层级关系，统一描述一步硬安全、递归可行性与小视界 future-feasibility。由此，progressive shielding 被重新解释为 conservativeness curriculum：hard-safe 底线始终保留，训练过程只调节 stronger layer 的介入方式。

基于当前实验结果，本文最稳妥的主张是：progressive conservativeness curriculum 在 hard-safe 始终保留的前提下，可以带来有限但可观察的安全/可行性收益。具体而言，`threshold_only_progressive` 相比 `non_progressive` 改善了 guarantee violation 与 recursive dead-end 指标，同时保持接近的 search performance；但该收益伴随更高 collision count 和 runtime，不能被写成全面支配。`safeearly_progressive`、H2 和 dual 的 mixed 结果进一步说明，更强 runtime filtering 并不自动带来更优 learned policy；这也是本文把 H2、dual 和 exact `A_hard` 诊断放在边界结果或支撑材料中的原因。
