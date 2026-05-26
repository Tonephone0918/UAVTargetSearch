# 方法初稿 v1（中文）

## 1. 问题定义

考虑一个多无人机协同搜索环境。设时刻 `t` 的全局状态为 `s_t`，每个 UAV 基于局部观测 `o_t^i` 由策略网络输出动作偏好，联合动作记为 `a_t = (a_t^1,\dots,a_t^n)`。环境转移由 `f(s_t, a_t)` 给出，状态空间中满足边界、威胁规避、机间安全距离以及交换冲突约束的集合记为 `\mathcal S_{\mathrm{safe}}`。本文关心的问题不是单纯在学习目标中加入一个安全惩罚，而是在执行前通过 shield 对动作空间进行结构化收缩，使最终执行动作满足 hard-safe 约束，并尽量避免把系统推进到未来无安全续行动作的状态。

在这一设置下，一步硬安全联合动作集可记为

\[
A^{\mathrm{safe}}(s_t)=\{a_t \in A \mid f(s_t,a_t)\in \mathcal S_{\mathrm{safe}}\}.
\]

如果只把安全问题理解为“当前动作执行后不立刻越界或碰撞”，那么只需判断动作是否属于 `A^{\mathrm{safe}}(s_t)`。但对于多无人机协同搜索，仅有一步 hard-safe 往往不足以保证持续可执行性：一个动作虽然当前安全，却可能让下一步所有动作都不可行，进而形成 dead-end。本文因此采用一个分层 allowed-action 框架，把“当前安全”和“未来仍可持续安全”分开建模。

## 2. Shield 作为 allowed-action filtering 的语义

本文不把 shield 定义为一个直接替 actor 选择动作的外部规划器，而把它定义为一个允许动作集过滤器。给定当前状态 `s_t`，shield 返回一个允许动作集合

\[
\mathcal A_t^{\mathrm{allow}}(s_t)\subseteq A^{\mathrm{safe}}(s_t).
\]

actor 先输出原始策略 `\pi_\theta(a\mid o_t)`，若原始提议动作已经属于 `\mathcal A_t^{\mathrm{allow}}(s_t)`，则该动作直接执行；否则 actor 在允许集合上进行 masked re-selection，即仅在允许动作内部重新归一化并选择动作。于是，最终执行动作始终来自允许集合，但对允许集合内部的相对偏好仍由 actor 决定。这样的语义有两个直接好处。其一，策略学习的主体地位得到保留，shield 不会退化为 planner takeover。其二，不同强度的安全逻辑都可以统一为“allowed set 的进一步收缩”，从而自然形成层级结构。

这一语义也对应了当前代码实现中的核心分工：shield 负责计算 `A_hard` 及其更强子集，并在原动作不允许时触发重选；而不是用一个外部优化器在每一步直接替换 actor 的决策。`TODO`：最终稿可补一个形式化命题，说明只要最终动作始终从 `\mathcal A_t^{\mathrm{allow}}(s_t)` 中选择，且该集合包含于 `A^{\mathrm{safe}}(s_t)`，则可得到一步 hard safety。

## 3. 分层允许动作集：`A_hard`、`A_rec` 与 `A_H^{viable}`

在本文中，允许动作集按照保守性强弱分为三层：

\[
A_hard(s_t), \qquad A_{rec}(s_t), \qquad A_H^{viable}(s_t).
\]

其中，`A_hard` 是 always-on 的一步硬安全底座。它只要求动作在一步转移后仍满足硬安全约束，因此是整个框架最宽、也最基础的 allowed set。`A_rec` 在 `A_hard` 的基础上加入一步递归可行性（recursive feasibility）约束，只保留那些执行后下一步仍存在至少一个硬安全续行动作的动作。`A_H^{viable}` 则把这一思想推广到有限小视界 `H`，要求存在一个长度为 `H` 的安全 continuation，使得系统在短视界内持续保持可行。

若暂时忽略近似误差，则三者满足自然的包含关系：

\[
A_H^{viable}(s_t)\subseteq A_{rec}(s_t)\subseteq A_{hard}(s_t).
\]

这条包含链构成了本文方法的理论骨架。`A_hard` 负责 hard-safe 底线，`A_rec` 和 `A_H^{viable}` 负责更强的可持续安全性。换言之，本文并不是在不同阶段切换“安全”与“不安全”，而是在硬安全始终保留的前提下，决定是否进一步收缩到更保守的 allowed set。

## 4. `A_hard` 的 exact/projected 语义

虽然 `A_hard` 在实现中常以逐 agent 的局部规则方式在线构造，但从理论上讲，它对应的是一个更基本的联合可行性对象。给定某个 agent `i` 的候选动作 `a_i`，若存在其他 agent 的一个联合动作 completion，使得整个联合动作满足一步硬安全，则 `a_i` 应被视为对 agent `i` 可接受。由此可以定义 exact/projected 的 `A_hard` 语义：

\[
A_{hard,i}^{\star}(s_t)
=
\{a_i \mid \exists a_{-i},\ (a_i,a_{-i}) \in A^{\mathrm{safe}}(s_t)\}.
\]

这个定义强调，单个 UAV 的“允许动作”并不是只看它自身是否安全，而是看该动作是否能够嵌入某个联合硬安全 completion。因而，exact `A_hard` 实际上是联合可行集在单 agent 上的投影（projection），而不是简单的局部规则拼接。

这一点对于多无人机场景尤其关键。某个动作在局部上看似无碰撞，但若它与其他 UAV 的可行动作组合不存在任何联合 completion，那么它就不应被视为真正的 admissible action。当前代码中的顺序式 `A_hard` 构造是这一 exact/projected 语义的近似实现：它使用固定裁决顺序和局部约束快速生成一个工程上足够高效的允许集合，但并不保证与 exact projected set 完全一致。

## 5. true dead-end 与 approximation-induced dead-end

一旦引入 exact/projected `A_hard` 的视角，dead-end 的含义就需要进一步细分。若某个状态下 exact projected `A_hard` 本身为空，那么系统处于真实无解的 true dead-end：不存在任何联合 completion 可以保证下一步硬安全。相反，若 exact projected `A_hard` 非空，但顺序近似或局部裁决过程仍返回空集，则该 dead-end 不是系统本身无解，而是由近似误差诱发的 approximation-induced dead-end。

这一区分不仅是理论上的细节，也直接影响实现与结果解释。当前实现中的 `sequential_with_exact_rescue` 可以理解为一种边界纠偏机制：在线主路径仍以顺序近似作为默认求解器，只在顺序近似返回空集或极小候选集时，调用 exact witness / rescue 来检查是否存在被误删的可行动作。因而，exact 求解器在本文中的角色更接近研究型 oracle，而不是每步都要替代在线规则的完整求解器。

对论文写作而言，这一视角至少提供了两点价值。第一，它把 dead-end 从一个模糊的“失败事件”拆分为“真实无解”和“近似误删”两种来源，使得 `A_hard` 的误差可以被清楚讨论。第二，它解释了为什么本文不能把当前实现简单称为普通 action mask：真正有意义的对象是 grounded 的 allowed-action set，而非若干独立规则的并置。`TODO`：最终稿可补充一个更正式的误差定义，例如对 false empty rate 和 projected-set disagreement 的形式化记号。

## 6. Progressive / threshold curriculum 的设计动机

在上述分层框架中，training-time curriculum 的核心问题不再是“何时开启安全”，而是“何时把 `A_hard` 升级为更保守的子集”。因此，本文把 progressive / threshold curriculum 解释为 conservativeness curriculum：`A_hard` 始终存在，而训练过程只是在不同阶段调节 `A_rec` 或 `A_H^{viable}` 的介入强度。当前代码主线下，风险阈值负责决定 stronger layer 是否触发；progressive schedule 则进一步决定在训练 early、mid、late 阶段使用什么模式、阈值和视界。

这样的设计至少有三层动机。第一，若在训练早期就长期使用过强过滤，actor 可能在过窄的允许动作空间内学习，从而损失探索与表征能力。第二，若完全只依赖 `A_hard`，虽然能保持一步安全，却可能频繁把系统推进到 future-feasibility 更差的状态。第三，多无人机协同搜索中 stronger layer 的在线代价不可忽视，因此更合理的做法是把它当作一种选择性介入机制，而不是全时段、全状态的默认求解器。

因此，threshold curriculum 当前最稳妥的解释是：它通过训练过程中的阈值化调度，控制 stronger layer 对 actor 的影响方式和影响频率，而不是放弃 hard-safe。当前可保守表述为：训练期对保守性注入强度的重新组织，可以带来部分安全/可行性指标上的有限改善；已有 matched gate-rate / compute-budget 证据不支持把该收益简单归因于 gate more 或 compute more，但还不能写成完整消除了这些 confound。

## 7. 当前主线为什么不是“shield off -> on warmup”

本文有意避免把主线写成“训练前期 shield off、后期 shield on”的 warmup 方案，原因有三。第一，这种叙事与当前代码不符。当前 progressive 设计仍然保留 `A_hard` always-on，只是在 `A_hard` 之上调节是否升级到 `A_rec` 或 `A_H^{viable}`。第二，这种叙事在理论上也不干净。若训练前期允许明显不安全动作通过，那么 hard-safe 约束就不再是始终有效的底线，后续再宣称方法满足硬安全将出现自相矛盾。第三，对于多无人机场景，训练早期本身就是高冲突、高混乱阶段，更不适合通过“暂时取消 hard-safe”来换取学习便利。

因此，本文更准确的写法应是：progressive 不是安全开关，而是保守性调度器；它调节的是 stronger filtering 在训练中的注入方式，而不是 hard-safe 是否存在。这个区分既是方法语义的关键，也是当前论文叙事与既有经验结果能够对齐的前提。

## 8. `H=2` 与 dual 在当前方法叙事中的位置

在方法层面，`H=2` 与 dual 都是自然的扩展层。`H=2` 对应把 `A_H^{viable}` 从 `H=1` 推广到更强的小视界 viability 检查；dual 则是在 progressive 基础上进一步对风险阈值进行运行时调节，试图在介入强度和在线成本之间做更细粒度折中。从框架完整性看，它们都属于本文 layered shield 研究线的一部分。

但在当前稿件中，这两者不应被写成已经成熟的主成功层。更准确的定位是：`H=2` 展示了 stronger runtime layer 的方法扩展方向，dual 展示了更复杂调度的潜在空间；然而现有结果对它们的支持仍是 mixed。因而，本轮中文初稿只需要把它们写成扩展层或边界层，说明该框架并不排斥更强视界和更复杂调度，但主文叙事仍以 `A_hard` 底座、`A_rec` 升级和 `threshold_only_progressive` 为核心。

在当前版本中，不建议把 `H=2` 或 dual 升格为主方法创新点；它们应服务于 boundary discussion，即说明 stronger runtime filtering 与 better learned policy 之间不存在简单单调关系。
