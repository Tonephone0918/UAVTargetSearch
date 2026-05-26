# Related Work 初稿 v1（中文）

## 1. Shielded RL and Safe Action Filtering

安全强化学习（safe reinforcement learning）通常试图在策略优化过程中同时考虑任务收益与约束满足。常见路线包括约束策略优化、风险敏感目标、控制屏障函数，以及执行时的 shielding [@garcia2015safeRLsurvey; @achiam2017cpo; @ames2017cbf; @alshiekh2018shielding]。与只在奖励中加入安全惩罚的方法相比，shielding 的核心优势在于它可以在动作执行前对策略输出进行约束，从而把一部分安全责任从训练目标转移到运行时过滤机制上。

本文与这类工作共享“执行前过滤”的基本思想，但不把 shielding 本身作为新发明。本文关注的是在多无人机协同搜索中如何把 shield 的语义写清楚：shield 不是一个替 actor 直接选择最优动作的外部 planner，而是一个 allowed-action filtering 模块。actor 仍然先表达动作偏好；当原始动作不满足允许集合时，actor 只在允许集合内部重新选择。这个视角使得策略学习和安全过滤之间的职责边界更清楚，也为后续讨论 progressive / threshold curriculum 提供了稳定语义基础。

当前可引用方向：shielded RL、runtime shielding、safe action filtering 与 constrained policy execution。action masking 相关工作在第 5 节单独讨论。

## 2. Safe Multi-agent Reinforcement Learning

多智能体安全强化学习（safe multi-agent reinforcement learning, safe MARL）进一步引入了智能体间耦合约束。单智能体中可局部判断的安全动作，在多智能体场景下可能取决于其他 agent 的同步动作。例如，多 UAV 系统不仅需要规避威胁区域，还需要避免机间碰撞、交换位置冲突，以及局部拥挤导致的下一步无安全动作。现有 safe MARL 与 multi-agent shielding 工作通常关注如何在联合动作空间中维护安全，或如何将集中式安全约束分解到分布式执行中 [@gu2023safeMARL; @xiao2023modelBasedDynamicShielding]。CTDE 风格的多智能体学习骨架也为这类任务提供了常见训练范式 [@lowe2017maddpg; @yu2022mappo]。

本文与这类工作的边界在于：本文不主要解决 shield 的去中心化可实现性，也不声称提出一般形式的 multi-agent shielding 算法。本文聚焦于一个具体但有代表性的多无人机协同搜索任务，并把其安全约束组织为以 grounded `A_hard` 语义为底座的 allowed-action framework。在这个框架下，`A_hard` 不是若干局部规则的简单堆叠，而是以 exact/projected `A_hard` view 为参照来理解的硬安全允许集合；顺序近似、exact rescue 与 dead-end 诊断都围绕这一参照语义展开。

当前可引用方向：safe MARL、multi-agent shielding、centralized training with decentralized execution 下的安全约束。

## 3. Look-ahead, MPC-like Shielding, and Viability

另一类相关工作使用 look-ahead、model-predictive control（MPC）或 viability kernel 思想来保证未来可行性。它们通常不只检查当前动作是否立即安全，还会评估动作执行后是否存在未来安全 continuation [@aubin1991viability; @mayne2000mpc; @wabersich2021predictiveSafetyFilter]。这类方法与本文的 `A_rec` 和 `A_H^{viable}` 有明显关联：`A_rec` 可以理解为一步递归可行性检查，而 `A_H^{viable}` 则对应有限小视界的 future-feasibility 约束。

本文需要强调的区别是，当前方法不应被写成标准 MPC controller。标准 MPC 往往通过滚动优化返回单个动作，而本文的 shield 返回的是 allowed action set。即使在 `A_H^{viable}` 层使用小视界可行性推理，本文仍保留 actor-shield 的分工：shield 收缩可执行动作空间，actor 在允许集合内部表达偏好并完成重选。因此，本文更适合被表述为 policy-preserving、MPC-like shielding framework，而不是由 MPC 接管策略执行。

这一点也解释了本文为何强调 exact/projected `A_hard` view。多 UAV 联合动作可行性天然具有组合结构；单个 UAV 的某个动作是否可接受，取决于是否存在其他 UAV 的联合 completion。用 exact/projected view 作为参照，可以区分 true dead-end 与 approximation-induced dead-end，也可以解释为什么工程上需要顺序近似与局部 rescue 的折中。

当前可引用方向：MPC-style shielding、viability kernel、look-ahead safety filters、recursive feasibility。

## 4. Dynamic, Adaptive, and Progressive Shielding

动态或自适应 shielding 试图根据训练阶段、风险水平或运行状态调节安全模块的介入强度。相关工作通常讨论何时启用 shield、如何降低过度保守性、或者如何在探索效率和约束满足之间取得折中 [@waga2022dynamicShielding; @xiao2023modelBasedDynamicShielding]。

本文采用 progressive / threshold curriculum，但其含义需要与普通 warmup 区分。本文当前主线不是“训练前期 shield off、后期 shield on”，而是在 `A_hard` 始终保留的前提下调节更强过滤层的介入强度。换言之，progressive 调节的是 conservativeness，而不是 hard-safe 的存在与否。风险阈值的作用也不是决定是否允许不安全动作通过，而是决定是否从 `A_hard` 升级到 `A_rec` 或 `A_H^{viable}`。

这个定位对结果解释同样重要。当前证据并不支持“更强过滤必然带来更优 learned policy”的单调叙事。`threshold_only_progressive` 是当前最稳的主正结果候选，但其收益是 mixed improvement；`H=2` 与 dual 当前更适合作为边界结果或机制材料。因而，本文将 dynamic/progressive shielding 的重点放在“保守性如何影响学习结果”上，而不是把 stronger shield 直接等同于 better policy。

当前可引用方向：dynamic/adaptive shielding、risk-gated shielding、progressive safety constraints。严格意义上的 progressive conservativeness curriculum 对应文献还需要投稿前继续补强。

## 5. UAV Cooperative Search, MAPPO, and Action Masking

多无人机协同搜索任务通常结合多智能体协同、部分可观测、动态威胁、目标发现和覆盖控制等因素。multi-UAV deep reinforcement learning 的综述工作强调了可扩展协同、多机任务分配和分布式执行中的挑战 [@frattolillo2023multiUAVSurvey]。MAPPO 等 centralized training with decentralized execution 风格的方法为这类任务提供了常用学习基座 [@yu2022mappo]。另一方面，在离散动作空间中使用 action masking 也很常见：通过屏蔽不可执行动作，可以减少明显非法动作并稳定训练 [@huang2022invalidActionMasking]。

本文与普通 action masking 的区别在于，本文不只关心“哪些动作当前非法”，还关心 allowed set 的语义来源和层级关系。普通 action mask 往往只对应局部不可执行规则；本文则把 mask 组织为 `A_hard -> A_rec -> A_H^{viable}` 的分层 allowed-action set，并把 `A_hard` 放在 exact/projected view 的参照下理解。由此，dead-end 不再只是一个统计失败事件，而可以被拆分为 true dead-end 和 approximation-induced dead-end。

在实验层面，本文使用 MAPPO 作为策略学习基座，shield 作为集中式执行前过滤模块接入。这样的组合不是为了声称 MAPPO 本身的新贡献，而是为了在一个可复现实验骨架上分析 layered shield、threshold curriculum 与 learned policy 之间的关系。

当前可引用方向：UAV cooperative search / multi-UAV DRL、MAPPO、CTDE、discrete action masking。更贴近“目标搜索 + 动态威胁 + shielding”的具体 UAV 任务文献仍建议投稿前补强。

## 6. 本文与已有工作的边界

本文不声称发明 shielding，也不把 look-ahead feasibility 或 action masking 本身写成新概念。本文当前更可防守的差异在于四点。

第一，本文以 grounded `A_hard` semantics 为底座，并使用 exact/projected `A_hard` view 作为参照，避免把在线顺序近似误写成完整 exact solver。第二，本文把多无人机协同搜索中的安全过滤统一组织为 layered allowed-action framework，即 `A_hard`、`A_rec` 与 `A_H^{viable}` 的层级收缩。第三，本文用 true dead-end 与 approximation-induced dead-end 诊断顺序近似造成的误差来源。第四，本文把当前 mixed 结果作为机制分析对象，强调 stronger runtime safety filtering 与 better learned policy improvement 并不天然等价。

因此，本文的定位不是“又一个 heuristic shield”，而是在一个多无人机协同搜索任务中，给出一套语义清楚、层级可解释、结果边界诚实的 shield 写法与实证分析框架。
