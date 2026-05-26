# 引言初稿 v1（中文）

多无人机协同搜索（multi-UAV cooperative search）要求多个智能体在受限时间和受限观测条件下协同发现目标、完成区域覆盖，并同时规避威胁区域与机间碰撞。与单智能体安全强化学习相比，这类任务的困难不只在于“避免一次危险动作”，还在于如何在多机耦合约束下持续保留后续可行动作，避免系统被推进到局部拥挤、交换冲突或下一步无安全续行动作的状态。因而，在该类场景中，安全模块既需要提供硬约束层面的即时保护，也需要尽可能减少对策略学习的破坏，并控制在线计算开销。

现有关于 shielding 的研究已经表明，在执行前对动作进行安全过滤是一条重要路线；然而，针对多智能体协同搜索任务，仍有两个问题经常没有被讲清。第一，一部分工作在叙事上容易把 shield 描述成一个外部控制器，仿佛它直接替 actor 选择“最优安全动作”。这种表述虽然直观，却会模糊策略学习与安全过滤之间的职责边界。第二，关于 progressive shielding 的讨论常被简化为“训练前期弱安全、训练后期强安全”的 warmup 方案，但如果这种“弱安全”被理解为允许不安全动作通过，那么 hard-safe 约束在语义上就已经被放松。对于多无人机任务而言，这种表述尤其危险，因为训练早期同样可能遭遇高密度冲突和高风险接近状态。

本文围绕上述问题，采用一条更保守也更可防守的主线。我们不把 shielding 本身写成新发明，而是把多无人机协同搜索中的安全机制重新组织为一个以 exact / grounded `A_hard` 为底座的分层 shield 框架。这里的关键不在于由 shield 直接输出一个动作，而在于由 shield 构造允许动作集合（allowed action set），并约束 actor 只能从该集合中执行动作。换言之，shield 的语义是 allowed-action filtering，而不是 planner takeover。在这一语义下，actor 仍然保留对允许集合内部动作的偏好表达，shield 则负责把不可执行、不可持续或过于激进的动作排除在外。

基于这一视角，本文把安全过滤组织为三层：一步硬安全集合 `A_hard`、递归可行集合 `A_rec`，以及有限小视界可行集合 `A_H^{viable}`。其中，`A_hard` 对应一步层面的 hard-safe 底线，用于保证边界、威胁、碰撞和交换约束不会立即被破坏；`A_rec` 在 `A_hard` 基础上进一步排除那些会把系统立即推向下一步 dead-end 的动作；`A_H^{viable}` 则在更短的前瞻视界上维持 future feasibility。由此，shield 的作用不再只是一个普通的 action mask，也不是一个临时拼接的 heuristic shield。更准确地说，它是一个有明确层级语义的 allowed-set framework：`A_hard` 提供硬安全底座，`A_rec` 和 `A_H^{viable}` 在此基础上逐步收紧允许集合，以提升可持续安全性。

这一框架也改变了我们对 progressive / threshold curriculum 的理解。本文当前主线并不把 progressive 解释为“shield off -> on warmup”，而是把它解释为 conservativeness curriculum。也就是说，训练过程中始终保留 `A_hard` 这一 hard-safe 底座，progressive 调节的是是否、以及以多大强度，从 `A_hard` 升级到更保守的 `A_rec` 或 `A_H^{viable}` 层。风险阈值（threshold）在这里的作用，是决定 stronger layer 何时介入，而不是决定 hard-safe 是否存在。这样的解释与当前代码语义一致，也更符合多无人机安全场景中“底线约束不应在训练早期被关闭”的基本要求。

值得强调的是，本文并不试图讲述一个“过滤越强，策略就一定学得越好”的简单故事。当前已有结果恰恰提示，这种单调叙事并不稳固。就 progressive 主线而言，`threshold_only_progressive` 是当前最稳的主正结果候选，但其收益主要体现为部分安全/可行性指标的改善，而不是对所有任务指标、风险指标和在线成本的全面支配。进一步地，H=2 stronger layer 与 dual scheduling 虽然在部分设置下显示出候选价值，却尚未形成稳定、无争议的 learned policy 优势。这个现象提示我们：stronger runtime safety filtering 与 better learned policy improvement 之间存在非平凡错配。运行时过滤更强，可能改善即时安全性或 dead-end 避免，但也可能同时压缩探索空间、改变训练分布，进而削弱最终学到的策略质量。

因此，本文的目标不是再提出一个泛化意义上的 heuristic shield，而是把多无人机协同搜索中的 shield 重新组织为一个语义更清楚、分层关系更明确、结果解释更可防守的研究对象。具体来说，我们希望回答以下问题：在 hard-safe 必须始终保留的前提下，如何用一个 grounded 的 `A_hard` 底座来统一描述 allowed-action filtering；如何把递归可行性与小视界 viability 组织进同一分层框架；以及为什么训练期对保守性介入强度的调节有时有益，而更强的运行时过滤却不必然导向更优的 learned policy。

本文当前最可防守的贡献可以概括为以下四点。第一，我们提出并分析了一个以 exact / grounded `A_hard` 为底座的分层 allowed-action shield 框架，将多无人机协同搜索中的安全控制统一为允许动作集合的构造、收缩与重选问题。第二，我们明确给出 `A_hard`、`A_rec` 与 `A_H^{viable}` 的关系，并强调 shield 的语义是 filtering 而不是 takeover，从而澄清 actor 与 shield 的职责分工。第三，我们引入 exact/projected `A_hard` 的解释，并以 true dead-end 与 approximation-induced dead-end 区分顺序近似造成的错误空集与真实无解状态。第四，我们基于当前已经完成的实验结果，保守地指出 threshold curriculum 能带来有限但相对稳定的收益，同时更强运行时过滤并不天然等价于更优 learned policy；这一错配现象本身构成了本文的重要机制结论。若后续 matched frontier 结果继续支持，这一机制分析还可以被进一步加强，但当前版本已经足以支撑一版诚实、可迭代的中文初稿。
