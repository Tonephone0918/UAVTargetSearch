# 论文初稿骨架 v1（中文）

## 1. 题目候选

1. 面向多无人机协同搜索的分层 Allowed-Action Shield：基于精确 `A_hard` 底座的渐进保守性课程与机制分析
2. 多无人机协同搜索中的分层安全动作过滤：从精确 `A_hard` 到渐进式保守性调度
3. 基于精确硬安全底座的多无人机协同搜索 Shield 框架：Allowed-Action Filtering 与学习机制分析

`TODO`：最终标题需要在“方法框架导向”和“机制分析导向”之间二选一。若投稿更看重系统化方法，可用题目 1；若更看重经验洞察，可把副标题进一步强化为“为什么 stronger filtering 不必然带来 better learned policy”。

## 2. 中文摘要（初稿）

多无人机协同搜索（multi-UAV cooperative search）中的安全强化学习既需要维持探索与搜索效率，又必须满足无人机间避碰、威胁规避和在线可执行性约束。现有工作常将 shield 视为一个固定启用的安全模块，或者在表述上将其近似为外部 planner 对策略动作的直接替换，这使得 hard-safe 语义、递归可行性（recursive feasibility）与训练期保守性调度之间的关系不够清晰。本文从一个更保守也更可防守的角度出发，将该问题重构为一个以精确一步硬安全集合 `A_hard` 为底座的分层 allowed-action shield 框架，其中 shield 的语义是 allowed-action filtering，而不是直接为 actor 选择“最优安全动作”。在此基础上，我们进一步组织出 `A_hard`、`A_rec` 与 `A_H^{viable}` 三层允许动作集，并把 progressive shielding 解释为保守性课程（conservativeness curriculum），即在 hard-safe 始终保留的前提下，按训练阶段或风险阈值调节更强过滤层的介入强度。

实验上，我们基于当前已有的 MAPPO 训练与评测结果，重点考察 `non_progressive`、`threshold_only_progressive` 与 `safeearly_progressive` 三条主线，并辅以 H1/H2 formal compare、matched gate-rate / compute-budget 诊断以及 H1/H2 checkpoint-shield `2x2` cross-eval。当前证据显示，`threshold_only_progressive` 是最稳妥的主正结果候选：它在 `guarantee_broken_rate` 与 `dead_end_rec_rate` 上相对 `non_progressive` 有所改善，但在 `collision_count` 与在线开销上并不形成全面支配。进一步地，H2 与 dual scheduling 当前更适合作为边界结果和机制反例，因为更强的运行时过滤虽然在部分设置下能降低 dead-end 或 guarantee-broken 指标，却没有稳定转化为更优的 learned policy。因而，本文的贡献不仅是一个分层 shield 框架，还包括对“shield-strength / learning-outcome 错配”现象的初步机制分析。

`FIXME`：摘要中的“当前证据显示”应在最终版本中替换为更标准的论文措辞，但要保留 mixed result 的诚实表述。

## 3. 引言初稿

多无人机协同搜索任务要求多个智能体在有限时间内协同覆盖未知区域、发现目标并规避动态威胁。在这一类任务中，安全约束不是可选附加项，而是直接决定策略是否具备部署意义的基本条件。与单智能体设置不同，多无人机场景中的安全约束不仅来自单机与威胁之间的距离限制，还来自多机之间的相互碰撞、交换位置（swap）以及局部拥挤引发的递归可行性退化。因此，一个能够兼顾安全、任务完成率与在线成本的 shield 机制，对于该类任务具有直接意义。

然而，现有关于 shield 的叙事存在两个常见问题。其一，一部分工作默认把 shield 理解为“外部安全控制器替 actor 选动作”，从而模糊了策略学习与安全过滤的职责边界。其二，关于 progressive shielding 的直觉性表述，常被写成“训练前期弱安全、后期强安全”的 warmup 方案，但这种表述在 hard-safe 场景下容易引入一个根本性漏洞：如果 early stage 真的允许不安全动作通过，那么后续再强调 hard-safe 就会变得不一致。对于多无人机协同搜索而言，这一漏洞尤其明显，因为训练早期同样可能遭遇高密度碰撞和威胁接近状态。

本文因此采用一条更保守、也更贴近当前代码与结果证据的主线。我们不把 shielding 本身写成新发明，而是把多无人机协同搜索中的 shield 重新组织为一个以精确一步硬安全集合 `A_hard` 为底座的分层 allowed-action framework。具体地，shield 在每个时刻计算一个允许动作集合（allowed action set）；actor 首先给出原始偏好，若其原动作不在允许集合内，则 actor 仅在允许集合内部重选。由此，shield 的作用是过滤（filtering）而不是接管（takeover）。在这一语义下，我们把允许集合进一步分解为一步硬安全层 `A_hard`、递归可行层 `A_rec` 和小视界可行层 `A_H^{viable}`，并将 progressive 重新解释为在 hard-safe 始终保留前提下，对更强保守层级的课程式调度。

值得强调的是，本文并不试图讲述一个“更强 shield 一定更好”的简单故事。当前已有结果恰恰表明，这种故事并不成立。正式比较中，`threshold_only_progressive` 相比 `non_progressive` 在部分安全相关指标上更优，但不形成对 `collision_count`、`search_rate` 和运行时成本的全面支配；H2 的 validate-only 工作点与 runtime cross-eval 显示更强过滤层本身可能有价值，但相应训练得到的 H2 checkpoint 尚未稳定优于 H1 checkpoint。这说明 stronger runtime filtering 与 better learned policy improvement 之间存在非平凡错配。本文将这一错配视为需要正面分析的经验事实，而不是需要被掩盖的叙事噪声。

本文当前最可防守的贡献有三方面。第一，我们形式化并实现了一个 grounded 的 layered allowed-action shield 框架，其中 `A_hard` 具有 exact / projected 语义，并可与 `A_rec`、`A_H^{viable}` 统一描述。第二，我们用 exact feasibility 诊断视角区分了 true dead-end 与 approximation-induced dead-end，从而为顺序近似 shield 的误差分析提供了明确对象。第三，我们用 progressive compare、H1/H2 cross-eval 与 matched gate-rate / compute-budget 诊断说明：训练期 threshold curriculum 可以带来相对稳定但有限的收益，而更强的未来过滤并不会自动转化为更优的 learned policy。这些发现共同构成了本文的主线。

## 4. 方法总览

本文的方法建立在 MAPPO 训练基座之上，安全模块采用集中式 shield（centralized safety shield）。在每个决策时刻，actor 根据局部观测输出动作偏好，shield 基于当前全局状态构造允许动作集合，并根据当前启用的层级选择 `A_hard`、`A_rec` 或 `A_H^{viable}` 作为最终允许集合。若 actor 的提议动作属于该集合，则动作直接执行；否则 actor 在允许集合内进行 constrained re-selection。由此，策略学习仍然保留在 actor 侧，而 shield 只承担“收缩可执行动作空间”的职责。

在计算层次上，`A_hard` 对应一步硬安全过滤，用于保证边界、威胁、机间距离和 swap 约束；`A_rec` 在 `A_hard` 的基础上进一步要求当前动作不会把系统立即推进到“下一步无安全动作”的状态；`A_H^{viable}` 则将这一 future-safe 检查推广到有限小视界 `H`。当前主线中，`A_hard` 始终保留，而 stronger layer 是否运行由 mode、risk threshold 和 progressive stage 决定。因而，本文的关键问题不是“要不要 shield”，而是“在 hard-safe 永远存在的前提下，何时值得升级到更保守、也更昂贵的 future-feasible 过滤层”。

## 5. 问题定义与分层 shield 框架

设系统状态为 `s_t`，联合动作空间为 `A`，环境转移为 `f(s_t, a_t)`。我们首先定义一步硬安全动作集：

\[
A^{\mathrm{safe}}(s_t)=\{a_t \in A \mid f(s_t,a_t)\in \mathcal S_{\mathrm{safe}}\}.
\]

与将 shield 理解为“输出一个最优安全动作”的做法不同，本文把 shield 定义为一个允许动作集过滤器：

\[
\mathcal A_t^{\mathrm{allow}}(s_t)\subseteq A^{\mathrm{safe}}(s_t).
\]

actor 先输出原始策略偏好 `\pi_\theta(a\mid o_t)`；若原动作不在 `\mathcal A_t^{\mathrm{allow}}(s_t)` 中，则 actor 在该集合上进行 masked re-selection。这样的定义有两个好处。第一，它保留了策略学习的主体地位，shield 不会退化为 planner takeover。第二，它允许我们把不同强度的安全逻辑统一为“allowed set 的进一步收缩”，从而自然形成分层结构。

在此基础上，本文采用如下三层结构：

\[
A_hard,\qquad A_{rec},\qquad A_H^{viable}.
\]

其中，`A_hard` 是 always-on 的一步硬安全底座；`A_rec` 只保留那些在执行后仍存在至少一步安全续行动作的动作；`A_H^{viable}` 则进一步要求存在长度为 `H` 的安全 continuation。按照这一记号，若忽略近似误差，应有

\[
A_H^{viable}(s_t)\subseteq A_{rec}(s_t)\subseteq A_{hard}(s_t).
\]

这一包含链是全文的理论骨架，也是后续讨论 progressive / threshold curriculum 的基础。因为一旦 `A_hard` 被固定为底线，progressive 的含义就不再是“是否安全”，而是“是否在当前时刻进一步收缩到更保守的子集”。

## 6. `A_hard` 的 exact / projected 语义与 dead-end 诊断视角

当前实现中，`A_hard` 的在线路径默认依赖顺序式（sequential）近似：系统按某种裁决顺序依次处理各个 UAV，并结合局部规则快速构造一步硬安全动作集合。这一路径在工程上足够高效，但它并不等价于精确联合可行性（exact joint feasibility）的投影结果。为此，本文引入 exact projected `A_hard` 的语义：对于 agent `i` 的动作 `a_i`，若存在其他 agent 的一个联合动作 completion 使得整体一步硬安全成立，则 `a_i` 属于 `A_{hard,i}^{\star}(s_t)`。换言之，exact `A_hard` 是“联合可行集在单个 agent 上的投影”，而不是局部规则的简单并置。

这一区分带来一个对本文非常重要的诊断视角：dead-end 并不只有一种来源。若 exact projected set 本身为空，则它是 true dead-end；若 exact projected set 非空，但顺序近似返回空集，则它是 approximation-induced dead-end。当前代码中的 `sequential_with_exact_rescue` 正是围绕这一差异构造：在线主路径先用顺序近似快速过滤，只在近似返回空集或极小候选集时调用 exact witness / rescue。这样一来，exact 不必充当每步在线求解器，而更像一个“对顺序近似进行边界纠偏的研究型 oracle”。

现有诊断结果支持这种写法。在较小的 exact-solver 诊断设置中，顺序集与 exact 集之间存在明显偏差：`seq_empty_exact_nonempty_rate` 大致落在 `0.14` 到 `0.37`，`seq_nonempty_exact_empty_rate` 大致落在 `0.24` 到 `0.42`，`seq_exact_jaccard` 也仅在约 `0.59` 到 `0.72` 之间。这说明顺序近似既可能制造“假空集”，也可能保留一些并不对应 exact projected admissibility 的动作。另一方面，rescue 机制能在有限额外开销下缓解这一问题。例如在 `model_compare_exact_hard_solver_fast2x1` 中，`safe_rescue` 和 `recursive_full_rescue` 相比纯 sequential 或 always-on exact 均表现出更低的 `collision_count` / `guarantee_broken_rate` 与更低的 exact 求解时间。`TODO`：正文最终应补充这些诊断目录的统一表格与误差定义。

## 7. progressive / threshold curriculum 的设计动机与机制

本文不采用“训练前期 shield off、训练后期 shield on”的叙事。当前代码与实验更一致的解释是：hard-safe 始终保留，而课程学习只作用于更强过滤层的启用方式。具体而言，`MAPPOTrainer` 中的 progressive schedule 会根据训练进度，把运行时配置在 early、mid、late 三个阶段之间切换。当前主线的两条 progressive 分支共享相同的前半段：early 阶段使用 `safe` 模式、`H=1`、高阈值 `\eta=0.9`，此时本质上保留 `A_hard` 而不做递归升级；mid 阶段切换到 `recursive`、`H=1`、`\eta=0.35`。两者的关键差异出现在 late 阶段：`threshold_only_progressive` 保持 `recursive/H=1/\eta=0.35`，而 `safeearly_progressive` 则进一步切换到 `recursive/H=2/\eta=0.55`。

这一设计恰好支持本文要强调的主叙事。首先，progressive 并不放弃 hard-safe；相反，early 阶段只是减少 stronger future-safe 层的介入，让训练先在较宽但仍安全的 `A_hard` 集合内学习。其次，`threshold_only_progressive` 的稳定性表明，当前最可防守的收益并不来自 horizon 扩张，而更可能来自训练期对递归过滤介入时机的重新组织。再次，`safeearly_progressive` 的 mixed 结果意味着“更晚阶段上 H2”并没有自然带来更优 learned policy，这与后文的 H1/H2 cross-eval 诊断相呼应。

`FIXME`：风险函数的正文写法需要谨慎。当前代码默认使用 `risk_base` 变体，其分量更接近 `proposed-clearance + clear-gap + support + region`，而早期理论文档仍保留了 `clear + region + hist` 的更简化表述。建议正文主叙事先把 risk 写成“用于触发 `A_hard -> A_rec` 升级的轻量阈值化风险分数”，把具体分量放到实验设置或 appendix，避免在当前初稿阶段过度绑定某个 risk 配方。

## 8. 实验设置初稿

本文当前实验采用 MAPPO 作为训练基座，shield 模块以集中式 pre-execution filtering 的方式接入决策流程。主结果以当前已经完成的正式 compare 为准，其中 progressive 主线的默认正式口径为 `3 training seeds x 5 eval seeds per checkpoint x 5 episodes per eval seed`，评测设备为 CPU。对于 H1/H2 边界结果，当前同时存在 `3x3` 与 `5x5` 两类比较；此外还包含 matched gate-rate / matched compute-budget、公平 cross-eval，以及若干 exact hard-solver 诊断目录。

指标方面，本文不建议把 `episode_return` 作为跨全部目录的主排名标准。原因是旧 baseline 与后续 `risk_base` 系列在 DPM reward normalization 口径上并不完全一致，因此回报尺度并非始终可直接比较。当前更稳妥的主指标应包括：任务指标 `search_rate` 与 `coverage_ratio`；安全指标 `collision_count`、`guarantee_broken_rate`；约束行为指标 `dead_end_hard_rate`、`dead_end_rec_rate`、`recursive_gate_rate`；以及在线开销指标 `perf_shield_time_ms`、`perf_recursive_time_ms`。`TODO`：最终稿需补充环境规模、UAV 数、threat 数、最大步长、checkpoint 选择规则以及是否使用 `best.pt` 的明确定义。

还需要说明的是，当前工作区中的包名仍沿用 `src/hrvdn`，但主实验已经切换到 MAPPOTrainer 和 MAPPO checkpoints。这一“代码目录名与实验骨干不一致”的历史遗留问题，在最终论文中需要通过实验设置段落或脚注明确解释，以免造成“本文是 HRVDN 还是 MAPPO 主线”的歧义。

## 9. 当前结果总结初稿

在 progressive 主线的正式比较中，`threshold_only_progressive` 是目前最可防守的主结果候选，但其收益应明确写成 mixed improvement，而不是全面胜利。以正式 `3x5x5` 口径为例，相比 `non_progressive`，`threshold_only_progressive` 的 `search_rate` 基本持平（约 `0.9987` 对 `0.9973`），`guarantee_broken_rate` 更低（约 `0.332` 对 `0.343`），`dead_end_rec_rate` 也更低（约 `0.444` 对 `0.464`）；但其 `collision_count` 并未同步降低（约 `92.57` 对 `90.79`），且运行时开销更高（re-aggregated raw 口径下 `perf_shield_time_ms` 约 `238ms` 对 `198ms`）。这说明 threshold curriculum 的收益更像是“在部分 future-feasibility 指标上更稳”，而不是对任务、安全和开销的统一支配。

`safeearly_progressive` 当前不宜写成主正结果。它在 `search_rate` 上达到 `1.0`，但 `collision_count`、`guarantee_broken_rate` 和 `dead_end_rec_rate` 没有形成相对 `threshold_only_progressive` 的一致改进。结合阶段 schedule 可以看到，`safeearly_progressive` 与 `threshold_only_progressive` 的主要差异出现在 late stage 是否切入 `H=2/\eta=0.55`。因此，当前最自然的解释不是“safeearly 更强所以应该更好”，而是“late-stage H2 升级并未稳定转化为更优 learned policy”。

H1/H2 边界结果进一步强化了这一判断。在 validate-only 的 `h2_risk_threshold_scan_medium2x2` 中，`H2@eta≈0.55` 确实显示出值得注意的 runtime 候选信号，例如它在该设置下具有更低的 `collision_count`、`guarantee_broken_rate` 和 `dead_end_rec_rate`。但一旦进入正式训练后的 learned checkpoint compare，H2 的表现就转为 mixed。例如在 `final_formal_h2_vs_h1_multiseed3x3` 中，`recursive_risk_rescue_h2_eta055` 的 `guarantee_broken_rate` 与 `dead_end_rec_rate` 低于 H1，但 `search_rate` 与 `collision_count` 并没有同步改善；在 `5x5` 口径下，H2 的部分安全指标更好，但 `search_rate` 仍未稳定优于 H1。换言之，H2 不是没有价值，而是其价值尚未被闭环训练稳定吸收。

最关键的机制证据来自 `H1/H2 checkpoint x H1/H2 shield` 的 `2x2` cross-eval。当前最强组合不是 `H2 ckpt + H2 shield`，而是 `H1 ckpt + H2 shield`：该组合在 `3x3` compare 中达到 `search_rate=1.0`，同时 `collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate` 与 `near_miss_rate` 都优于 `H1 ckpt + H1 shield`。相反，`H2 ckpt + H1 shield` 并未优于 `H1 ckpt + H1 shield`。这意味着当前收益更像来自“把更强的 H2 shield 作为 runtime layer 挂在更成熟的 H1 policy 上”，而不是“H2 training 已经学出更优 policy”。这正是本文想强调的 stronger runtime filtering 与 better learned policy 之间的错配现象。

最后，dual scheduling 当前也不宜进入正文主成功故事。在 `formal_compare_with_dual_multiseed5x5` 中，`threshold_only_dual_progressive` 的运行时开销确有下降，但 `collision_count`、`guarantee_broken_rate` 与 `dead_end_rec_rate` 明显不如 `threshold_only_progressive`。因此，dual 更适合被放在当前稿件的边界结果或 discussion 中，用来说明“调度更复杂并不自动带来更好的 learned outcome”。

## 10. 讨论与局限性初稿

本文当前最重要的讨论不是“我们是否已经找到一个全面更强的 shield”，而是“为什么 grounded `A_hard` 与 stronger layer 的组织方式，会影响训练得到的 policy 质量”。从现有证据看，`A_hard` 作为 always-on 底座是有必要的，因为它为整个系统提供了统一的 hard-safe 语义；exact projected `A_hard` 与 rescue 诊断也说明，顺序近似的误差是真实存在的。然而，更强的 future-feasible 过滤层一旦介入训练闭环，其收益并不会简单地随着过滤强度单调上升。H2 与 dual 的 mixed 结果说明，运行时更强并不意味着策略学得更好，这可能与探索空间收缩、训练分布偏移、风险触发稀疏性以及 actor 对 mask 的适应方式有关。

当前稿件也存在几个明确局限。第一，主线收益目前主要集中在 `threshold_only_progressive`，而不是 `H=2` 或 dual；因此本文不能写成“更长 horizon 带来稳定收益”的论文。第二，不同历史 compare 目录之间的 reward normalization 口径并不完全一致，因此 `episode_return` 只能作为辅助指标。第三，关于 risk 的理论描述与当前代码默认 `risk_base` 之间仍有表达层面的张力，这需要在最终稿中进一步统一。第四，exact-solver 诊断目前主要集中在较小设置或专门的对角线诊断目录，尚未在全部主环境规模上形成同等系统的分析。`TODO`：如 codex1 后续能补一组与正文主环境完全同口径的 exact-diagnostic summary，这一节会更完整。

综合来看，当前稿件最适合主张的是：本文提供了一个以 grounded `A_hard` 为底座的 layered allowed-action shield 框架，并基于现有实证结果说明了 progressive threshold curriculum 的有限但稳妥收益，以及 stronger runtime filtering 与 learned policy improvement 之间的非平凡错配。相反，任何关于“H2 已经稳定成功”“dual 已经完成闭环”或“本文显著优于所有 baseline”的表述，当前都不应写得太满。
