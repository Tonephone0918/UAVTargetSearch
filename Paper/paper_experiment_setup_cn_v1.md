# Experimental Setup 初稿 v1（中文）

## 1. 环境与任务

本文实验基于多无人机协同搜索环境。环境中多个 UAV 在离散地图上协同搜索目标，并需要同时规避威胁区域、机间碰撞与交换位置冲突。当前代码默认配置为 `20 x 20` 地图、`10` 架 UAV、`10` 个目标、`5` 个动态威胁，最大 episode 长度为 `120` 步；UAV 间安全距离为 `1`，威胁安全距离为 `2`。目标与威胁均可具有动态行为，威胁位置按设定周期更新。`TODO: 待 codex1 确认正式 formal compare 是否完全使用默认环境参数，或是否存在命令行覆盖。`

任务目标是尽可能发现目标并保持较高覆盖，同时避免违反硬安全约束。由于多 UAV 动作同时执行，单个 UAV 的局部安全动作并不一定对应联合安全动作；因此实验特别关注 shield 对联合安全、递归可行性和 dead-end 行为的影响。

## 2. 策略学习基座

本文使用 MAPPO 作为策略学习基座。actor 根据局部观测输出动作偏好，critic 使用集中式信息进行训练。本文不把 MAPPO 本身作为方法创新点，而是将其作为稳定的多智能体强化学习骨架，用于评估不同 shield 语义和 curriculum 设置对 learned policy 的影响。

在当前实现中，训练入口默认选择 `mappo`，实验 checkpoint 也以 MAPPO 系列为主。需要注意的是，代码目录仍沿用 `src/hrvdn` 命名，这是历史遗留结构；论文实验设置中应明确说明实际训练骨架为 MAPPO，以免造成方法名混淆。

## 3. Shield 设置

实验中的 shield 作为执行前 allowed-action filtering 模块接入策略。actor 先提出动作；若动作不在当前 allowed set 中，则 actor 在 allowed set 内重新选择。shield 不直接替 actor 输出单个最优动作，因此其语义不是 planner takeover。

本文方法部分采用三层 allowed-action set：

- `A_hard`：always-on 的一步 hard-safe 允许动作集合，用于保证边界、威胁、机间碰撞和 swap 约束。
- `A_rec`：在 `A_hard` 基础上加入一步递归可行性检查，避免立即进入下一步无安全动作的状态。
- `A_H^{viable}`：将递归可行性推广到有限小视界 `H`，用于表达 stronger look-ahead layer。

在线实现以高效顺序近似为主，并以 exact/projected `A_hard` view 作为语义参照。本文写作中应避免把主路径表述为“每步 exact 求解”；更稳妥的表述是：`A_hard` 以 grounded semantics 为底座，exact/projected view 用于定义参照对象、诊断顺序近似误差，并支持 rescue / diagnostic 分析。

## 4. Progressive 主线比较对象

正文主结果建议只围绕三条 progressive 主线展开。

第一，`non_progressive` 是固定递归过滤基线。它在训练和评测中保持 `recursive` 模式、`H=1` 和风险阈值 `eta=0.35`，用于表示不随训练阶段改变保守性介入强度的设置。

第二，`threshold_only_progressive` 是当前最稳的主正结果候选。它在 early 阶段使用 `safe` 模式、`H=1`、高阈值 `eta=0.9`，主要保留 `A_hard`；mid 阶段切换为 `recursive`、`H=1`、`eta=0.35`；late 阶段继续保持 `recursive/H=1/eta=0.35`。因此，该分支的核心不是扩大 horizon，而是在训练过程中调节 `A_hard -> A_rec` 的介入时机。

第三，`safeearly_progressive` 是主线对照/消融。它的 early 和 mid 阶段与 `threshold_only_progressive` 相同，但 late 阶段切换到 `recursive/H=2/eta=0.55`。该分支用于检验 late-stage stronger layer 是否自然带来更优 learned policy。当前结果不支持将其写成主成功结果。

## 5. H2 与 Dual 的边界定位

`H=2` 与 dual scheduling 都属于方法框架中的自然扩展，但当前不作为正文主成功层。`H=2` 对应更强的小视界 viability 检查；dual scheduling 则在 progressive 基础上进一步调整运行时风险阈值，试图在介入强度和计算成本之间取得折中。

当前写作中，H2 和 dual 应作为边界结果、机制材料或 appendix 候选。它们可以帮助说明 stronger runtime safety filtering 与 better learned policy improvement 之间不存在简单单调关系，但不应被写成已经稳定优于主线 `threshold_only_progressive` 的方法。

## 6. Training Seed 与 Eval Seed 口径

正文主结果使用 `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/` 中的 formal compare 口径。该口径对 `3` 个 training seeds 的 checkpoint 进行评测，每个 checkpoint 使用 `5` 个 eval seeds，每个 eval seed 运行 `5` 个 episodes。该口径用于比较 `non_progressive`、`threshold_only_progressive` 与 `safeearly_progressive` 的主任务和安全指标。

training-seed 级别的 `per_checkpoint_metrics.csv` 用于观察不同训练种子之间的波动，不应与最终 aggregate summary 混写。H2 边界结果当前主要来自 `runs/final_formal_h2_vs_h1_multiseed3x3/`，属于固定 checkpoint 的 `3 eval seeds x 3 episodes` 比较；其证据强度低于 progressive 主线 formal compare。dual 边界结果来自 `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/`，可以作为 discussion 或 appendix 材料。

## 7. 指标定义与主次排序

本文主指标分为三类。任务指标包括 `search_rate` 和 `coverage_ratio`，用于衡量目标发现与覆盖表现。安全指标包括 `collision_count`、`guarantee_broken_rate`、`near_miss_rate` 等，用于衡量硬安全违反和接近风险。shield 行为指标包括 `dead_end_hard_rate`、`dead_end_rec_rate`、`recursive_gate_rate`、`action_replacement_rate` 等，用于解释 allowed-action filtering 如何影响策略执行。

当前不建议把 `episode_return` 作为跨全部目录的统一主指标。原因是旧 baseline 与后续 `risk_base` 系列在 reward normalization 上存在历史口径差异，直接比较可能混入奖励尺度变化。正文主排序应优先依赖任务、安全和 shield-behavior 指标；`episode_return` 可以作为辅助参考，但需要明确口径限制。

## 8. Runtime 指标引用口径

runtime 指标主要包括 `perf_shield_time_ms`、`perf_recursive_time_ms`、`perf_exact_hard_time_ms`、`perf_step_time_ms` 等。当前存在一个需要明确说明的口径问题：`formal_compare_multiseed5x5/summary_metrics.csv` 中部分 `perf_*` 聚合与后续 re-aggregated 机制汇总不完全一致。

因此，正文中任务与安全指标优先引用 `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`；runtime 指标优先引用 `runs/progressive_mechanism_20260428/summary_metrics.csv`。如果后续 codex1 提供统一重聚合主表，本文应更新为同一张 final table；在当前版本中，必须保留这一口径说明，避免把不同来源的 runtime 聚合混为同一证据层级。
