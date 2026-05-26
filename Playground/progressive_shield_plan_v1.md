# Progressive Shield Plan v1

生成时间：`2026-04-22`

本文档用于替代旧版 `progressive_shield_plan.md` 中已经过时、过长、带有历史分叉的内容。  
本版只保留与**当前代码实现**、**当前论文主线**和**下一阶段实验任务**一致的部分。

---

## 1. 当前固定主线

当前论文主线固定为：

1. `MAPPO + shield`
2. `progressive shielding`
3. `recursive-feasible / small-horizon look-ahead shield`
4. `risk-aware dual scheduling`

但这 4 点需要按下面的**当前收口语义**理解：

- `A_hard` 是 always-on 的一步 hard-safe 允许动作集，不能被 gate 关掉。
- `A_rec` 是在 `A_hard` 基础上进一步施加 recursive-feasible 约束的更强允许动作集。
- shield 的语义始终是 `action-set filter`，不是直接替 Actor 选“最优安全动作”。
- Actor 先输出原始偏好；若原动作不在允许集内，则只在允许集内重选。
- risk 的作用是决定是否从 `A_hard` 升级到 `A_rec`，而不是决定是否启用 hard-safe。

---

## 2. progressive 的当前定义

当前主线下，`progressive` 不再表示“训练前期 shield off、后期再 gradually on”。

当前采用的是：

\[
A_{hard} \rightarrow A_{rec} \rightarrow A_H^{viable}
\]

也就是说：

- hard-safe 底线始终保留；
- progressive 调节的是**保守层级**；
- 训练进程影响的是“何时更多启用更强约束”，而不是“是否允许不安全动作通过”。

这一定义比早期的 `off -> on warmup` 更适合作为正式论文主线，因为：

- 理论口径更干净；
- 更容易与 recursive-feasible 和 look-ahead 统一；
- 不会引入“前期其实不安全”的表述漏洞。

---

## 3. 当前代码已经实现的内容

### 3.1 Shield 语义

当前代码已经实现并跑通过的核心语义是：

- `A_hard` always-on
- `A_rec` selective upgrade
- actor constrained re-selection
- 显式 dead-end 语义
- 可用 profiling / validate / summary 统计链路

### 3.2 A_hard 层

当前 `A_hard` 相关实现包括：

- 顺序求解 `sequential`
- 精确一步联合求解 `exact`
- 在线候选模式 `sequential_with_exact_rescue`
- `most_constrained_first` 裁决顺序
- 小范围 `hard repair`
- `_future_safe_exists()` 的多 witness + 小 beam 增强

当前工程判断是：

- `exact` 更适合作为 oracle / 诊断器；
- `sequential_with_exact_rescue` 更适合作为在线主路径候选；
- 纯 `sequential` 已被证明会带来明显的假空集和假非空问题。

### 3.3 Recursive gate

当前 recursive 已明确拆成 3 种 gate：

- `recursive(full)`：每步都算 `A_rec`
- `recursive(risk)`：风险高时才算 `A_rec`
- `recursive(legacy)`：保留旧启发式 gate，仅作兼容 baseline

当前正式推荐的主比较对象是：

- `off`
- `safe`
- `recursive(full)`
- `recursive(risk)`

`legacy` 不是当前第一主线，只能作为补充对照。

### 3.4 风险函数

当前已落地的风险函数主版本是：

\[
\xi = w_{clear}\,\xi^{clear} + w_{region}\,\xi^{region} + w_{hist}\,\xi^{hist}
\]

并且语义位置已经收口为：

- `post-A_hard`
- `pre-A_rec`

即：

1. 先构造 `A_hard`
2. 基于 `A_hard` 的局部统计量算风险
3. 再决定是否升级到 `A_rec`

当前不建议把新的 learned risk、uncertainty 分支或复杂新模型混进主线。

### 3.5 Dead-end / fallback 语义

当前已经去掉默认 silent fallback，空允许集时只允许两种显式语义：

- `fail_closed`
- `emergency`

并且：

- `emergency` 不再伪装成 shield 成功；
- `guarantee_broken`、`dead_end_*`、`emergency_*` 已能进入 validate / CSV / TensorBoard。

---

## 4. 当前代码还没有正式进入闭环的内容

以下内容在配置或文档里已经出现，但**尚未形成当前代码闭环**：

### 4.1 progressive conservativeness

`progressive_enabled` 目前仍属于预留接口。  
当前训练过程并没有真正按照训练进度动态切换 `A_hard / A_rec / A_H^{viable}` 的使用强度。

### 4.2 H=2 look-ahead shield

`H=2` look-ahead 已经以最小侵入方式接入当前 shield 框架，并完成了：

- validate-only 阈值扫描
- `matched gate-rate / matched compute budget` 公平对比
- 一轮正式训练后评测
- `H1/H2 checkpoint x H1/H2 shield` 的 `2x2 cross-eval`

但当前主线仍然是 `H=1`。  
`H=2` 已经被证明具有 runtime stronger-layer 价值，但还没有被证明适合直接升级为正式 standalone 主 baseline。

### 4.3 完整 dual scheduling

`risk_schedule_enabled` 目前未形成真正的 runtime / training 双调度系统。  
当前更准确的状态是：只完成了 `risk-aware recursive gate` 这一半。

---

## 5. 当前科研进度判断

按当前主线划分，当前进度大致处于：

- `Step 1`：formal baselines 基本完成
- `Step 2`：风险 gate 基本完成第一轮收口
- `Step 3`：`H=1` shield 语义、诊断与工程优化已推进到后半段
- `Step 4`：尚未真正开始
- `Step 5`：尚未开始

更具体地说，当前已经完成的是：

- `off / safe / recursive(full) / recursive(risk)` 基本框架
- strict shield 语义收口
- exact solver / rescue / diagnostics
- profiling 与 validate 指标链路
- dead-end 机制显式化

当前还没完成的是：

- `H=2` runtime 价值向稳定训练收益的转化
- progressive 训练期调度
- risk-aware dual scheduling 完整故事
- 与主线完全一致的最终论文实验矩阵

---

## 6. 当前最重要的阶段性结论

### 6.1 A_hard 的故事已经更清楚了

当前可以明确讲的不是“我们已经彻底解决了 A_hard”，而是：

- `A_hard` 必须 always-on；
- 纯顺序近似会制造假空集与假非空；
- `exact` 有研究价值，但在线成本偏高；
- `sequential_with_exact_rescue` 是当前最稳妥的在线候选。

### 6.2 当前 recursive 的重点不是“零碰撞”

当前 recursive 的重点应表述为：

- 在 `A_hard` 之上提供更强的 future-feasibility 过滤；
- 通过 risk gate 控制额外递归检查成本；
- 在安全、任务与计算代价之间形成可解释 trade-off。

### 6.3 当前 H=2 的问题与结论

当前 `H=2` 已经完成 validate-only、formal compare 与 `2x2 cross-eval` 三类诊断。现阶段应明确记录以下结论：

- `H=2` 不是实现失败，而是“已经出现候选好点，但整体还没有被证明稳定、划算、且本质优于 H=1”。
- 旧的 `perf_recursive_time_ms` 在 `H=2` 下存在递归重复累计问题；该问题已经修正。现在：
  - `perf_recursive_time_ms` 更接近 wall-clock stronger-check 时间；
  - `perf_recursive_work_time_ms` 保留累计工作量语义。
- `H=2` 不能直接复用 `H=1` 的 risk threshold。直接使用 `eta=0.35` 时，`H=2` 在 `recursive(risk)` 下表现为“更贵还更差”。
- `H=2` 在 `recursive(full)` 下有一定 safety gain 信号，但单位时间收益偏低，当前不值得作为正式训练主线重启。
- `H=2` 在 `recursive(risk)` 下出现了一个 validate-only 候选好点：`eta≈0.55`。该点表现为更低的 `collision_count`、`guarantee_broken_rate` 和更低的 `perf_shield_time_ms`。
- 在已完成的 `matched compute budget` 对比中，`H=2@eta≈0.55` 相比匹配预算的 `H=1` 点仍表现更好；在 `matched gate-rate` 对比中，其优势明显收窄，说明收益并非“全面支配”，而更像一个稀疏但有效的 stronger layer。
- `2x2 cross-eval` 的关键结论是：四组里最强的不是 `H2 ckpt + H2 shield`，而是 `H1 ckpt + H2 shield`；反过来 `H2 ckpt + H1 shield` 也没有优于 `H1 ckpt + H1 shield`。这说明当前收益更像来自 `H=2 shield` 这个 runtime stronger layer 本身，而不是当前 `H=2` 闭环训练已经把该收益稳定学出来。

因此，当前对 `H=2` 的正式判断是：

- 不建议直接升级为论文主线 standalone baseline；
- 可以保留为 `progressive shielding + dual scheduling` 中的 stronger-layer 候选；
- 当前下一步重点不再是重复证明 `H=2` 是否存在，而是解释并修复“为什么 `H=2` 在 runtime 有价值、但在当前闭环训练后没有稳定转化为更优 checkpoint”。

### 6.4 已完成的关键实验记录

当前已经完成、且对论文主线有直接意义的实验包括：

- `off / safe / recursive(full) / recursive(risk)` 的 `H=1` formal baselines  
  意义：完成当前主线的基本实验骨架，确认 `A_hard always-on + risk-gated A_rec` 的正式对比口径。

- `A_hard` 的 `sequential / exact / sequential_with_exact_rescue` 诊断与对比  
  意义：证明纯顺序近似会产生明显假空集与假非空；明确 `exact` 更适合 oracle/诊断器，`sequential_with_exact_rescue` 更适合在线主路径。

- `H=2` 的 validate-only 阈值扫描  
  意义：证明 `H=2` 不能直接复用 `H=1` 的阈值；在 `recursive(risk)` 下识别出 `eta≈0.55` 这个候选工作点。

- `matched gate-rate / matched compute budget` 公平比较  
  意义：说明 `H=2@eta≈0.55` 的收益不只是“单点偶然好看”。在匹配计算预算时，`H=2` 仍体现出 stronger-layer 价值；在匹配 gate-rate 时，其优势明显收窄，说明它更像稀疏高价值层，而不是全面替代 `H=1` 的主 baseline。

- `H=2` 正式训练后的主比较  
  意义：说明训练后的 `H=2` 在 `guarantee_broken_rate`、`dead_end_rec_rate` 和在线开销上更好，但没有稳定压过 `H=1`，因为 `collision_count` 与 `search_rate` 没有同步变好。

- `2x2 cross-eval`：`H1/H2 checkpoint x H1/H2 shield`  
  意义：直接区分“收益来自 horizon 本身”还是“收益来自当前 H=2 训练闭环”。  
  结论：当前最强组合是 `H1 ckpt + H2 shield`，而不是 `H2 ckpt + H2 shield`；因此目前更应把 `H=2` 理解为一个有价值的 runtime stronger-layer 候选，而不是已经成熟的主 baseline。

### 6.4 当前最需要警惕的分叉

以下方向不应在当前论文里继续发散：

- 把 shield 改成直接替 Actor 选动作
- 把 hard-safe 变成可选项
- 回到“前期 shield off、后期再开”的早期 warmup 主线
- 过早引入 learned safe mask / 新训练分支 / 新模型

---

## 7. 当前论文最稳妥的实验骨架

### 7.1 正式 baseline

当前最推荐的正式 baseline 是：

1. `off`
2. `safe`
3. `recursive(full)`
4. `recursive(risk)`

必要时可补：

5. `recursive(legacy)` 作为兼容启发式对照

### 7.2 当前推荐的 shield 默认工程配置

若目标是推进正式训练/重训，当前更建议优先尝试：

- `A_hard` 使用 `sequential_with_exact_rescue`
- dead-end 默认保持显式语义，不恢复 silent fallback
- recursive 主比较保持 `full` vs `risk`
- `risk_base(clear + region + hist)` 先冻结为当前工作版本

### 7.3 当前重点指标

论文与实验应重点围绕 4 组指标组织：

- 安全：`collision_count`、`threat_violation_count`、`guarantee_broken_rate`
- 约束行为：`shield_trigger_rate`、`action_replacement_rate`、`dead_end_hard_rate`、`dead_end_rec_rate`
- 任务性能：搜索率、回报、发现效率
- 计算代价：`perf_hard_time_ms`、`perf_recursive_time_ms`、`perf_exact_hard_time_ms`、gate run/skip rate

---

## 8. 下一阶段任务排序

### P0：冻结 H=1 主线闭环

目标：

- 用当前更干净的 shield 语义跑一轮正式训练/重训
- 冻结 `off / safe / recursive(full) / recursive(risk)` 口径
- 冻结 `rescue` 作为在线候选、`exact` 作为诊断器的分工

### P1：完成 H=2 look-ahead

目标：

- 保留 `H=1` 作为当前主线 standalone baseline
- 不再重复做已经完成的 `matched gate-rate / matched compute budget / 2x2 cross-eval`
- 将 `H=2` 明确定位为后续 `progressive + dual scheduling` 的 stronger-layer 候选
- 若后续还要继续补证据，优先考虑 `triggered-only` 诊断，而不是立刻重启新的大规模 `H=2` baseline 扫描

当前 `H=2` 的公平验证已经基本完成；下一步重点应转向“如何把其 runtime 价值转化为训练后的稳定收益”。

### P2：再做真正的 progressive conservativeness

目标：

- 让训练进度决定何时更多启用 `recursive / look-ahead`
- 仍保持 `A_hard` always-on
- 不采用 `shield off -> on` 的 warmup 口径

### P3：最后再做完整 dual scheduling

目标：

- 统一训练进度调度与 runtime 风险调度
- 把 `progressive` 和 `risk-aware gate` 组合成一个更完整的双调度框架

---

## 9. 当前最适合的理论创新落点

当前论文更适合强调下面 3 个理论点，而不是“又发明了一个新风险分数”：

1. 分层允许动作集框架  
   \[
   A_H^{viable}(s) \subseteq A_{rec}(s) \subseteq A_{hard}(s)
   \]

2. gate 不破坏一步 hard-safe  
   只要最终动作始终从 `A_hard` 或其子集内选取，则 hard-safe 底线保持不变。

3. selective recursive 的复杂度优势  
   `recursive(risk)` 与 `recursive(full)` 共享同一个 `A_hard` 安全底座，但其额外递归检查代价在期望上更低。

如果后面 `H=2` 跑通，再把 “horizon 增大导致允许集单调收缩” 补成第四个理论支点。

---

## 10. 相关工作与撞题风险

截至 `2026-04-22` 的公开检索，结论可以概括为：

- `shielding` 作为 safe RL / safe MARL 的大方向，已经有比较成熟的先行工作；
- 但“`MAPPO + multi-UAV cooperative target search + always-on A_hard + risk-gated A_hard -> A_rec + 后续 H=2 / progressive / dual scheduling`”这一整套组合，目前没有检索到完全同构的公开论文。

### 10.1 已经明显存在的先行方向

1. 单智能体 shielding
   - Safe Reinforcement Learning via Shielding, 2017  
     <https://arxiv.org/abs/1708.08611>

2. 多智能体 shielding
   - Safe Multi-Agent Reinforcement Learning via Shielding, 2021  
     <https://arxiv.org/abs/2101.11196>

3. 多智能体 model-predictive / look-ahead shielding
   - MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding, 2019  
     <https://arxiv.org/abs/1910.12639>

4. dynamic / adaptive shielding
   - Adaptive Shielding under Uncertainty, 2020  
     <https://arxiv.org/abs/2010.03842>
   - Model-based Dynamic Shielding for Safe and Efficient Multi-Agent Reinforcement Learning, 2023  
     <https://arxiv.org/abs/2304.06281>

5. MAPPO 或多 UAV 搜索场景下的安全增强/动作掩码
   - Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios, 2024  
     该类工作说明“`MAPPO + multi-UAV search + safety heuristics/action mask`”并不是空白方向。  
     <https://www.mdpi.com/2504-446X/8/8/378>

### 10.2 当前尚未被完整覆盖的组合

当前仍然可以主打、且尚未看到完全同构公开结果的点是：

- 面向 multi-UAV cooperative target search 的分层 allowed-action shield 框架；
- `A_hard` always-on 的一步 hard-safe 底座；
- shield 只做 allowed-set filtering，不直接替 Actor 选最优动作；
- `risk-aware` 地决定是否从 `A_hard` 升级到 `A_rec`；
- 将 `exact solver / exact rescue / false-empty diagnostics / profiling` 纳入同一条实现与实验主线；
- 在此基础上继续扩展到 `H=2 look-ahead + progressive conservativeness + dual scheduling`。

换句话说，当前最有希望的创新，不是“发明 shielding”，而是把一条**分层 allowed-set、安全保持、选择性升级、计算可解释**的统一框架，在 multi-UAV cooperative search 上做完整。

### 10.3 当前不宜使用的 claim

论文中不建议使用以下说法：

- “首个 shielded reinforcement learning 方法”
- “首个 safe multi-agent shielding 方法”
- “首个 dynamic / adaptive shielding 方法”
- “首个安全 MAPPO”
- “首个面向 UAV 的 shield 方法”

这些 claim 风险都偏高，容易被现有文献直接顶掉。

### 10.4 当前更安全的 claim

当前更稳妥的表述是：

- 提出一套面向多 UAV 协同搜索的分层 allowed-action shielding framework；
- 在 `A_hard` 始终保持一步 hard-safe 的前提下，实现 selective recursive-feasible / look-ahead upgrade；
- 用 risk-aware 机制控制更强 shield 的介入频率与计算代价；
- 系统分析安全、任务性能与在线计算开销之间的 trade-off。

### 10.5 对当前主线的直接启示

这次检索给出的直接结论是：

- 如果只停留在 `H=1 recursive(risk)`，创新性是有的，但更像“在已有 shielding / safe MARL 上做一个较好的任务化实例”；
- 如果把 `H=2 look-ahead`、`progressive conservativeness`、`dual scheduling` 和理论命题补齐，整篇工作的独立性会明显更强；
- 因此，当前最关键的不是继续发散新模型，而是尽快把“分层 shield + 选择性升级 + 计算代价故事”做完整。

## 11. 一句话版本

当前最准确的一句话主线是：

> 在 `A_hard` 始终保持一步 hard-safe 的前提下，先完成 `MAPPO + safe / recursive(full) / recursive(risk)` 的 H=1 闭环，再扩展到 `H=2` look-ahead，并进一步发展为 progressive conservativeness 与 risk-aware dual scheduling 的统一框架。

---

## 12. 使用说明

后续若继续推进文档与代码，请以本文件为优先版本。  
旧版 `progressive_shield_plan.md` 建议视作历史讨论稿，不再作为当前主线的唯一依据。

---

## 13. `2026-04-28` 到 `2026-05-20` 的初稿冲刺安排

当前目标已经从“继续发散方法分支”切换为“在 `2026-05-20` 前完成一版可投稿初稿”。  
因此，接下来实验推进必须服从论文交付节奏，而不是继续无限扩展 `H=2` 或 `dual` 分支。

### 13.1 当前收口判断

截至 `2026-04-28`，当前结果应按下面方式解释：

- `threshold-only progressive` 是当前最稳定的正向主结果候选；
- `safeearly progressive` 更适合保留为主线语义更干净的消融/对照；
- `H=2` 当前更像 runtime stronger-layer 候选，还没有证明其闭环训练后能稳定优于 `H=1`；
- `dual scheduling` 当前主要体现为降低在线计算成本，但 safety 指标尚未打赢主线 progressive。

因此，当前论文主线应先锁定为：

1. `A_hard` 的 exact/grounded hard-safe 底座
2. 分层 allowed-action shield：`A_hard -> A_rec -> A_H^{viable}`
3. `threshold curriculum / threshold-only progressive`
4. stronger filtering 与 learned policy improvement 并不天然等价的机制分析

### 13.2 五月 20 日前的实验优先级

接下来实验优先级固定如下：

1. 主线结果收口  
   重点比较：
   - `non_progressive`
   - `threshold_only_progressive`
   - `safeearly_progressive`（主线消融）

2. 机制证据补强  
   需要补的不是更多新方法，而是：
   - progressive 为什么有效；
   - 收益是否只是因为 gate 开得更多；
   - stronger runtime filtering 为什么没有自然转化为更优 learned policy。

3. `H=2` 和 `dual` 限时抢救  
   两者都只能作为附加分支继续推进，并且必须设置止损线。  
   若下一轮小范围修正后仍不能稳定打赢 `threshold-only progressive`，则正式降级到 appendix / discussion / future work。

### 13.3 正式口径冻结建议

从现在开始，正式主结果的比较口径建议冻结为：

- `3 training seeds`
- `5 eval seeds per checkpoint`
- `5 episodes per eval seed`
- `device=cpu`

若后续补充 quick validate、ablation 或 smoke test，可以使用更小预算；  
但进入正文主表的结果，应尽量保持上述统一口径，避免再次切换评测标准。

### 13.4 具体时间安排

#### `04-28` 到 `05-02`：锁主线并整理已有结果

目标：

- 确认正文主线只围绕 `non_progressive / threshold_only_progressive / safeearly_progressive` 展开；
- 整理已有 formal compare、training-seed compare、`A_hard` 诊断结果；
- 初步确定正文主表、主图和 appendix 表格的分工；
- 同步开始写 `Introduction`、`Method`、`Experimental Setup`。

此阶段不建议：

- 重启大规模 `H=2` 新训练；
- 为 `dual` 再扩展新机制分支；
- 继续变动主评测口径。

#### `05-03` 到 `05-09`：补机制实验

这一阶段最重要，因为它决定论文能否从“经验上更好”升级到“有解释力”。

建议优先补 3 类机制证据：

1. 训练过程曲线  
   至少包含：
   - `collision_count`
   - `guarantee_broken_rate`
   - `recursive_gate_rate`
   - `episode_return` 或等价 reward 指标

2. progressive stage 统计  
   分 early / mid / late 统计：
   - gate 行为
   - safety 指标
   - effective threshold

3. matched analysis  
   尽量回答：
   - progressive 的收益是不是只因为 gate 更常开；
   - 在接近 gate-rate 或接近 compute budget 下，主线 progressive 是否仍有优势。

#### `05-10` 到 `05-13`：`H=2` 和 `dual` 各给一次限时机会

这一步不是重新开主线，而是做有止损的补救。

建议：

- `H=2` 只保留当前最有希望的一条 runtime 候选，例如 `refine_only` 一类；
- `dual` 只做小范围 band / margin 收缩，不再新增复杂机制；
- 若这轮结果仍然不能稳定优于主线 progressive，则不再继续消耗正文时间。

这一步的目的不是强行把 `H=2` 或 `dual` 升成主结果，  
而是判断它们是否值得保留为：

- appendix 中的正向补充；
- 讨论 stronger layer 设计边界的机制案例；
- future work 的自然延伸。

#### `05-14` 到 `05-16`：统一汇总图表

目标：

- 锁定正文主表；
- 锁定正文主图；
- 生成 appendix 对比表；
- 把所有需要写进文中的数值、文件路径和实验口径核对一遍。

从这一步开始，不再允许新增方法分支。  
只允许补统计、补图和修正结果呈现。

#### `05-17` 到 `05-20`：完成一版初稿

建议优先完成的部分：

1. `Introduction`
2. `Related Work`
3. `Method`
4. `Experimental Setup`
5. `Main Results`
6. `Mechanism Analysis`
7. `Limitations / Discussion`

其中：

- `H=2` 与 `dual` 若没有转正，应写入 `Discussion` 或 appendix；
- `A_hard` 的 exact/approximate 诊断故事应进入正文方法或机制分析部分；
- “stronger runtime filtering 与 better learned policy 不等价”应作为一个明确的讨论结论保留。

### 13.5 五月 20 日前的最小必需实验清单

若只以“完成一版可以写成稿件的实验闭环”为目标，则最小必需内容包括：

1. 主结果表  
   `non_progressive` vs `threshold_only_progressive`，`safeearly_progressive` 作为消融。

2. 一组训练过程曲线  
   体现 progressive curriculum 的训练动态与 safety / reward 收敛差异。

3. 一组 stage 级统计或可视化  
   说明 early / mid / late 的 effective shield 行为确实不同。

4. 一张 `A_hard` 诊断表  
   比较 `sequential / exact / sequential_with_exact_rescue`，支撑底座语义与近似误差诊断。

5. 一张 appendix 表  
   收纳 `H=2` 与 `dual scheduling` 的边界结果，说明为什么它们当前没有进入主线。

### 13.6 当前不建议再做的事

在初稿完成前，不建议继续做以下工作：

- 重启大规模 `H=2` 多轮正式训练；
- 为 `dual` 新增第三层或更复杂 risk 结构；
- 再次改动主评测口径；
- 让 `H=2` 或 `dual` 继续绑架论文主线；
- 在没有机制证据的前提下，仅靠新名字堆叠“创新点”。

### 13.7 当前阶段的一句话执行原则

从现在到 `2026-05-20`，最重要的不是“把所有方向都做赢”，  
而是：

> 先交出一篇主线自洽、结果可信、机制解释足够硬、并且对 `H=2 / dual` 的边界有清楚交代的论文初稿。

---

## 14. 顶刊化收口后的下一阶段工作安排

更新时间：`2026-05-10`

本节是在当前实验和理论文档基础上，对后续工作的重新排序。  
核心判断是：当前不应继续扩展新模块，而应把已有结果整理成“理论-机制-实验”闭环。

当前最值得冲击的论文叙事不是：

- “更强 shield 全面带来更好性能”；

而是：

- 以 grounded `A_hard` 为底座，提出分层 allowed-action shield 语义；
- 用 exact/projected `A_hard` 解释 multi-UAV allowed action 的联合可行性来源；
- 用 progressive / threshold curriculum 展示保守性注入的有限但稳定收益；
- 用 `H=2` 与 `dual` 的 mixed 结果揭示 stronger runtime filtering 与 better learned policy 之间的非单调关系。

### 14.1 第一优先级：统一最终主结果表

目标是先消除结果口径风险，形成一张可以进入正文的 final main table。

主比较对象固定为：

- `non_progressive`
- `threshold_only_progressive`
- `safeearly_progressive`

主指标固定为：

- `search_rate`
- `coverage_ratio`
- `collision_count`
- `guarantee_broken_rate`
- `dead_end_rec_rate`
- `recursive_gate_rate`
- `perf_shield_time_ms`
- `perf_recursive_time_ms`

口径原则：

- 任务与安全指标优先引用 `formal_compare_multiseed5x5`；
- runtime 指标优先引用 `progressive_mechanism_20260428` 的 re-aggregated 统计；
- `episode_return` 不作为跨全部目录的主排序指标；
- 若 final table 中混用了不同来源，必须在 caption 或实验设置中明确说明。

这一阶段的目标不是重新证明 `threshold_only_progressive` 全面胜利，  
而是把它写成当前最稳的 mixed but defensible improvement。

### 14.2 第二优先级：补强 exact/projected `A_hard` 诊断

这是当前最值得补强的机制实验，因为它直接支撑理论中最有价值的部分。

需要回答的问题：

- sequential `A_hard` 和 exact/projected `A_hard` 的差异有多大；
- false empty 有多少；
- false nonempty 有多少；
- `sequential_with_exact_rescue` 能修复多少；
- rescue 带来的额外计算代价是多少。

建议整理成一张正文或 appendix 强机制表，至少包含：

- `seq_empty_exact_nonempty_rate`
- `seq_nonempty_exact_empty_rate`
- `seq_exact_jaccard`
- `rescue_success_rate`
- `perf_exact_hard_time_ms`
- `perf_shield_time_ms`

这组结果的写作目的不是证明在线每步都应该用 exact solver，  
而是支撑：

- exact/projected `A_hard` 是理论参照对象；
- sequential 是工程近似；
- dead-end 可以拆成 true dead-end 与 approximation-induced dead-end；
- rescue 是边界纠偏机制，而不是 planner takeover。

### 14.3 第三优先级：把 `H=2` 和 `dual` 收束为 mismatch 机制证据

当前不建议继续把 `H=2` 或 `dual` 硬推成主成功分支。

它们更适合承担以下角色：

- `H=2`：stronger runtime layer 的候选方向；
- `dual`：更复杂运行时阈值调度的边界尝试；
- 二者共同支撑 stronger runtime filtering 与 better learned policy improvement 不天然等价。

若还需要补实验，应只做小而干净的 matched analysis：

- matched gate-rate；
- matched compute-budget；
- matched intervention frequency。

这类实验的目标不是寻找新的全面最优点，  
而是检查在相近介入频率或相近计算预算下，更强过滤是否仍然稳定转化为更好 learned policy。

若结果继续 mixed，则不要视为失败；  
应把它写成本文的机制发现：

> 更强 runtime safety filtering 可以改善部分即时安全或 future-feasibility 指标，但它也可能改变训练分布、压缩探索空间，从而不必然提升最终 learned policy。

### 14.4 第四优先级：补 progressive stage-level 图

当前 progressive 叙事需要一张清楚的 stage-level 图来支撑。

建议图中至少展示：

- early / mid / late 的 shield mode；
- effective threshold；
- `recursive_gate_rate`；
- `dead_end_rec_rate`；
- `perf_shield_time_ms` 或 `perf_recursive_time_ms`。

图的核心表达应是：

- early 阶段主要停留在 `A_hard`；
- mid / late 阶段逐渐引入 `A_rec`；
- `safeearly_progressive` late-stage 切入 `H=2`，但没有稳定转化为更优 learned policy；
- `threshold_only_progressive` 的收益不是简单来自 gate more，而是来自更合适的训练期保守性注入。

### 14.5 第五优先级：做最小泛化或压力测试

顶刊审稿很可能会追问泛化性。  
在不重启大规模训练的前提下，建议至少补一组 eval-side stress test。

候选扰动包括：

- UAV 数量变化；
- threat 数量或密度变化；
- 地图规模变化；
- 动态威胁或动态目标强度变化。

最小目标不是证明所有场景下都全面更优，  
而是确认以下机制是否仍然存在：

- exact/projected `A_hard` 与 sequential 近似存在差异；
- threshold curriculum 仍能带来部分 safety / future-feasibility 改善；
- stronger filtering 与 learned policy improvement 之间仍非简单单调关系。

若时间不足，优先保留 `A_hard` 诊断与主表重聚合，泛化测试可作为 appendix 或 future work。

### 14.6 理论部分的收口任务

`shield_safety_theory.tex` 已经提供了较好的理论基础，但还需要进一步收紧成论文中的几个硬命题。

建议正文中保留以下命题：

1. 分层 allowed set 的包含关系  
   \[
   A_H^{viable}(s_t) \subseteq A_{rec}(s_t) \subseteq A_{hard}(s_t)
   \]
   并说明 horizon 增大时 viable set 单调收缩。

2. Gate 不破坏 hard safety  
   risk gate 与 progressive gate 只决定是否从 `A_hard` 升级到其子集，因此不会放松一步 hard-safe 底线。

3. Exact/projected `A_hard` 的 sound / complete 参照语义  
   单 agent 动作是否 admissible 应由是否存在 joint hard-safe completion 决定。

4. Sequential approximation 的误差诊断  
   定义 false empty、false nonempty、projected-set disagreement，并用实验表支撑。

当前不建议把 `bicycle_inspired_training_stabilizer.tex` 纳入本篇主线。  
它更适合作为下一篇关于 update-level training stabilizer 或 admissible-update filtering 的扩展方向。

### 14.7 推荐执行顺序

若以冲击顶刊为目标，推荐按以下顺序推进：

1. 统一 final main table；
2. 补强 exact/projected `A_hard` 诊断表；
3. 整理 `H=2 / dual` 为 mismatch 机制证据；
4. 生成 progressive stage-level 图；
5. 补最小泛化或压力测试；
6. 将理论命题与实验表逐一对齐；
7. 再更新中文 master draft 和英文初稿。

如果时间只能支持一项新增工作，优先做：

> exact/projected `A_hard` 诊断实验与表格。

原因是它最能把本文从“MAPPO 加 shield 的工程实验”抬升为“multi-agent allowed-action shield 语义与机制分析”。

### 14.8 当前阶段的一句话判断

下一阶段的核心不是扩展更多方法分支，  
而是：

> 把已有的 `A_hard` 理论、progressive 主结果、`H=2 / dual` mixed 证据，整理成一条关于 layered allowed-action filtering 与 filtering-learning mismatch 的顶刊化叙事。
