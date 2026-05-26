# Progressive Shielding 研究推进方案与论文文本草稿

## 1. 研究方向定位

本工作的核心不是将 `shield` 机制本身作为创新点，而是针对多无人机协同搜索中的探索-安全-计算代价权衡，提出一种**训练进程感知的渐进式屏蔽机制**（progressive shielding mechanism）。需要特别说明的是，早期对 progressive 的直觉曾偏向“训练前期弱屏蔽、后期强屏蔽”的 `off -> on` 式 warmup；但当前工程与论文实验已经收口为另一条更严格的主线：一步 hard-safe 始终保留，训练进程主要调节的是更强保守层级的介入强度。当前工程与论文实验优先以 `MAPPO + shield` 为主线推进，先在更直接、结果积累更多的 MAPPO 基座上完成方法闭环；`HRVDN` 保留为后续迁移验证或扩展方向，而不是当前第一投稿主线。

建议的方法名：

- `PS-MAPPO`：Progressive-Shielded MAPPO
- `RGS-MAPPO`：Risk-Gated Shielded MAPPO
- `PS-HRVDN`：Progressive-Shielded HRVDN（保留作后续迁移版本）

若当前稿件以 `MAPPO + shield` 为主，建议优先使用 `PS-MAPPO`；若后续再将整套方法迁移到值分解主线，可再恢复 `PS-HRVDN` 命名。

---

## 2. 当前主线与方法升级路径

结合最近一轮 baseline、profile 和 risk 验证结果，当前主线已经不再是“单独讨论要不要加 shield”，而是围绕一条分层增强路线来展开：

1. `risk-aware recursive gate`
2. `recursive-feasible / small-horizon look-ahead shield`
3. `progressive conservativeness + dual scheduling`

当前应统一把问题表述为：

- `A_hard` 是每步都计算的、cheap、always-on 的一步 hard-safe 动作集；
- `A_rec` 是只在需要时才额外计算的 recursive future-safe 动作集；
- progressive 的增强发生在 `A_hard -> A_rec -> A_H^{viable}` 这一层级上，而不是放松硬安全；
- shield 始终是 action-set filter，Actor 先给出原始偏好，若原始动作不允许，再在允许动作集内重选。

---

### 2.1 风险感知 recursive gate 的当前定位

当前更合理的做法，是先把 `A_hard` 作为 always-on 基线，再用低成本连续风险分数决定是否值得升级到 `A_rec`。当前 v1 风险定义为：
 
\[
\xi_{i,t}=w_1\,\xi_{i,t}^{\text{clear}}+w_2\,\xi_{i,t}^{\text{region}}+w_3\,\xi_{i,t}^{\text{hist}}
\]

其中：

- `clear`：复用 `A_hard` 流水线中的几何余量统计，反映动作候选的安全裕量；
- `region`：复用局部边界、threat、拥挤度等区域风险特征；
- `hist`：反映过去窗口内 shield 介入频率。

需要强调的是，当前正式启用的只有这三项。下面两类量仍然只作为后续 TODO 预留，不应在当前阶段混入主结论：

- `TODO-1`：`\xi_{i,t}^{\text{feas-proxy}}`，例如基于 `|A_i^{rule}(s_t)|/|A_i|` 或 top-k survival ratio 的可行性代理量
- `TODO-2`：`\xi_{i,t}^{\text{unc}}` / preference-conflict，例如 Actor 的 top-1 是否频繁被 rule mask 打掉、top-k 通过率、Q-gap 或 logits gap

这样做的好处是：

- 风险计算足够 cheap；
- 风险与 `A_hard` 语义一致；
- 风险只决定“是否升级到更强 shield”；
- 便于后续把 gate 继续扩展成更完整的 scheduling 机制。

---

### 2.2 更强 shield 层级的当前理解

当前不应把 shield 看成单一模块，而应看成一组从弱到强的允许动作集层级：

- `Level 0`：`A_hard`，always-on 的一步 hard-safe 层
- `Level 1`：`A_rec`，在 `A_hard` 基础上进一步保证未来一步仍存在安全动作的 recursive-feasible 层
- `Level 2`：`A_H^{viable}`，小 horizon look-ahead 的短期可行层

因此，后续更强 shield 的推进顺序应是：

1. 先把当前 `safe` 与 `recursive` 的分层语义彻底跑通；
2. 再在此基础上扩展到 `H=2` 的 small-horizon look-ahead shield；
3. 然后再根据计算代价决定是否继续增加 horizon，而不是一开始就做大规模未来联合动作枚举。

这里的关键不是“把 shield 做得越复杂越好”，而是：

- `safe / recursive / look-ahead` 三层要有清晰可比性；
- 更强 shield 的代价要可 profile、可解释；
- `H=2` look-ahead shield 必须能体现额外可行性收益；
- `off / safe / recursive / look-ahead` 需要形成稳定 baseline。

---

### 2.3 Progressive 的正确含义

当前 progressive shielding 不应再被理解为“前期允许 unsafe 动作通过，后期再慢慢收紧”，因为这会与硬安全约束直接冲突。

更合理的 progressive 路径是：

\[
A_hard
\rightarrow
A_rec
\rightarrow
A_H^{viable}
\]

也就是说，progressive 的对象不是“是否安全”，而是保守层级：

- 前期以 `A_hard` 为主，尽量保留探索空间；
- 中期按需更多升级到 `A_rec`；
- 后期再逐步引入更强的 short-horizon viable 约束。

因此，后续真正需要讨论的是：

- progressive conservativeness 如何调度；
- `lambda_t` 和 `kappa_t` 分别调什么；
- 何时从 `hard-safe-only` 升级到 `recursive`，再升级到 `look-ahead`。

### 2.3.1 当前主线与早期设想的区别

为避免后续讨论中再次混淆，这里明确区分两条不同语义：

- 早期设想主线：`shield warmup / off -> on`
  - 训练前期尽量不启用 shield，优先保留探索；
  - 随训练推进，再逐步打开 shield 干预；
  - 其核心问题是“安全机制何时开始介入训练”。

- 当前收口主线：`always-on A_hard + progressive conservativeness`
  - `A_hard` 作为一步 hard-safe 底线从训练一开始就始终保留；
  - progressive 调节的是从 `A_hard` 升级到 `A_rec`、再到 `A_H^{viable}` 的保守层级；
  - risk-aware scheduling 决定的是“何时值得升级到更强约束”，而不是“是否启用 hard-safe”。

两者的根本区别在于：

- 早期设想把 progressive 理解为“安全机制从无到有”；
- 当前主线把 progressive 理解为“在 hard-safe 始终保留前提下，保守性逐步增强”。

就当前论文口径而言，后者更适合作为正式主线，因为它能保持更干净的 hard-safe 理论表述，也更容易与 `recursive-feasible / look-ahead / dual scheduling` 统一起来。

---

### 2.4 当前实验推进顺序

结合目前已有 baseline 和实现反馈，当前更合理的推进顺序是：

1. `off / safe / recursive / look-ahead` 的分层 baseline
2. 连续 risk gate 与 legacy gate 的对比验证
3. progressive conservativeness 的训练期调度
4. risk-aware dual scheduling 的联合设计
5. 可视化与统计链路补齐，保证结果可解释、可复现

后续实验的重点应放在：

- 更强 shield 是否真的带来更好的递归可行性；
- 风险 gate 是否能让更强 shield 只在必要时介入；
- Actor constrained re-selection 是否仍保持当前语义；
- 指标体系是否能同时刻画安全、代价和计算开销。

### 2.5 2026-04-18 更新：MAPPO 主线、完成度与当前卡点

当前论文实验建议正式切换为：

- `MAPPO + off / safe / recursive(full) / recursive(risk gate)` 的分层主线；
- `recursive(legacy gate)` 作为可选的启发式补充 baseline，而不是当前第一优先级；
- `HRVDN` 不作为当前实验闭环的必要部分；
- progressive 与 dual scheduling 作为第二阶段增强，而不是当前第一轮投稿的前置条件。

按当前状态估计完成度：

- 若按“方向已跑通并看到初步有效结果”计算，约为 `60%`；
- 若按“形成一版口径干净、可投稿的论文实验闭环”计算，约为 `45%`；
- 若按“冲击强刊甚至更高层级”的标准计算，约为 `30%-35%`。

当前最主要的卡点不是新模型不足，而是实验口径和语义尚未完全收紧：

- baseline reward normalization 与对比口径仍需彻底统一；
- `fallback / emergency / fail-closed` 语义尚未冻结；
- strict hard-safe 结果与当前 fallback 结果仍有混杂风险；
- profiling 链路还不足以完整支撑 `A_hard` 与 `A_rec` 的代价故事；
- 理论命题、证明草图与复杂度分析还没有正式成型；
- `H=2` look-ahead、progressive conservativeness、dual scheduling 还没有进入正式实验闭环。

### 2.6 理论创新落点与后续命题

若希望在现有 `MAPPO + shield` 工作上进一步靠近“理论创新”，当前最值得补强的不是新的风险函数，而是把分层 shield 框架正式化。建议优先围绕以下 3 个命题展开：

1. 命题一：分层允许动作集的包含关系  
   形式化定义 `A_hard`、`A_rec`、`A_H^{viable}`，并证明：
   \[
   A_H^{viable}(s_t) \subseteq A_{rec}(s_t) \subseteq A_{hard}(s_t) \subseteq A
   \]
   同时说明当 `H` 增大时，`A_H^{viable}` 随 horizon 单调收缩。这一命题构成整篇工作的理论骨架。

2. 命题二：gate 不破坏一步 hard-safe  
   证明 risk gate、progressive gate 或其他 selective scheduling 机制只决定“是否从 `A_hard` 升级到更强集合”，而不改变一步 hard-safe 底线。只要最终动作始终从 `A_hard` 或其子集内选取，则一步 hard-safe 保持性不被破坏。

3. 命题三：选择性 recursive 的复杂度优势  
   在 `recursive(full)` 与 `recursive(risk gate)` 的对比下，证明 selective recursive 共享相同的 `A_hard` 安全底座，但其额外递归检查开销在期望上不高于 always-on recursive。该命题是后续计算效率故事与 profiling 分析的理论支点。

围绕上述 3 点，当前论文更容易形成的理论创新表述是：

- 不是“提出了一个新的 risk score”；
- 而是“提出了一个分层允许动作集、且具有安全保持性质与计算可控升级机制的 shield 框架”。

## 3. 论文可直接使用的四项内容

下面给出你前面提到的 4 项内容的可直接落稿版本。

注：本节 `3.1-3.4` 主要保留为早期 `HRVDN` 版本的论文草稿模板。若当前稿件正式切换到 `MAPPO + shield` 主线，则需要将其中的 `HRVDN / PS-HRVDN / 混合奖励框架` 表述，系统性替换为 `MAPPO / PS-MAPPO / MAPPO 训练基座` 的对应写法。

---

## 3.1 论文里的“创新点/贡献”正式中文版

可直接放在引言末尾“本文贡献如下”：

1. 针对多无人机协同搜索任务中早期探索需求与后期安全需求不一致的问题，本文提出一种训练进程感知的渐进式屏蔽机制。不同于传统固定式安全屏蔽方法，该机制能够根据训练阶段动态调节安全干预强度。  

2. 所提方法在训练前期保留较强的策略探索能力，在训练后期逐步强化无人机间避碰与威胁规避约束，从而有效缓解搜索效率与执行安全性之间的冲突。  

3. 本文将渐进式屏蔽机制与 HRVDN 的混合奖励训练框架相结合，构建了“探索—安全”联合课程学习机制，使多智能体在部分可观测、有限通信、动态目标与动态威胁并存的环境中实现更稳定的协同决策。  

4. 实验结果表明，所提方法在保持目标搜索性能的同时，能够显著降低碰撞风险与威胁违规率，并提升策略后期收敛稳定性与部署可行性。  

---

## 3.2 引言中 related work 后面的“本文区别于现有工作”的一段

可直接作为 related work 后的过渡段：

现有安全强化学习研究已经在 shield 机制、多智能体安全协同以及自适应安全干预等方面取得了一定进展，但现有方法大多将安全机制视为全训练过程中的固定模块，容易在训练早期过度限制动作探索，从而影响复杂任务中的策略发现能力。对于多无人机协同搜索问题而言，这一矛盾尤为突出：一方面，智能体需要在训练前期通过充分探索学习协同搜索模式；另一方面，随着策略逐步收敛，系统又必须具备更强的避碰和威胁规避能力，以满足实际部署中的安全需求。基于此，本文不将 shield 本身作为创新点，而是面向多无人机动态搜索场景，提出一种与训练进程耦合的渐进式屏蔽机制，并进一步将其与 HRVDN 的混合奖励训练过程联合设计，实现从探索主导到安全主导的平滑训练过渡。

---

## 3.3 方法名 + 摘要里的 4 句核心卖点

### 方法名

推荐方法名：

- `PS-HRVDN (Progressive-Shielded HRVDN)`

### 摘要核心卖点句式

下面 4 句可拆分进摘要中：

1. To address the conflict between exploration efficiency and execution safety in multi-UAV cooperative search, we propose a Progressive-Shielded HRVDN (PS-HRVDN).  

2. Different from conventional always-on safety mechanisms, the proposed method gradually strengthens shield intervention along the training process, preserving early-stage exploration while improving late-stage safety.  

3. The progressive shield is integrated with the hybrid-reward learning framework, forming a joint exploration-safety curriculum for partially observable multi-UAV target search in dynamic environments.  

4. Experimental results demonstrate that PS-HRVDN maintains competitive search performance while significantly reducing collision risk and improving training stability in the later stage.  

对应中文摘要句式：

1. 针对多无人机协同搜索中探索效率与执行安全性之间的冲突，本文提出一种渐进式屏蔽混合奖励值分解网络方法，即 PS-HRVDN。  

2. 与传统全程固定启用的安全机制不同，所提方法根据训练进程逐步增强屏蔽介入力度，在保留前期探索能力的同时提升后期安全性。  

3. 本文将渐进式屏蔽机制与混合奖励学习框架相结合，构建面向动态环境下部分可观测多无人机目标搜索的探索—安全联合课程学习机制。  

4. 实验结果表明，PS-HRVDN 在保持较强搜索性能的同时，能够显著降低碰撞风险，并提高训练后期的收敛稳定性。  

---

## 3.4 将创新点进一步包装成数学化定义

下面给出一个结合当前实现经验后的、更加贴近后续论文主线的数学表达。这里特别强调：shield 的作用不是直接替代 Actor 选“最优安全动作”，而是**限制动作空间后让 Actor 在安全动作集内重新选择**。

### 1) 原始策略输出

对第 `i` 个 UAV，在时刻 `t` 的局部观测为 `o_i^t`，Actor 输出离散动作偏好：

\[
\pi_\theta(\cdot \mid o_i^t)
\]

若采用值函数形式，可理解为 Actor 对每个动作给出 logits 或偏好分数，再由策略在离散动作空间 `A_i` 中做选择。

### 2) 一步安全动作集合

基于 `C3` 与 `C4`，定义一步安全动作集合：

\[
A_i^{safe}(s_t)=\left\{a \in A_i \mid
\|u_{i,t+1}(a)-u_{j,t+1}\| \ge R_U^{\min},\ 
\|u_{i,t+1}(a)-e_{k,t+1}\| \ge R_E^{\min}
\right\}
\]

其中 `u_{i,t+1}(a)` 表示在当前联合动作参考下，UAV `i` 执行动作 `a` 后的预测位置。

### 3) 一步递归可行动作集合

为了避免进入“当前安全、下一步无安全动作可选”的死局状态，定义一步递归可行动作集合：

\[
A_i^{rec}(s_t)=
\left\{
a \in A_i^{safe}(s_t)
\mid
A_i^{safe}(s_{t+1}(a)) \neq \varnothing
\right\}
\]

这里的 `s_{t+1}(a)` 表示当前 UAV 执行动作 `a` 后的一步预测状态。当前实现允许采用近似递归可行性判断；后续可自然扩展到小 horizon 的 look-ahead 版本。

### 4) 小 horizon 前瞻可行动作集合

进一步可定义 `H` 步前瞻可行动作集合：

\[
A_{i,H}^{viable}(s_t)=
\left\{
a \in A_i
\mid
\text{存在长度为 } H \text{ 的后续安全动作序列}
\right\}
\]

在当前研究路线中，`H=1` 对应递归可行版本，后续方法升级重点是从当前 greedy 近似扩展到 `H=2` 或 `H=3` 的 small-horizon look-ahead shield。

### 5) Actor 约束重选机制

给定允许动作集合 `\mathcal A_i^{allow}(s_t)`，shield 不直接替换动作，而是对 Actor 输出做 mask，并在允许动作集中重新归一化：

\[
\pi_\theta^{mask}(a \mid o_i^t,\mathcal A_i^{allow})
=
\frac{\pi_\theta(a\mid o_i^t)\mathbf{1}[a\in\mathcal A_i^{allow}]}
{\sum_{a' \in A_i}\pi_\theta(a'\mid o_i^t)\mathbf{1}[a'\in\mathcal A_i^{allow}]}
\]

随后由 Actor 在 `\mathcal A_i^{allow}` 中重新选择最终动作：

\[
\tilde a_i^t \sim \pi_\theta^{mask}(\cdot \mid o_i^t,\mathcal A_i^{allow})
\]

其中：

- 若当前模式为 `safe`，则 `\mathcal A_i^{allow}=A_i^{safe}(s_t)`；
- 若当前模式为 `recursive`，则优先取 `\mathcal A_i^{allow}=A_i^{rec}(s_t)`；
- 若后续启用小 horizon 版本，则可取 `\mathcal A_i^{allow}=A_{i,H}^{viable}(s_t)`。

### 6) Progressive conservativeness，而非放松硬安全

当前主线不再将 progressive 理解为“前期允许不安全动作通过、后期再严格限制”，而是始终保留一步硬安全约束，并逐步增强保守层级。可定义保守层级：

\[
\mathcal A_i^{(0)}(s_t)=A_i^{safe}(s_t),\quad
\mathcal A_i^{(1)}(s_t)=A_i^{rec}(s_t),\quad
\mathcal A_i^{(2)}(s_t)=A_{i,H}^{viable}(s_t)
\]

再通过一个随训练进程变化的保守性等级 `\kappa(t)` 选择当前允许动作集合：

\[
\mathcal A_i^{allow}(s_t)=\mathcal A_i^{(\kappa(t))}(s_t)
\]

因此，progressive 的本质是：**硬安全始终保留，但约束从一步安全逐步增强到递归可行和短期前瞻可行。**

### 7) 风险感知双调度

在上述基础上，引入：

- `lambda_t`：更强 shield 介入强度；
- `kappa_t`：动作空间收缩等级。

综合训练进度 `p_t`、风险水平 `\bar\xi_t` 和风险变化趋势 `\Delta\xi_t`，可定义：

\[
z_t = \alpha p_t + (1-\alpha)\bar\xi_t + \gamma \max(0,\Delta \xi_t)
\]

\[
\lambda_t = \sigma(c_\lambda(z_t-b_\lambda)),\qquad
\kappa_t = \left\lfloor K\cdot \sigma(c_\kappa(z_t-b_\kappa)) \right\rfloor
\]

其中 `lambda_t` 与 `kappa_t` 共同调节“何时升级到更强保守模式”和“动作空间收缩到什么程度”。

### 8) Shield 触发惩罚

为促使 Actor 主动学习不触发 shield 的决策行为，可引入 trigger-based penalty：

\[
r_t' = r_t - \beta_t \,\mathbb{I}_{shield}(t)
\]

其中：

- `\mathbb{I}_{shield}(t)=1` 表示当前步触发了 shield；
- `beta_t` 可为常数，也可随训练阶段和风险变化；
- 与其惩罚“是否 unsafe”，更贴近当前实现的是惩罚“是否触发 shield 重选”。

### 9) 数学化创新总结

因此，当前主线可概括为：

> 在始终保持一步硬安全约束的前提下，通过 Actor 约束重选、递归可行保持、小 horizon 前瞻可行扩展，以及风险感知双调度机制，逐步提升多 UAV 协同搜索中的安全可持续性与训练效率平衡。

---

## 4. 建议你下一步优先完成的具体任务

按当前 `MAPPO + shield` 主线，建议将任务拆成两个阶段：先完成一版可投稿实验闭环，再扩展 progressive 与 dual scheduling。

### 4.1 第一阶段：先做出一版口径干净的 `MAPPO + shield` 论文闭环（预计 `3-4` 周）

1. 第 `1` 周：冻结语义与实验口径  
   - 明确 `fallback / emergency / fail-closed` 的最终处理规则；  
   - 统一 reward normalization、checkpoint 对比口径与 evaluation protocol；  
   - 明确当前正式 baseline 为 `off / safe / recursive(full) / recursive(risk gate)`。

2. 第 `2` 周：重跑正式 baseline  
   - 生成统一多 seed 结果；  
   - 确认 `risk gate` 相比 `recursive(full)` 的 safety、计算与介入差异；  
   - 检查 task 指标是否饱和，必要时补充更有区分度的任务指标。

3. 第 `3` 周：做风险函数第一轮扫描  
   - 先扫 `risk_threshold`；  
   - 再扫 `clear / region / hist` 权重；  
   - 目标不是追单点最优，而是证明 `risk gate` 比 always-on recursive 更合理、更稳定、更节省递归调用。

4. 第 `3-4` 周：补 profiling 与代价故事  
   - 最小代价补充 `A_hard`、risk gate、`A_rec` 的时间统计；  
   - 回答 `A_hard` 是否足够 cheap、recursive(full) 多贵、risk gate 节省了多少额外检查；  
   - 形成 safety-task-compute 三条主线上的正式图表。 

5. 第 `4` 周：整理论文骨架  
   - 画 `PS-MAPPO` 方法框图；  
   - 写方法小节，重点写清 `A_hard`、`A_rec`、Actor constrained re-selection 与 risk gate；  
   - 同步写出 3 个理论命题的正式表述与证明草图；  
   - 整理主表、主图、消融表和实验设置说明。

### 4.2 第二阶段：向更强版本扩展（再增加 `3-5` 周）

1. 实现并验证 `H=2` small-horizon look-ahead shield  
   - 前提是 `safe / recursive` 语义已经干净，且 profiling 已经说明 `A_hard` 与 `A_rec` 的代价结构；  
   - 目标是证明 look-ahead 不是简单变慢，而是能减少递归死局或提升可行性。

2. 引入 progressive conservativeness  
   - 将保守层级明确写成 `A_hard -> A_rec -> A_H^{viable}`；  
   - progressive 的调度对象是“保守层级”，而不是“是否允许 unsafe 动作”。

3. 做 risk-aware dual scheduling  
   - 将训练进度 `p_t`、平均风险 `bar_xi_t` 和风险变化趋势 `Delta xi_t` 纳入统一调度；  
   - 明确 `lambda_t` 管“何时更常升级”，`kappa_t` 管“升级到什么层级 / 收缩到什么程度”。

4. 补最终消融与鲁棒性验证  
   - `risk gate` vs `recursive(full)`；  
   - `risk gate` vs `legacy gate`（可选补充启发式 baseline）；  
   - `recursive` vs `look-ahead`；  
   - fixed schedule vs risk-aware schedule；  
   - 不同 seed、不同环境密度或 threat 设置下的泛化表现。

### 4.3 结合理论方向的近期工作顺序

围绕当前已有实现与上面的 3 个理论命题，近期最合理的工作顺序应是：

1. 先冻结 `A_hard`、`A_rec` 与最终动作选择语义  
   - 这是命题一和命题二成立的前提；  
   - 若 `fallback` 仍会把动作放回任意 `valid action`，则 hard-safe 保持性很难在论文里讲干净。

2. 以 `off / safe / recursive(full) / recursive(risk gate)` 跑干净第一轮正式结果  
   - 这是命题三最直接的实验支撑；  
   - 也能最清楚地区分“更强 shield 本身的价值”和“selective gate 的价值”。

3. 补 `A_hard` 与 `A_rec` 的 profiling  
   - 把理论上的复杂度优势落到实际时间占比、递归调用率、平均候选检查数上；  
   - 没有这一步，命题三会停留在抽象层面。

4. 同步整理 3 个命题的正式表述与 proof sketch  
   - 先写集合包含关系；  
   - 再写 gate 不破坏 hard-safe；  
   - 最后写 selective recursive 相对 always-on recursive 的复杂度优势。

5. 在第一轮闭环稳定后，再决定是否加入 `legacy gate`  
   - 若需要更完整的经验对照，可将其作为补充 heuristic baseline；  
   - 但它不应阻塞当前主线，因为理论叙事的核心并不依赖 `legacy gate`。

### 4.4 当前建议的总体时间判断

- 若只目标一版扎实的 `MAPPO + hard-safe / recursive / risk-gated shield` 论文，预计还需 `3-4` 周；
- 若希望进一步补上 `H=2`、progressive conservativeness 与 dual scheduling，预计总周期约为 `6-9` 周；
- 在第一阶段未完成前，不建议过早发散到新模型、新数据流或新的训练分支。

---

## 5. 当前一句话主线总结

下面这句话可以作为当前整条主线的压缩表述：

> 当前最核心的问题，已经从“是否在 MAPPO 上加 shield”转变为：在 `A_hard` 始终保持一步 hard-safe 的前提下，用低成本风险 gate 决定何时升级到 `A_rec`，再逐步扩展到 `small-horizon look-ahead`，并进一步与训练进度耦合成 `progressive conservativeness`。

## 6. 方向潜力评估与发表前景判断

下面对该方向的研究潜力和发表前景做一个相对客观的评估。

### 6.1 总体判断

结论可以概括为：

- 该方向具有较好的研究价值和论文发表潜力；
- 仅凭“随着训练进行逐步增强 shield”这一单点创新，通常不足以稳定支撑顶刊级工作；
- 如果能够将其扩展为一个具有方法普适性、理论支撑和系统实验验证的完整框架，则有机会冲击较强期刊，甚至具备向顶刊靠近的潜力。

换句话说，这不是一个“天然顶刊点”，但它是一个**可以继续做深并长成高水平工作的研究雏形**。

---

### 6.2 为什么这个方向值得做

该方向具备潜力，主要来自以下几个方面：

1. 问题本身有实际价值  
   多无人机协同搜索、动态目标、动态威胁、有限通信以及部分可观测等因素共同构成了一个非常现实的无人系统应用场景，安全性问题并非附属问题，而是部署层面的核心问题。

2. 抓住了一个真实矛盾  
   多智能体强化学习需要早期探索，而安全机制又往往抑制探索。尤其在 UAV 协同搜索中，固定强安全机制会明显削弱前期策略发现能力，这个矛盾是真实存在且尚未被充分解决的。

3. 方法方向合理  
   从固定式 shield 走向渐进式、训练进程感知的 shield，在逻辑上是自然成立的，也容易与课程学习、风险调度和安全强化学习建立联系。

4. 容易形成有效实验结果  
   该方向较容易在 collision count、threat violation count、search rate、coverage ratio、convergence stability 等指标上形成较清晰的效果对比，因此具备较好的实验可验证性。

---

### 6.3 当前创新强度的现实评价

需要明确的是，以下内容本身并不是空白方向：

- shielded RL 已有较多工作；
- multi-agent shielding 已有研究基础；
- dynamic/adaptive shielding 已经开始出现；
- curriculum safe RL 也不是全新概念。

因此，如果论文仅仅表述为：

> 我们在 MAPPO 中加入了一个随训练逐步增强的 shield。

那么这个创新更像是一个**合理且有效的方法改进**，但还不足以自然支撑“顶刊级创新强度”。

更准确地说，你当前这个点的创新强度属于：

- 高于普通工程调参；
- 低于完全原创的方法范式；
- 属于“有潜力进一步放大”的中等偏上创新点。

---

### 6.4 发表层级的初步判断

从当前构想出发，可以做如下分层判断：

1. 作为硕士/博士阶段的核心论文方向  
   是合适的，而且有较高可行性。

2. 发表中上水平期刊/会议  
   如果方法实现完整、实验设计规范、对比充分，那么希望较大。

3. 冲击领域内较强期刊  
   有机会，但前提是需要在“方法完整性、实验充分性、理论支撑”三个方面明显加强。

4. 冲击真正意义上的顶刊  
   不能只依靠当前这个朴素版本，必须将其提升为更系统、更一般化的方法框架。

因此，最现实的判断是：

> 这个方向“值得做，也有希望发强论文”，但若目标是顶刊，必须继续升级，而不能停留在简单的 schedule-based shield 设计上。

---

### 6.5 若想提升到顶刊水准，需要补强的关键点

如果希望把这个方向做成真正高水平甚至顶刊导向的工作，建议至少从以下几个方向强化。

#### 1) 从经验调度升级为有依据的调度机制

当前最容易实现的是手工设计一个 `lambda(t)`，例如线性或 sigmoid 增长。但这种方式偏经验化。

更强的做法是让 `lambda(t)` 同时依赖：

- 训练进度；
- 当前碰撞风险；
- 策略不确定性；
- 最近阶段的 collision rate；
- 环境复杂度变化。

即构造：

\[
\lambda_t = f(\text{training progress}, \text{risk}, \text{uncertainty})
\]

这样可以把方法从“人工调 schedule”提升到“风险感知安全调度框架”。

#### 2) 增加一定的理论分析

顶刊工作通常不满足于仅有实验现象，还希望看到方法性质分析。可考虑的理论切入点包括：

- 若安全动作集合非空，则 shield 后执行动作始终满足 `C3/C4`；
- 给出碰撞/违规概率上界；
- 分析渐进式 shield 对探索空间收缩的影响；
- 讨论后期安全性增强对稳定收敛的作用机制。

即使不是很重的理论，只要能给出明确命题、条件与结论，也会显著提升论文层次。

#### 3) 不只提升安全性，还要证明综合性能

如果方法只是降低碰撞率，但显著损害搜索性能，那么更像安全补丁而不是核心推进。

更理想的目标是同时证明：

- 前期探索效率优于固定强 shield；
- 后期安全性优于无 shield；
- 最终搜索性能不劣于甚至优于原始方法；
- 训练稳定性更强。

这样才能体现“探索—安全平衡”是真正成立的。

#### 4) 做成具有普适性的框架

如果论文只写成“针对某一篇 UAV 搜索模型做的安全补丁”，上限会偏低。

如果能够上升为：

> A training-progress-aware shield scheduling framework for safe cooperative MARL under partial observability.

那么 UAV 协同搜索就只是该框架的重要应用场景之一，论文的普适价值会更高。

#### 5) 增加第二创新支柱

若只靠 progressive shielding 一个点，创新密度可能不够。建议额外叠加一个更强的创新支柱，例如：

- 风险预测式 shield；
- 不确定性感知的自适应 `lambda(t)`；
- 多步安全预测而非一步安全判别；
- shield 介入信息反哺策略学习；
- 局部 shield 与集中式联合冲突消解相结合。

这样可以把论文从“单点增强”提升为“成体系的方法设计”。

---

### 6.6 当前方向的主观评分

下面给出一个阶段性主观评分，仅用于帮助你把握研究投入与预期：

- 问题价值：`8/10`
- 当前创新强度：`6/10`
- 可扩展潜力：`8/10`
- 发强论文潜力：`7.5/10`
- 直接顶刊潜力：`5.5/10`
- 补强后的升级潜力：`8/10`

这个评分的含义是：

- 它不是一个低水平方向；
- 它值得继续深挖；
- 但若目标是顶刊，必须继续补强，而不是停在当前版本。

---

### 6.7 最终建议

对这个方向，最合理的推进策略不是一开始就执着于“能不能顶刊”，而是分阶段推进：

1. 先将 `PS-HRVDN` 做成一版完整、可靠、可复现的方法；
2. 先证明它显著优于 `no shield` 和 `fixed shield`；
3. 再把 `lambda(t)` 从简单 schedule 升级为风险/不确定性感知机制；
4. 再补理论分析和更强泛化实验；
5. 最后再决定冲击的期刊层级。

一句话总结就是：

> 这个方向值得做，而且有希望做成强论文；但若目标是顶刊，必须把“渐进式 shield”从一个直观想法升级为一个具有理论支撑、普适表述和系统实验验证的完整安全协同强化学习框架。

---

### 6.8 若考虑拆成两篇论文，顶刊优先时的推荐拆法

如果后续确实考虑拆成两篇论文，那么更合理的策略不是把当前材料机械切成两半，而是采用“旗舰主论文 + 后续升级论文”的结构。

核心判断是：

- 以当前进展看，`risk` 单独拿出来还不够像一篇顶刊主论文的核心；
- 当前更有希望冲高水平的，是“`A_hard -> A_rec -> A_H^{viable}` 的分层 shield 框架 + progressive conservativeness”这条统一主线；
- 因此，第一篇应优先围绕统一框架做厚，第二篇再把 risk 优化提炼成更独立的问题。

更推荐的拆分方式是：

#### 第一篇：以统一 progressive shield 框架冲高水平

第一篇的主题应聚焦于：

- `A_hard` 始终开启的一步 hard-safe 基线；
- `A_rec` 与 `A_H^{viable}` 形成的分层可行性保持；
- actor-compatible 的 constrained re-selection 机制；
- 训练进度感知的 progressive conservativeness；
- 安全性、递归可行性、任务性能与在线计算代价之间的整体平衡。

在这篇里，`risk` 可以作为辅助 gate 保留，但不必把它包装成第一主贡献。只要它能够合理支持“按需升级到更强 shield”，就足以服务整篇主线。

#### 第二篇：把 risk 升级为独立的计算感知安全调度问题

第二篇只有在下面条件满足时，才值得单独冲高水平：

- `risk` 的判别力明显强于当前版本；
- 它不再只是 `clear + region + hist` 的权重微调；
- 它能够系统回答“在有限计算预算下，何时值得从 `A_hard` 升级到 `A_rec` 或更强层级”这一更大问题。

换句话说，第二篇应从“risk function 调参”升级为“计算预算感知的自适应安全调度框架”。

#### 顶刊优先时的资源投入建议

如果目标明确是冲击顶刊，那么当前资源分配应优先满足第一篇主论文，而不是把过多时间提前压在 risk 微调上。更合理的顺序是：

1. 先把统一框架、分层 shield 语义和核心 baseline 做扎实；
2. 先补理论、`H=2` look-ahead、复杂度分析和更系统的实验；
3. 再把 `risk` 做到“够用、可解释、能省算力”；
4. 等第一篇主框架站稳后，再把 risk 提炼成第二篇独立问题。

一句话总结：

> 若以顶刊优先，最稳妥的策略不是现在就把 risk 单独拆走，而是先集中火力完成一篇框架型主论文，再把 risk 优化沉淀为后续的第二篇高水平工作。

---

### 6.9 两篇论文的明确任务拆分

为了避免后续实现和实验不断越界，下面把两篇论文的边界、必做项和可后置项明确列出来。

#### 论文一：统一 progressive shield 框架主论文

这篇论文的定位是顶刊优先的旗舰主论文，核心目标不是把某一个 gate 做到最优，而是把整条统一主线讲完整、做扎实。

主问题应表述为：

> 在始终保持 `A_hard` 一步 hard-safe 的前提下，如何通过 `A_rec` 和 `A_H^{viable}` 的分层可行性保持，以及训练进度感知的 progressive conservativeness，在安全、探索、递归可行性和在线计算代价之间实现更优平衡。

这篇论文必须包含的内容：

- `A_hard -> A_rec -> A_H^{viable}` 的理论分层与实现语义
- actor-compatible constrained re-selection，而不是外部控制器替 Actor 选动作
- `off / safe / recursive / look-ahead` 的清晰 baseline 关系
- progressive conservativeness 的训练期解释与调度逻辑
- 安全性、任务性能、递归可行性和计算代价四类指标的联合评估
- 至少一套能自圆其说的理论命题、性质分析和复杂度说明

这篇论文中 risk 的角色应限定为：

- 用于支持“是否从 `A_hard` 升级到 `A_rec`”
- 是一个辅助 gate，而不是第一主贡献
- 只需要做到“可解释、有效、能省算力”，不要求在这一篇里把 risk 做到极致

这篇论文可以暂缓或弱化的内容：

- 极致优化 risk 的分量设计
- 更复杂的 uncertainty / preference-conflict 建模
- learned risk network
- 更重的 offline 标注体系
- 完整 decentralized shield

#### 论文二：risk-aware 计算感知安全调度论文

第二篇论文只有在 risk 真正形成独立科学问题后才值得展开。它的核心不应再是“换几个 risk 分量看看效果”，而应升级为：

> 在有限计算预算下，如何用可解释的风险或不确定性信号，自适应决定何时值得从 `A_hard` 升级到 `A_rec` 或更强层级。

这篇论文要成立，至少需要满足：

- risk 对 `need_rec` 或其他更合理 oracle 标签的判别力明显强于当前版本
- risk 不只是阈值和权重微调，而是形成了更完整的 feature / gate / budget 设计
- online 结果能稳定体现“更少 recursive 代价 + 不更差的安全或可行性”
- 评估协议足够独立，能够支撑“risk-aware scheduling”作为独立卖点

这篇论文更适合包含的内容：

- 更强的 offline relabeling / oracle 设计
- feasibility-proxy、fragility、uncertainty、preference-conflict 等 richer signal
- gate threshold 的自适应选择
- risk 与计算预算、环境复杂度、训练阶段的联合调度
- 更细的 precision / recall / gate_rate / compute-cost 曲线分析

#### 当前执行边界

从现在开始，后续代码和实验应按下面的边界执行：

1. 第一篇论文的框架边界不再摇摆。  
   当前固定主线就是 `progressive shielding + recursive-feasible / small-horizon look-ahead shield + risk-aware dual scheduling`。

2. risk 的当前优化目标是“服务第一篇主论文”，不是立刻孵化第二篇。  
   也就是说，当前优化 risk 的标准是：它是否足以支撑 `A_hard -> A_rec` 的升级逻辑，而不是它是否已经形成独立论文。

3. 只有当 risk 的区分能力和独立叙事明显增强后，才正式把它拆成第二篇论文。  
   在那之前，risk 仍归属于第一篇的辅助模块，而不是单独主轴。

#### 当前最实际的任务顺序

基于上面的拆分，接下来最实际的顺序应是：

1. 继续巩固第一篇主论文的框架语义和 baseline 体系
2. 继续优化 risk，但只在“辅助 gate”边界内推进
3. 若 risk 后续出现明显跃升，再把它整理成第二篇的独立问题

一句话说，接下来的 risk 优化应当带着一个明确约束：

> 先把 risk 做成第一篇主论文中一个足够可信、足够省算力、足够可解释的升级判据；至于第二篇 risk 论文，要等它真正长成独立问题后再拆。

---

## 7. 关于安全动作集非空性的进一步思路

在前面的理论分析中，严格安全性往往依赖于条件

\[
A^{\mathrm{safe}}(s_t)\neq\varnothing.
\]

因此，一个更深层的问题是：能否通过更强的 shield 设计，使系统尽量避免进入“安全动作集为空”的死局状态。

### 7.1 基本判断

结论是：

- 仅靠一个普通的 shield 算子，通常不能对所有状态无条件保证 `A^{\mathrm{safe}}(s_t)` 非空；
- 但可以通过更强的动作筛选逻辑，主动避免系统进入将来无安全动作可选的状态；
- 这意味着 shield 不仅要过滤“当前不安全动作”，还要过滤那些“虽然当前安全、但会导致下一步无安全动作可选”的动作。

也就是说，我们真正希望维持的不是单纯的当前安全，而是**可持续安全（sustainable safety）**或**递归可行安全（recursive-feasible safety）**。

---

### 7.2 一步递归可行 shield 的核心思想

普通安全动作集定义为：

\[
A^{\mathrm{safe}}(s_t)
=
\left\{
a_t \in A \mid f(s_t,a_t)\in \mathcal S_{\mathrm{safe}}
\right\}.
\]

为了避免进入“下一步无动作可选”的状态，可以进一步定义一步递归可行动作集：

\[
A^{\mathrm{rec}}(s_t)
=
\left\{
a_t \in A \mid
f(s_t,a_t)\in \mathcal S_{\mathrm{safe}}
\ \text{and}\
A^{\mathrm{safe}}(f(s_t,a_t))\neq\varnothing
\right\}.
\]

这个定义的含义是：

- 当前动作执行后系统仍然安全；
- 并且下一时刻仍至少存在一个安全动作可选。

相比普通 shield，它避免了系统被送入“当前不撞，但下一步必死”的状态。

---

### 7.3 小 horizon look-ahead shield 的扩展方向

一步递归可行只是最小版本。进一步可以定义小 horizon 的前瞻可行动作集，例如：

\[
A_H^{\mathrm{viable}}(s_t)
=
\left\{
a_t \in A \mid
f(s_t,a_t)\in \mathcal S_{\mathrm{safe}}
\ \text{and there exists a future safe sequence for } H \text{ steps}
\right\}.
\]

这个版本的含义是：

- 当前动作不仅要保证一步后安全；
- 还要保证未来 `H` 步内存在一条安全可行的动作序列。

工程上可以将其理解为一种小规模 look-ahead shield，用于避免局部短视策略导致的“迟早进入死胡同”问题。

---

### 7.4 与“接近障碍物时收缩动作空间”的关系

这一思路可以通过风险驱动的动作空间收缩来实现。例如，当 UAV 靠近障碍物、边界或 threat 区域时，不再开放全部动作，而是逐步删去高风险动作，只保留能够维持后续可行性的动作。

一种概念性表达为：

\[
A^{\mathrm{allow}}(s_t)=
\begin{cases}
A, & \rho(s_t)\le \eta_1,\\
A\setminus\{\text{high-risk actions}\}, & \eta_1<\rho(s_t)\le\eta_2,\\
A^{\mathrm{turn-only}}(s_t), & \eta_2<\rho(s_t)\le\eta_3,\\
A^{\mathrm{rec}}(s_t)\ \text{or}\ A_H^{\mathrm{viable}}(s_t), & \rho(s_t)>\eta_3,
\end{cases}
\]

其中 `rho(s_t)` 是风险度量，`\eta_1,\eta_2,\eta_3` 是风险分级阈值。

这正对应了“接近障碍物时，只保留左转和右转，而不再允许继续前进”的策略思想。

---

### 7.5 为什么这条线比单纯 progressive shield 更强

单独的 progressive shield 主要解决的是：

- 训练早期探索；
- 训练后期安全。

而加入一步递归可行 shield 和小 horizon look-ahead shield 之后，方法开始进一步处理：

- 如何避免走入下一步无安全动作可选的死局；
- 如何在局部前瞻范围内维持安全可持续性；
- 如何把当前安全提升为短期未来的可恢复安全。

因此，该方向的理论层次从“减少碰撞”升级为“维持安全可行决策能力”，整体方法深度明显提升。

---

### 7.6 这一组合方向的潜力再评估

如果将方法升级为三部分统一框架：

1. `progressive shielding`：解决训练过程中的探索-安全权衡；
2. `one-step recursive-feasible shielding`：避免进入下一步无安全动作的状态；
3. `small-horizon look-ahead shielding`：进一步增强短期未来可行性；

那么这已经不再是简单的安全补丁，而开始具备如下特征：

- 训练层、执行层和预测层形成统一结构；
- 可以围绕“可持续安全决策”建立更强的理论分析；
- 实验上可以设计更有说服力的指标，如 dead-end state 次数、递归可行率、未来可行性保持率等；
- 与现有固定 shield 或单步 shield 相比，方法层次更高，也更具一般化空间。

基于当前判断，这一组合方向已经明显强于原始的单一 progressive shield 方案，并开始具备“冲击高水平甚至顶刊候选工作”的潜力。

更具体地说，其阶段性主观评估可更新为：

- 问题价值：`8.5/10`
- 当前创新潜力：`7.5/10`
- 理论可塑性：`8/10`
- 强论文潜力：`8.5/10`
- 顶刊冲击潜力：`7/10`

需要强调的是：

- 目前可以说“开始具备冲击顶刊的潜力”；
- 但要真正达到该层级，仍需要把三部分统一成一个完整框架，并补充风险感知调度、理论证明、复杂度分析与系统实验。

---

### 7.7 当前最推荐的发展主线

基于目前的讨论，最推荐的技术推进路线为：

1. 先完成一步递归可行 shield 的定义、算法与理论；
2. 再扩展到小 horizon 的 look-ahead shield；
3. 再将其与 progressive shield 统一成一个训练-执行一体化框架；
4. 最后引入风险感知自适应调度，使 shield 强度与训练进度、当前风险、可行性裕量和不确定性共同关联。

如果上述路线能够完整落地，那么论文的总体定位可以进一步提升为：

> 面向多无人机协同搜索的训练进程感知、可持续安全、前瞻可行的统一屏蔽框架。

---

## 8. 用连续风险分数驱动 recursive gate

当前 risk gate 的作用已经明确：它不是用来决定是否启用 hard-safe，而是用来决定是否值得从 `A_hard` 升级到 `A_rec`。

因此，当前正确流水线是：

- 先构造 `A_hard`
- 再基于 `A_hard` 及其局部统计量计算风险
- 高风险时才升级做 `A_rec`
- 如果 `A_rec` 为空，则回退到 `A_hard`

这一步的意义在于：把原来偏手工的 binary high-risk gate，升级成可解释、可扫描阈值、可扩展分量的连续 gate，同时不改变 shield 的基本语义。

---

### 8.1 当前 v1 风险函数的正式定义

当前正式启用的 per-agent、per-step 风险分数为：

\[
\xi_{i,t}=w_1\,\xi_{i,t}^{\text{clear}}+w_2\,\xi_{i,t}^{\text{region}}+w_3\,\xi_{i,t}^{\text{hist}},
\qquad
w_1+w_2+w_3=1,
\quad w_k\ge 0
\]

它应满足以下要求：

- 计算 cheap，尽量复用 `A_hard` 流水线里的局部几何特征与缓存；
- 语义明确，位置是 `post-A_hard / pre-A_rec`；
- 只决定 recursive gate，不放松 hard-safe；
- 为后续的 scheduling 机制预留扩展位，但当前不引入额外风险网络。

当前推荐的默认权重是：

\[
w_1=0.5,\qquad w_2=0.3,\qquad w_3=0.2
\]

---

### 8.2 当前正式启用的三项风险

#### 1) Clearance risk

令 `m_{i,t}` 表示当前 agent 在当前 step 下、基于 `A_hard` 候选动作统计得到的几何余量，例如 `min_candidate_clearance`，则有：

\[
\xi_{i,t}^{\text{clear}}
=
\mathrm{clip}\left(1-\frac{m_{i,t}}{M_c},0,1\right)
\]

其中 `M_c` 是 clearance 归一化常数。该项的意义是：候选动作的安全裕量越小，当前状态越接近需要升级到更强 shield 的区域。

#### 2) Region risk

region 项复用 `A_hard` 周边已经提取出的局部风险特征，包括：

- `near_boundary`：是否靠近边界
- `local_threat_count`：局部 threat 数量
- `crowded`：局部 UAV 是否拥挤

定义为：

\[
\xi_{i,t}^{\text{region}}
=
\frac{
\mathbb I_{\text{boundary}}+
\tilde n^{\text{threat}}_{i,t}+
\mathbb I_{\text{crowded}}
}{3}
\]

其中 `\tilde n^{threat}` 是归一化后的局部 threat 数量。该项的意义是：即使单个动作的 clearance 还不算太差，若当前局部空间本身更危险，也更值得触发更强的 future-safe 检查。

#### 3) History risk

history 项反映过去窗口内 shield 介入频率。为避免自引用，理论上应只使用过去窗口：

\[
\xi_{i,t}^{\text{hist}}
=
\frac{1}{W}
\sum_{\tau=t-W}^{t-1}
\mathbb I_{\text{shield}}(i,\tau)
\]

其中 `W` 为窗口长度。该项的意义是：若一个 agent 最近持续需要 shield 重选，说明它正反复进入脆弱区域，当前 step 更可能值得升级到 `A_rec`。

---

### 8.3 当前不启用但预留的扩展项

为了保持当前版本足够轻量，下面两类项只保留为 TODO，不在本轮主结论中启用。

#### TODO-1：Feasibility proxy

一个自然候选是：

\[
\xi_{i,t}^{\text{feas-proxy}}
=
1-\frac{|A_i^{rule}(s_t)|}{|A_i|}
\]

或者进一步改成 top-k survival ratio 等更贴近运行时 candidate pruning 的版本。它的目标是捕捉“`A_hard` 已经开始失去动作余量”的脆弱性。

#### TODO-2：Uncertainty / preference-conflict

另一个候选方向是：

- Actor 的 top-1 动作是否频繁被 `rule mask` 打掉
- top-k 动作保留率
- Q-gap 或 logits gap 的不确定性度量

这类量更偏向策略偏好与安全约束之间的冲突度量，适合作为后续 scheduling 的补充分量，但当前还不应抢在 `clear + region + hist` 前面。

---

### 8.4 风险如何决定 recursive gate

当前最关键的语义是：风险分数不是在 `safe` 和 `recursive` 两个实验模式之间切换，而是在 `recursive` 模式内部，决定某个 agent 这一步是否值得升级到 `A_rec`。

形式化地说：

\[
\xi_{i,t} \ge \eta
\quad \Rightarrow \quad \text{run recursive check on top of } A_hard
\]

对应实现语义应为：

- `off`：不走 shield
- `safe`：只使用 `A_hard`，即 `hard-safe-only mode`
- `recursive`：先构造 `A_hard`，再计算 `\xi_{i,t}`，只有当 `\xi_{i,t}` 超过阈值 `\eta` 时才进一步算 `A_rec`

后续如果继续扩展，这个 gate 还可以自然地连接到：

- `shield intervention strength`
- `action-space shrinkage level`
- `look-ahead` 的 horizon 选择

---

### 8.5 这不是新的控制器，而是更便宜的升级判据

这一层需要在论文和实现里都说清楚：

> 当前连续风险函数的作用，是为更强 shield 提供一个 cheap 的升级判据，而不是替 Actor 决定动作，更不是放松 hard-safe。

因此，它满足四条约束：

- 不引入新的风险网络；
- 不改变 Actor 先提议、shield 后过滤的执行逻辑；
- 不把 `A_hard` 变成可选项；
- 不把 recursive gate 混同为完整的 scheduling。

---

### 8.6 当前这一线已经带来的启发

从 `clear + region + hist` 以及后续几版小改动的验证结果看，当前这一线已经至少说明了四件事：

1. 连续 risk gate 是可以工作的。  
   它确实能减少 recursive 开销，并且比 `always recursive` 更现实。

2. 但 v1 风险函数的判别力还不够强。  
   当前 precision 和 recall 都偏低，说明“是否值得升级到 `A_rec`”还没有被很好地区分开。

3. `hist` 在固定 `safe` 轨迹验证中帮助有限。  
   后续应优先补强与 `A_hard` 余量和脆弱性更直接相关的分量。

4. 接下来的重点应是“先把 gate 做准”，而不是立刻扩展成完整 dual scheduling。  
   也就是说，短期内更值得做的是离线打标、阈值扫描、权重扫描和 cheap 分量补强。

## 9. 当前路线修正与后续科研主线（2026-04）

结合最近一轮正式 baseline 准备、代码实现反馈以及对在线计算瓶颈的重新判断，当前研究路线做如下修正与收束。

### 9.1 不切换到 `learned safe mask` 主线

虽然“`rule mask + learned safe mask + backup action + rare exact shield`”这条路线在工程上很有吸引力，也确实能缓解在线 exact recursive shield 的计算负担，但它与已有 safe-action mask / recovery / action filtering 一类工作重合度更高，不适合作为当前论文的主线创新。

因此，当前论文**不改主方向**，继续沿着之前确定的三点主线推进：

1. `progressive shield`
2. `one-step recursive-feasible / small-horizon look-ahead shield`
3. `risk-aware dual scheduling`

也就是说：

- `learned safe mask` 可以保留为远期扩展方向；
- 但当前论文主线仍然是“训练进程感知 + 递归可行保持 + 风险自适应调度”的统一 shield 框架。

---

### 9.2 吸收工程经验，但不改变研究问题

最近的实现和 profile 结果说明：

- 真正拖慢训练的不是神经网络，而是在线 shield，尤其是 recursive 可行性检查；
- naive 的“每步全动作枚举 + 递归检查”不适合作为长期在线主链路；
- 但这并不意味着要放弃 recursive / look-ahead / scheduling 这条研究主线。

更合理的理解是：

- **研究问题不变**；
- **实现策略需要更克制、更可计算**。

因此，后续的 `look-ahead shield` 不再追求“全联合精确未来可行性枚举”，而应强调：

- 小 horizon；
- 局部可行性近似；
- 顺序裁决；
- 缓存与剪枝；
- 必要时只在关键状态调用更强判断。

这属于吸收工程诊断结果来约束实现方式，而不是改变方法创新点。

---

### 9.3 Progressive 的重新定义

当前路线下，progressive 不应再被理解为“前期允许 unsafe 动作通过、后期再逐步变严格”，因为这会与硬安全约束冲突。

更合理的定义是：

- 一步硬安全始终保留；
- progressive 调节的是**保守层级**而不是“是否安全”。

也就是说，后续 progressive 的增强路径应理解为：

\[
A^{safe}
\rightarrow
A^{rec}
\rightarrow
A_H^{viable}
\]

或等价地理解为：

- 前期：一步安全为主；
- 中期：递归可行为主；
- 后期：短期前瞻可行为主。

因此，progressive 的核心作用是缓解“更强安全机制带来更高任务代价”的问题，而不是放弃硬安全。

---

### 9.4 Baseline 的当前结论与意义

当前已经完成了 `off / safe / recursive` 的 smoke baseline，结果表明：

- `collision_count` 不是区分 `safe` 与 `recursive` 的主要指标；
- 真正有区分度的是：
  - `action_replacement_rate`
  - `shield_trigger_rate`
  - `avg_rec_action_count`
  - `dead_end_rec_rate`
  - `shield_penalty_rate`

这说明：

1. `recursive` 确实比 `safe` 更保守；  
2. `recursive` 确实更强调递归可行性保持；  
3. `recursive` 目前也确实带来了更高任务代价。  

这组现象非常重要，因为它直接支持后续两点：

- 为什么需要 `progressive shield`：缓解强保守性带来的性能损失；
- 为什么需要 `risk-aware scheduling`：让更强保守机制只在必要时介入。

换句话说，当前 baseline 的作用不是证明 recursive 已经最好，而是证明：

> 现有指标体系已经能稳定区分“更保守 / 更强调递归可行性 / 带来额外任务代价”这三类特征。

### 9.4.1 2026-04-20：针对 `dead_end_hard_rate / dead_end_rec_rate` 的一次定向实现优化

这一轮工作不是切换科研主线，也不是引入新模型，而是在当前 `MAPPO + shield` 主线下，对 `A_hard` 和 `A_rec` 的实现细节做一次**最小侵入式修正**，目标是减少“实现层假死局”。

这次优化前的两个核心判断是：

- `dead_end_hard_rate` 偏高，并不一定说明当前状态真的无一步 hard-safe 动作；
- 很多 `A_hard = \varnothing` 更可能来自当前顺序裁决方式过于刚性；
- `dead_end_rec_rate` 偏高，也不一定说明真的不存在 future-safe continuation；
- 很多 `A_rec = \varnothing` 更可能来自 `_future_safe_exists()` 只保留单条贪心 witness，导致把“存在 future-safe 延续”的状态误判成无解。

因此，这一轮实现优化的重点不是放松 hard-safe 判据，而是**改善联合裁决与递归可行性近似器本身**。

#### 1. 对 `A_hard` 的实现优化

当前 `A_hard` 仍然保持为 always-on 的一步 hard-safe 动作集，但联合裁决方式做了两点增强：

1. 新增 `agent adjudication ordering`
   - 支持 `fixed`
   - 支持 `most_constrained_first`
   - 当前默认切到 `most_constrained_first`

2. `most_constrained_first` 的排序只依赖 cheap 特征，不引入昂贵搜索：
   - `valid_action_count` 少的 agent 优先；
   - `near_boundary` 的 agent 优先；
   - `local_threat_count` 更高的 agent 优先；
   - `local_uav_count / crowded` 更高的 agent 优先。

3. 新增一个 very small local repair 机制：
   - 当当前 agent 的 `A_hard = \varnothing` 时，不再立刻认定死局；
   - 允许回看最近 `1-2` 个已裁决 agent；
   - 只在它们原本 admissible 的动作里试少量替代 witness；
   - 若修复成功，再重新计算当前 agent 的 `A_hard`。

这一步的意义是：

- 不改变 hard-safe 约束本身；
- 不用 unsafe 动作去“填满” `A_hard`；
- 但能减少“因为顺序过差而导致的实现层空集”。

#### 2. 对 `A_rec` 的实现优化

当前 recursive 语义仍然保持为：

- `recursive(full)`：每步都算 `A_rec`
- `recursive(risk)`：只在高风险时才从 `A_hard` 升级到 `A_rec`
- `recursive(legacy)`：保留作兼容模式

但 `_future_safe_exists()` 的内部近似方式做了增强：

1. 从单 witness 改成多 witness
   - 未来 agent 不再只取 `hard_actions[0]`
   - 改为尝试少量多个候选，例如：
     - base action witness
     - clearance 最大的 witness
     - top-k 高 clearance witness

2. 新增 small beam
   - 不再只保留一条 future candidate sequence；
   - 改为保留极小宽度的 beam；
   - 当前默认 beam width 很小，只是为了减少假空集，不是做真正的 look-ahead 搜索。

这一步的意义是：

- 不引入 `n-step` look-ahead；
- 不改变当前 recursive 的主线定义；
- 只是让当前 `H=1` 的 recursive feasibility checker 更不容易把“存在 continuation”的状态误判为无 continuation。

#### 3. 这一轮实现中明确保持不变的语义

这一轮实现优化特意保持了下面这些底线不变：

- `A_hard` 仍然是 always-on 的一步 hard-safe 基线，risk gate 不能关闭它；
- shield 仍然只是 action-set filter，不直接替 Actor 选“最优安全动作”；
- Actor 先输出原始偏好，若原始动作不允许，再在允许动作集内重选；
- `fail_closed / emergency` 的显式 dead-end 语义保持不变；
- 没有恢复 silent fallback；
- 没有引入真正的 `n-step` look-ahead。

#### 4. 新增配置与统计

为支撑这轮优化，当前实现新增了最小必要配置：

- `shield_adjudication_order`
- `shield_hard_repair_enabled`
- `shield_hard_repair_depth`
- `shield_future_witness_mode`
- `shield_future_beam_width`
- `shield_future_witness_top_k`

同时新增并打通了诊断统计：

- `hard_repair_attempt_count`
- `hard_repair_success_count`
- `hard_repair_success_rate`
- `future_witness_branch_count`
- `avg_future_witness_branch_count`
- `future_beam_width_used`

这些统计已经进入 `validate / summary CSV / TensorBoard` 链路，后续可以直接用于回答：

- `dead_end_hard_rate` 是否因为更合理的联合裁决而下降；
- `dead_end_rec_rate` 是否因为更强的 multi-witness / beam 近似而下降；
- 当前 repair 与 beam 到底带来了多少额外计算代价。

#### 5. 当前对这轮优化的定位

这轮工作更适合被定位为：

> 对分层 shield 实现语义的一次必要收紧与近似器修正，用于减少“实现层假死局”，从而让 `off / safe / recursive(full) / recursive(risk)` 的 baseline 更干净、更可解释。

它的价值主要体现在：

- 帮助 `A_hard` 更接近“真正的一步 hard-safe 联合允许集”；
- 帮助当前 `A_rec` 更接近“存在 future-safe continuation 时不轻易误杀”的近似版本；
- 为后续 profiling、baseline 对比和理论命题整理提供更干净的实现底座。

---

### 9.5 当前最合理的推进顺序

结合当前代码状态和已有验证结果，后续最稳妥的推进顺序应当是“先验证，再扩展”。

#### Step 1：先把 `off / safe / recursive` baseline 跑稳

目标：

- 先把 smoke 升级成可复现的 baseline；
- 明确各模式在安全、任务和计算代价上的差异；
- 让指标体系先稳定下来。

这一步的意义是给后续风险 gate 和更强 shield 提供可靠参照系。

#### Step 2：先把当前风险 gate 校准清楚

目标：

- 先把 recursive gate formalize 成连续风险阈值机制；
- 继续验证 `clear + region + hist` 的有效性；
- 明确 legacy gate 与 risk gate 的差异；
- 再逐步补 feasibility-proxy、uncertainty 等候选分量。

在这一步之前，不应过早把主精力切到完整的 progressive scheduling。

#### Step 3：再推进更强的 shield 层

目标：

- 先把当前 `safe` 和 `recursive` 的语义边界彻底跑通；
- 再扩展到 `H=2` 的 small-horizon look-ahead shield；
- 继续改进 `_greedy_future_safe_exists` 一类近似判断；
- 明确不同 horizon 带来的收益与代价。

只有当更强 shield 的收益能够被稳定测到时，progressive 才有坚实对象可调。

#### Step 4：再做 progressive conservativeness

这一步是在 `safe / recursive / look-ahead` 三层已经清晰之后，再去回答“如何 progressive”。

目标：

- 根据训练进度逐步提升保守层级；
- 决定何时更多启用 recursive / look-ahead；
- 明确保守性增强的节奏和触发条件。

#### Step 5：最后再做更完整的 risk-aware dual scheduling

这一步才真正把 shield 和 progressive 进一步统一成双调度问题，例如：

- `shield intervention strength`
- `action-space shrinkage level`

更后面的扩展再考虑：

- `decentralized shield`
- rollout 可视化与更细的诊断链路


### 9.6 当前最稳妥的论文总定位

结合以上修正，当前论文最稳妥、最统一的定位可以写成：

> 面向多无人机协同搜索任务，本文围绕“训练进程感知的保守性增强、短期前瞻可行性维护和风险自适应双调度”构建统一的 shield 框架，在始终保持一步硬安全约束的前提下，平衡安全性、递归可行性、任务性能 与在线计算代价。

如果进一步压缩成一句话，则为：

> 本工作的核心不是将 exact recursive shield 做成更快的在线主决策器，而是在保持原主线创新结构的前提下，将其实现方式改造成更可计算、更可扩展、更适合多 UAV 协同搜索场景的统一安全决策框架。

---

### 9.7 下一篇论文中 risk 优化的候选创新点

结合当前 `baseline_v1 / v_next / vnext_tune_*` 的离线验证结果，risk 下一篇论文不应再简单表述为“继续调权重”，而应明确围绕下面这个核心矛盾展开：

> 当前 risk score 已经表现出一定的排序潜力，但这种排序潜力并不能自然转化为一个稳定、可部署、可控预算的 runtime gate。

也就是说，下一篇 risk 论文真正值得写的，不是“把分数再调高一点”，而是系统回答：

- 为什么 `exact top-k` 排序上看起来更强的 risk，在单阈值 gate 下会明显退化；
- 如何把 `need_rec` 的离线排序能力，稳定映射到在线可部署的 `A_hard -> A_rec` 升级决策；
- 如何让 risk 不仅有“分数”，还具备 `budget-aware / calibration-aware / deployment-aware` 的 gate 语义。

可以考虑凝练成以下几个工作包。

#### Work Package A：Risk Ranking 与 Runtime Gate 的统一建模

目标：

- 明确区分 `ranking quality` 与 `threshold realizability`；
- 解释为什么某些 risk 在 `eligible-only exact top-k` 上很强，但在单阈值 `eta` 下不稳定；
- 设计从风险排序到 gate 行为的一致性指标，而不只看 `precision / recall`。

可作为创新点的表述：

- 提出一套面向 `A_hard -> A_rec` 升级问题的双视角评估框架，同时刻画
  - `exact top-k eligible ranking`
  - `single-threshold runtime realizability`
- 证明风险函数的“离线排序能力”和“在线阈值可实现性”并不等价。

#### Work Package B：Budget-Aware / Calibration-Aware Risk Gate

目标：

- 不再只依赖固定阈值 `eta`；
- 研究给定目标 gate budget 时，如何稳定控制 `recursive_gate_rate`；
- 让 risk score 具备更好的校准性，而不是只具备相对排序性。

可探索方向：

- `budget-aware thresholding`
- 分位数阈值 / percentile gate
- 分 bucket 的局部校准
- 将 `eligible_gate_rate` 显式作为控制目标

可作为创新点的表述：

- 将 risk-aware recursive shielding 从“阈值触发”提升为“预算约束下的可校准升级机制”。

#### Work Package C：面向 `need_rec` 的更精细标签体系

目标：

- 不再只用单一 `need_rec` 标签；
- 区分
  - `eligible and need_rec`
  - `dead_end_hard / A_hard empty`
  - `ineligible but A_hard nonempty`
- 研究不同 bucket 对风险学习和 gate 行为的影响。

这一点很重要，因为当前结果已经表明：

- `dead_end_hard`
- `ineligible_nonempty`

这两类样本在安全上重要，但与 `A_hard -> A_rec` 升级语义不同，混在一起会污染 risk 结论。

可作为创新点的表述：

- 提出一种分桶式风险语义，将 `recursive upgrade risk` 与 `hard-mask failure risk` 区分建模。

#### Work Package D：Risk 分量从“几何直觉”走向“可部署判别器”

目标：

- 在不引入额外风险网络的前提下，继续挖掘 cheap 但更有效的 risk 分量；
- 优先使用当前 shield 流水线里已存在的中间统计；
- 强调“cheap + explainable + deployable”，而不是“再堆一个模型”。

当前最值得继续的方向包括：

- `proposed_action` 相关的精细 clearance / support 统计；
- 与 `A_hard` 收缩程度相关的局部 fragility 指标；
- 风险分数的分段化或分桶化组合，而不是单纯线性加权；
- 面向 gate 的 monotonic calibration，而不是只面向 raw score。

可作为创新点的表述：

- 提出一种基于 shield 内部可复用统计的轻量级解释性 risk family，用于在线 gate 决策。

#### Work Package E：从 Risk Gate 走向 Dual Scheduling

目标：

- 将当前 risk 只用于 `A_hard -> A_rec` 升级判断，进一步扩展为更完整的调度信号；
- 为后续的 `progressive conservativeness + dual scheduling` 提供可复用接口。

后续可接的方向包括：

- risk 决定是否启用 `A_rec`
- 训练进度决定保守层级上限
- 两者共同决定 `intervention strength` 与 `action-space shrinkage level`

这一点适合作为下一篇论文与当前主线之间的桥梁：

- 当前主线论文：先把 progressive shielding 主线跑稳
- 下一篇 risk 论文：重点回答 risk 如何从“分数”成长为“可调度、可部署、可校准的 gate 信号”

#### 当前对下一篇 risk 论文的最合理定位

如果把后续 risk 工作压缩成一句话，当前最值得推进的定位是：

> 下一篇论文不再把重点放在“提出一个新的 risk 分数公式”，而是研究如何将轻量、可解释的风险排序能力，转化为可预算控制、可阈值校准、可在线部署的 recursive shield 升级机制。

这条路线的好处是：

- 与当前主线论文形成明显区分；
- 不需要切换到 learned risk network 或额外 teacher 路线；
- 更容易形成一套独立而完整的实验问题、指标体系和理论叙事。

#### 当前主线中对 risk 的冻结建议

在当前主线论文里，risk 建议先冻结为“可用版本”，不要无限展开。当前最值得作为 runtime 候选推进的是：

- `v_next` 家族中的 `vnext_tune_proxy_prop_gap_region`
- 对应 runtime 配置为：
  - `risk_variant = v_next`
  - `risk_threshold = 0.35`
  - `risk_vnext_weight_prop_clear = 0.40`
  - `risk_vnext_weight_clear_gap = 0.35`
  - `risk_vnext_weight_support = 0.00`
  - `risk_vnext_weight_region = 0.25`

它不是离线排序最强的版本，但目前是最适合作为在线单阈值 recursive gate 候选继续做正式实验的版本。
