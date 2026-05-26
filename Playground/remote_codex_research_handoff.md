# Remote Codex 科研交接摘要

## 1. 项目背景

当前工作围绕论文《UAV Swarm Cooperative Search for Moving Targets via Hybrid-Rewards Deep Reinforcement Learning》展开，目标是在其多无人机协同搜索框架上，加入一条新的安全增强研究主线，而不是简单做工程补丁。

当前已经明确的研究目标是：在保留前期探索能力的同时，提高训练后期和部署阶段的安全性与可行性。

---

## 2. 已确定的主研究主线

请始终沿着下面这条主线推进，不要切换到别的方向：

1. `progressive shielding`
2. `recursive-feasible / small-horizon look-ahead shield`
3. `risk-aware dual scheduling`

其中：

- `progressive` 的含义是“随着训练推进，shield 的保守性逐渐增强”，不是“前期允许 unsafe action 通过”。
- `hard-safe` 的一步安全约束应始终保留。
- 真正逐步增强的是 `A_hard -> A_rec -> A_H^{viable}` 这一层级上的保守性，而不是放松安全。

---

## 3. 明确不作为主线的方向

以下方向可以作为工程参考，但不要作为当前论文主贡献切入点：

- learned safe mask
- offline teacher / offline safe supervisor
- 额外训练一个 learned risk network
- 大幅切换到另一套安全 RL 框架

原因：这些路线与已有工作重叠更大，容易偏离当前已经形成的论文叙事。

---

## 4. 已确定的关键语义与理论口径

### 4.1 Shield 的作用方式

不要把 shield 设计成“直接替 Actor 选最优安全动作”。

正确语义应为：

1. Actor 先输出原始动作偏好；
2. shield 根据当前规则构造允许动作集；
3. 若原始动作不在允许集内，则 Actor 在允许集内重新选择；
4. 同时记录 shield 触发次数，并可基于触发添加惩罚。

也就是说，shield 是 `action-set filter`，不是外部专家控制器。

### 4.2 理论分层

当前理论分层已经基本明确：

- `A_hard`：always-on、cheap、一步 hard-safe 动作集
- `A_rec`：在 `A_hard` 基础上进一步保证“下一步仍存在安全动作”的 recursive-feasible 动作集
- `A_H^{viable}`：更小 horizon 的 look-ahead viable 动作集

研究重点不是“是否加 shield”，而是“如何在安全、递归可行性和训练探索之间做分层调度”。

### 4.3 Progressive 的正确解释

progressive shielding 应理解为：

\[
A^{hard}
\rightarrow
A^{rec}
\rightarrow
A_H^{viable}
\]

即：训练后期逐步提高保守性，而不是早期放开 unsafe action。

---

## 5. 当前代码方向上的关键演进

### 5.1 已有 shield 基础机制

代码侧已经做过以下工作：

- 增加了 shield 机制；
- 保留了 `off / safe / recursive` 三种模式；
- 当前语义是：
  - `off`：不走 shield
  - `safe`：只做 always-on 的一步 hard-safe 过滤
  - `recursive`：在 hard-safe 基础上，必要时进一步做 future-safe / recursive 检查

### 5.2 已做过的 baseline smoke

已经做过小规模 smoke baseline，对比了 `off / safe / recursive`。

目前最能区分 `safe` 和 `recursive` 的，不是 `collision_count`，而是：

- `action_replacement_rate`
- `shield_trigger_rate`
- `avg_rec_action_count`
- `dead_end_rec_rate`
- `shield_penalty_rate`

现有结论是：

- `recursive` 更保守，干预更多；
- 它在递归可行性相关指标上更好；
- 但 smoke 级别实验下任务回报未必更高；
- 说明现在这套指标已经可以区分“更强可行性约束”和“额外任务代价”。

---

## 6. 已做过的性能优化

后续代码已经围绕“让 shield 跑得动”做过一轮优化，重点不是换研究主线，而是让现有主线能进入正式训练。

已知优化包括：

- 将原来的逐动作在线验证流程，改成 `rule-based hard mask + gray-zone refine`
- recursive 检查改成条件触发，而不是每步全开
- 增加 cache
- 增加候选动作剪枝 / top-k candidate pruning
- 保留 legacy exact 路径做 benchmark
- 将性能 profile 接入训练和评估汇总

已报告的性能结果：

- `safe`：约 `10.5 -> 36.8 steps/s`
- `recursive`：约 `8.5 -> 9.3 steps/s`
- `off`：约 `72 steps/s`

当前瓶颈：

- `off` 已基本不受 shield 慢路径拖累
- `safe` 已明显加速
- `recursive` 仍偏重，主要成本在 rule mask 本身和 future-safe 检查

---

## 7. 当前 risk 方向的正式决定

### 7.1 风险函数的主决策

当前决定先从一个低成本、可解释的连续风险函数开始，而不是训练风险网络。

第一版正式风险函数为：

\[
\xi_{i,t}
=
w_1 \xi_{i,t}^{clear}
+
w_2 \xi_{i,t}^{region}
+
w_3 \xi_{i,t}^{hist}
\]

推荐初始权重：

\[
w_1 = 0.5,\quad w_2 = 0.3,\quad w_3 = 0.2
\]

### 7.2 三个分量的定义

#### clear

\[
\xi_{i,t}^{clear}
=
\mathrm{clip}\left(1-\frac{m_{i,t}}{M_c},0,1\right)
\]

其中 `m_{i,t}` 为当前 agent 的几何安全余量，例如 `min_candidate_clearance`。

#### region

\[
\xi_{i,t}^{region}
=
\frac{\mathbb I_{boundary} + \tilde n^{threat}_{i,t} + \mathbb I_{crowded}}{3}
\]

这里强调局部区域风险，而不是全图扫描。

#### hist

\[
\xi_{i,t}^{hist}
=
\frac{1}{W} \sum_{\tau=t-W}^{t-1} \mathbb I_{shield}(i,\tau)
\]

注意这里应只使用过去窗口，不包含当前步，否则会形成自引用。

### 7.3 暂缓项

以下两项已被明确列为后续 TODO，不是当前第一优先级：

- `feasibility-proxy`
- `uncertainty / preference-conflict`  

可能形式包括：

\[
\xi_{i,t}^{feas-proxy} = 1 - \frac{|A_i^{rule}(s_t)|}{|A_i|}
\]

以及基于 top-k survival、logit gap、Q-gap 的不确定性度量。

---

## 8. risk 函数当前实现状态

已经有一版代码实现了 `clear + region + hist` 风险函数，并用它来驱动 recursive gate。

实现要点如下：

- `clear` 复用现有 `min_candidate_clearance`
- `region` 复用现有 `near_boundary / local_threat_count / crowded`
- `hist` 采用每个 agent 的 shield 干预滑动窗口历史
- 风险指标已写入训练 / 验证 / CSV / TensorBoard

已报告的新增指标包括：

- `avg_risk_score`
- `avg_risk_clear`
- `avg_risk_region`
- `avg_risk_hist`
- `high_risk_rate`
- `recursive_gate_rate`
- `high_risk_agent_count`
- `recursive_gate_agent_count`

当前 smoke 中已有示例值，说明链路是通的。

---

## 9. 关于 A_safe 与 A_hard 的关键澄清

这是当前非常重要的一点，请优先理解并在代码和论文表述中统一。

### 9.1 为什么要澄清

当前实现里“每步都计算的一步安全动作集”过去常被写成 `A_safe`，但这容易和更一般的理论 safe set 混淆。

为了避免混乱，应将当前实现层的 always-on 这一层明确命名为：

- `A_hard`

然后区分：

- `A_hard`：每步都算的 cheap hard-safe 层
- `A_rec`：高风险时才进一步做的 recursive future-safe 层

### 9.2 当前正确流水线

当前应该统一理解为：

1. Actor 输出动作偏好；
2. 构造 `A_hard`；
3. 基于 `A_hard` 及其几何统计量计算当前风险分数；
4. 若处于 `recursive` 模式且风险高，则再在 `A_hard` 基础上计算 `A_rec`；
5. 若 `A_rec` 为空，则回退到 `A_hard`；
6. Actor 在最终允许动作集内重新选择。

### 9.3 这意味着什么

这意味着当前实现的风险是：

- `post-A_hard`
- `pre-A_rec`

而不是“先算风险，再决定是否计算 hard-safe”。

也就是说：

- 风险决定的是“是否升级到更强 shield”
- 不是“是否启用一步 hard-safe”

---

## 10. 当前主要局限

请把这些局限明确视为当前状态，而不是已经解决的问题：

1. 当前 `A_rec` 不是精确未来联合可行性判断，而是 greedy future-safe existence 近似。
2. threat 是随机移动的，但 hard constraint 主要检查的是当前 threat 位置下的一步后状态，没有显式展开 threat 随机迁移。
3. 还没有做 decentralized shield。
4. 还没有把 shield 接到 rollout 可视化链。
5. 当前 `A_safe / A_rec` 的统计口径不是精确联合动作集大小，而更像 centralized sequential adjudication 下的逐 agent 统计。
6. `dead_end_safe_count / dead_end_rec_count` 当前记录的是“至少一个 agent 对应动作集为空”的 step，不是精确联合 dead-end。
7. `hist` 必须只用过去窗口，理论文档和代码都要保持一致。
8. 当前代码里如果 `A_hard` 为空，仍可能保留 legacy fallback 到 valid-mask 的行为；这会与“hard-safe 始终保证安全”的强理论表述产生张力，需要谨慎处理。

---

## 11. 当前最合理的科研推进顺序

请按下面顺序推进，而不是跳线：

1. 跑正式 baseline：
   - `off`
   - `safe`
   - `recursive + legacy gate`
   - `recursive + risk gate`
2. 先校准风险函数，而不是立刻上复杂调度：
   - 扫 `risk_threshold`
   - 扫 `w1,w2,w3`
   - 扫 `risk_hist_window`
3. 明确 `A_hard` 是否足够 cheap：
   - 如果 cheap，就保留 always-on
   - 如果仍不够 cheap，再讨论更进一步的前置风险或更粗 rule mask
4. 在 risk gate 稳定后，再推进 `H=2` small-horizon look-ahead shield
5. 再在此基础上做 `progressive conservativeness`
6. 最后才做更完整的 `dual scheduling`

---

## 12. 远程 Codex 接手时的优先任务

远程 Codex 接手后，建议先做以下几件事：

1. 先阅读本文件，再读取代码中的 `shield.py / config.py / main.py / stats.py / validate.py`。
2. 确认当前 `clear + region + hist` 风险函数实现是否与本摘要一致。
3. 将当前实现层的 always-on 安全集合明确统一为 `A_hard` 命名，避免与理论 `safe set` 混淆。
4. 测一下 `A_hard` 的真实计算代价，判断它是否足够 cheap。
5. 如果 `A_hard` 的代价可接受，则后续继续沿着“risk gate 决定是否从 `A_hard` 升级到 `A_rec`”这条线推进。

---

## 13. 对远程 Codex 的明确约束

请严格遵守以下约束：

- 不要偏离当前三点主线
- 不要把 shield 改成直接替 Actor 选动作
- 不要把 progressive 理解成前期允许 unsafe
- 不要贸然把主线切到 learned safe mask / offline teacher
- 优先做最小侵入式修改
- 任何理论与实现口径不一致的地方，要显式指出，不要默默混过去

---

## 14. 一句话总结

当前最关键的科学问题已经从“要不要加 shield”转变为：

如何在 `A_hard` 始终保障一步安全的前提下，利用低成本连续风险分数，按需升级到 `A_rec` 乃至小 horizon look-ahead，并进一步结合训练进度实现 progressive conservativeness。
