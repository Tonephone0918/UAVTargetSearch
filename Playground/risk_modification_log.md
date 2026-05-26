# Risk Function 修改记录

## 1. 文档目的

本文件用于记录当前 risk function 的演进过程、评估协议、关键指标和阶段性结论，作为后续继续调阈值、调权重、补分量和推进 `risk-aware dual scheduling` 的依据。

当前统一语义如下：

- `A_hard`：每步都计算的、cheap、always-on 的一步 hard-safe 动作集
- `A_rec`：只在需要时才进一步计算的 recursive future-safe 动作集
- 风险分数的角色：`post-A_hard / pre-A_rec`
- shield 语义不变：Actor 先给原始动作偏好，shield 只构造允许动作集，若原始动作不允许，再在允许集内重选

补充说明：

- 当前验证使用的 checkpoint 是 `/home/ps/thf_code/UAVTargetSearch/checkpoints/_perf_train_recursive/best.pt`
- 这是一份已有 checkpoint 的评估验证，不是“加入 risk 后重新正式训练”的结果

---

## 2. 评估协议

### 2.1 Online 运行时 gate 扫描

结果文件：

- `Playground/risk_gate_validation_perf_train_recursive.json`

特点：

- 直接在运行时用风险分数驱动 recursive gate
- 统计 `gate_rate / precision / recall / perf_recursive_time_ms / perf_steps_per_sec`
- `need_rec` 的 oracle 仍由全量 `A_rec_oracle` 打标

### 2.2 固定 `safe` 轨迹离线打标

结果文件：

- `Playground/risk_gate_offline_safe_validation_perf_train_recursive.json`
- `Playground/risk_gate_offline_safe_validation_v3_perf_train_recursive.json`

特点：

- 先固定一条 `safe` 轨迹，再对同一批 agent-step 做 oracle 打标
- 这样可以避免不同 gate 反过来改变轨迹分布，便于比较 risk 本身的判别力
- 当前轨迹统计：
  - `agent_step_count = 200`
  - `eligible_agent_step_count = 183`
  - `eligible_agent_step_rate = 0.915`
  - `need_rec_count = 15`
  - `need_rec_rate = 0.075`

### 2.3 当前使用的标签与指标

- `need_rec = 1`：`proposed_action ∈ A_hard` 且 `proposed_action ∉ A_rec_oracle`
- `eligible`：`proposed_action ∈ A_hard`
- `Eligible Precision`：只在 `eligible` 样本上计算的 precision
- `wasted_gate_rate`：gate 打开了，但并没有命中真正 `need_rec` 的比例

---

## 3. 风险版本时间线

### 3.1 baseline_v1：`clear(min_candidate) + region + hist`

构成：

- `risk = 0.5 * clear_min_candidate + 0.3 * region + 0.2 * hist`
- `clear` 来源于 `min_candidate_clearance`
- `region` 来源于边界/局部 threat/拥挤度等局部特征
- `hist` 来源于过去窗口中的 shield 介入历史

语义定位：

- 这是第一版正式启用的连续风险函数
- 风险计算位置是 `A_hard` 之后、`A_rec` 之前

Online 结果：

- 最优阈值（按 `precision + recall`）是 `0.35`
- `need_rec_rate = 5.5%`
- `gate_rate = 11.0%`
- `precision = 9.1%`
- `recall = 18.2%`
- `perf_recursive_time_ms = 7.48`
- `perf_steps_per_sec = 55.05`

对照性能：

- `recursive + legacy gate`：`gate_rate = 15.0%`，`perf_steps_per_sec = 49.89`
- `recursive + risk gate`：`gate_rate = 11.0%`，`perf_steps_per_sec = 55.05`
- `recursive_always`：`gate_rate ≈ 99.5%`，`perf_steps_per_sec = 14.56`

Offline 固定 `safe` 轨迹结果：

- 最优阈值（按 `precision + recall`）是 `0.2`
- `gate_rate = 20.5%`
- `eligible_gate_rate = 13.66%`
- `wasted_gate_rate = 8.0%`
- `precision = 7.3%`
- `eligible_precision = 12.0%`
- `recall = 20.0%`

阶段结论：

- v1 可以明显省掉一部分 recursive 代价，比 `always recursive` 和 `legacy gate` 都更省
- 但它对 `need_rec` 的判别力偏弱，precision 和 recall 都不够高
- 固定轨迹结果说明问题主要不在“阈值没扫到”，而在风险分量本身区分度有限

### 3.2 ablation_hist0：去掉 `hist`

构成：

- `risk = 0.5 * clear_min_candidate + 0.3 * region + 0.0 * hist`
- 不做权重重归一化，直接把 `hist` 强制为 `0`

Offline 固定 `safe` 轨迹结果：

- 最优阈值是 `0.2`
- `gate_rate = 19.0%`
- `eligible_gate_rate = 12.57%`
- `wasted_gate_rate = 7.5%`
- `precision = 7.9%`
- `eligible_precision = 13.0%`
- `recall = 20.0%`

阶段结论：

- 相比 baseline_v1，`hist=0` 有轻微改善
- 这说明在当前固定 `safe` 轨迹评估下，`hist` 对判别 `need_rec` 的帮助不明显，甚至可能引入噪声
- 因此后续验证里，`hist` 不应默认被视为“必选有效项”

### 3.3 v2_proposed_action_clearance：把 `clear` 换成 proposed-action 版本

构成：

- `risk = 0.5 * clear_proposed_action + 0.3 * region + 0.2 * hist`
- 只在 `proposed_action ∈ A_hard` 时启用 proposed-action clearance；否则 clear 项记为 `0`

Offline 固定 `safe` 轨迹结果：

- 最优阈值是 `0.5`
- `gate_rate = 1.0%`
- `eligible_gate_rate = 1.09%`
- `wasted_gate_rate = 0.0%`
- `precision = 50.0%`
- `eligible_precision = 50.0%`
- `recall = 6.7%`

补充：

- 在较低阈值 `0.2` 下，它的 `gate_rate = 11.0%`，`eligible_precision = 10.0%`，`recall = 13.3%`

阶段结论：

- v2 的 clear 项更贴近“当前 proposed action 自身是否危险”，因此 precision 很高
- 但它太窄，只能抓到一小部分非常明显的样本，导致 gate rate 和 recall 都太低
- 结论是：它可以作为高置信度补充信号，但不适合作为单独主 gate

### 3.4 v3_hybrid_clear：`clear_min + clear_prop + region`，去掉 `hist`

构成：

- `risk = 0.50 * clear_min + 0.35 * clear_prop + 0.15 * region`
- `hist = 0`

Offline 固定 `safe` 轨迹结果：

- 最优阈值（按 `precision + recall`）是 `0.65`
- `gate_rate = 1.0%`
- `eligible_gate_rate = 1.09%`
- `wasted_gate_rate = 0.0%`
- `precision = 50.0%`
- `eligible_precision = 50.0%`
- `recall = 6.7%`

补充：

- 在阈值 `0.2` 下，`gate_rate = 19.0%`，`eligible_precision = 13.0%`，`recall = 20.0%`
- 但在中间阈值区间，整体区分度依然不稳定

阶段结论：

- 把 `clear_min` 和 `clear_prop` 混合起来，并没有明显提升总体判别力
- 它在低阈值时基本退化回 `hist=0` 的表现，在高阈值时又变成极窄 gate
- 因此 `hybrid clear` 本身还不是决定性改进

### 3.5 v3_hybrid_clear_fragility：在 hybrid clear 上补 `hard-set fragility`

构成：

- `risk = 0.35 * clear_min + 0.25 * clear_prop + 0.25 * fragility + 0.15 * region`
- `fragility = 1 - |A_hard| / |A_valid|`
- `hist = 0`

Offline 固定 `safe` 轨迹结果：

- 最优阈值（按 `precision + recall`）是 `0.2`
- `gate_rate = 18.0%`
- `eligible_gate_rate = 10.93%`
- `wasted_gate_rate = 8.0%`
- `precision = 8.3%`
- `eligible_precision = 15.0%`
- `recall = 20.0%`

另一种阈值选择：

- 若按 `eligible_precision + recall` 选阈值，最佳阈值是 `0.65`
- 此时 `gate_rate = 3.5%`
- `eligible_precision = 100.0%`
- `recall = 6.7%`

阶段结论：

- 这是目前几版里相对最平衡的一版
- 在保持 `recall = 20.0%` 的同时，`eligible_precision` 从 baseline_v1 的 `12.0%` 提升到 `15.0%`
- 但整体仍不足以说明“risk 已经很好地识别出需要升级到 `A_rec` 的样本”

---

## 4. 当前综合判断

截至目前，可以形成以下阶段性结论：

1. 当前 risk gate 有工程价值，但判别力仍偏弱。  
   它已经能减少 recursive 开销，并且显著好于 `always recursive` 的纯暴力做法，但还不够强，不能说已经把 `need_rec` 识别得很好。

2. `hist` 在固定 `safe` 轨迹上没有体现出正收益。  
   它更像是一个容易受轨迹分布影响的滞后信号，而不是当前最核心的判别项。

3. `proposed_action_clearance` 有高 precision，但覆盖面太窄。  
   它适合做高置信度补充特征，不适合单独承担主 gate。

4. `fragility` 值得保留为后续候选分量。  
   虽然提升幅度不大，但它比单纯 `clear + region + hist` 更贴近“`A_hard` 已经开始失去余量”的语义。

---

## 5. 当前最值得继续做的方向

基于现有结果，下一步最值得优先推进的是：

1. 继续围绕 `A_hard` 的余量与脆弱性补强风险分量，而不是贸然引入新网络。  
   当前最缺的不是更复杂模型，而是更能区分“`A_hard` 看起来安全，但实际上值得升级到 `A_rec`”的 cheap 特征。

2. 优先继续做固定 `safe` 轨迹离线打标验证。  
   这条评估协议更干净，最适合判断 risk 本身有没有判别力。

3. 把 risk v1 视为可保留的工程版 gate，而不是已经定型的最终论文版 gate。  
   它适合作为当前 `recursive gate v1` 的实现起点，但还需要继续调权重、调阈值、补分量。

---

## 6. 对后续文档与实现的提醒

- 理论和实现里都应优先使用 `A_hard -> risk gate -> A_rec` 的表述
- `safe` 模式应表述为 `hard-safe-only mode`
- `hist` 项若继续保留，理论上应只使用过去窗口，避免自引用
- 当前 `A_rec_oracle` 是在顺序裁决语义下做的 oracle 打标，不是严格的联合动作全集 oracle
