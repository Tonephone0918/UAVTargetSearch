# 结果备忘 v1（中文）

## 0. 使用原则

1. `episode_return` 不是当前跨全部目录的主排序指标。
   原因：旧 baseline 与后续 `risk_base` 系列存在 DPM reward normalization 口径差异。
2. 正文主排名优先使用任务、安全与 shield-behavior 指标：
   `search_rate`、`collision_count`、`guarantee_broken_rate`、`dead_end_rec_rate`、`recursive_gate_rate`、`perf_shield_time_ms`
3. 若不同目录的结果口径不同，必须显式标出：
   - `single-checkpoint compare`
   - `training-seed compare`
   - `formal compare`
   - `diagnostic / validate-only`

## 1. 当前引用的结果目录与口径

### A. progressive 主线正式比较

1. `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5_raw/per_seed_metrics.csv`
   - 口径：`3 training seeds x 5 eval seeds x 5 episodes`
   - 含义：raw final table；每一行对应一个 `training_seed x eval_seed`
   - 建议用途：主结果的底层原始来源

2. `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/per_checkpoint_metrics.csv`
   - 口径：每个 training seed 的 best checkpoint，先在 `5 eval seeds x 5 episodes` 上聚合
   - 含义：观察训练种子间波动
   - 建议用途：训练种子层面的方差和不确定性

3. `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`
   - 口径：对 3 个 training seeds 再聚合后的 summary
   - 含义：旧版 final summary
   - 注意：任务/安全指标基本可用，但部分 `perf_*` 聚合与 raw 重聚合不完全一致

4. `runs/progressive_mechanism_20260428/summary_metrics.csv`
   - 口径：从 raw / checkpoint schedule / stage metrics 回填后的机制分析汇总
   - 含义：专门用于 progressive stage 与 re-aggregated perf 统计
   - 建议用途：正文的机制图和 runtime 描述

5. `runs/progressive_mechanism_20260428/stage_metrics.csv`
   - 口径：按 `early / mid / late / fixed` 切开的阶段统计
   - 含义：解释 `threshold_only_progressive` 与 `safeearly_progressive` 的实际课程差异
   - 建议用途：机制分析，不作为主表唯一来源

### B. dual 边界比较

1. `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/summary_metrics.csv`
   - 口径：`non_progressive` / `threshold_only_progressive` / `safeearly_progressive` / `threshold_only_dual_progressive`
   - 含义：dual 分支的正式 `3x5x5` 比较
   - 建议用途：discussion 或 appendix

2. `runs/formal_progressive_seed_compare_20260426/formal_compare_with_dual_multiseed5x5/per_checkpoint_metrics.csv`
   - 口径：dual 比较的 training-seed 级汇总
   - 建议用途：观察 dual 是否只是个别 seed 好看

### C. H1/H2 learned checkpoint compare

1. `runs/final_formal_h2_vs_h1_multiseed3x3/summary_metrics.csv`
   - 口径：固定 checkpoint，`3 eval seeds x 3 episodes`
   - 含义：H1/H2 learned policy 的单 checkpoint 正式比较
   - 建议用途：H2 边界结果

2. `runs/final_formal_h2_vs_h1_multiseed5x5/summary_metrics.csv`
   - 口径：固定 checkpoint，`5 eval seeds x 5 episodes`
   - 含义：更接近正式 compare 的 H1/H2 learned policy 比较
   - 建议用途：比 `3x3` 更稳，但仍属边界结果

### D. H1/H2 `2x2` cross-eval

1. `runs/h1_h2_cross_eval_multiseed3x3/summary_metrics.csv`
   - 口径：`H1/H2 checkpoint x H1/H2 shield`
   - 含义：区分“收益来自 runtime shield”还是“收益来自 learned checkpoint”
   - 建议用途：机制结论的关键证据

2. `runs/h1_h2_cross_eval_offdiag_multiseed3x3/summary_metrics.csv`
   - 口径：只保留 off-diagonal 两格
   - 含义：更直接看 `H1 ckpt + H2 shield` 与 `H2 ckpt + H1 shield`
   - 建议用途：正文图或 appendix 精简表

### E. H1/H2 matched gate-rate / compute-budget

1. `runs/h1_h2_fixedpoint_compare_stable3x3/summary_metrics.csv`
2. `runs/h1_h2_fixedpoint_compare_stable3x3/matched_gate_rate.csv`
3. `runs/h1_h2_fixedpoint_compare_stable3x3/matched_compute_budget.csv`
4. `runs/h1_h2_fixedpoint_compare_refine3x3/summary_metrics.csv`
5. `runs/h1_h2_fixedpoint_compare_refine3x3/matched_gate_rate.csv`
6. `runs/h1_h2_fixedpoint_compare_refine3x3/matched_compute_budget.csv`

- 口径：围绕 `H2@eta≈0.55` 与多个 H1 阈值点做“匹配 gate-rate / 匹配 compute-budget”对比
- 含义：检验 H2 是否全面支配 H1
- 建议用途：H2 作为 stronger layer 的机制边界证据

### F. H2 validate-only 阈值扫描与 runtime ablation

1. `runs/h2_risk_threshold_scan_medium2x2/summary_metrics.csv`
   - 口径：validate-only，小规模 medium `2x2`
   - 含义：寻找 H2 的候选阈值工作点
   - 建议用途：说明 `eta≈0.55` 不是拍脑袋来的

2. `runs/h2_future_runtime_ablation_20260427/compare_h2ckpt_3x3/summary_metrics.csv`
3. `runs/h2_future_runtime_ablation_20260427/partials/compare_h2ckpt_3x3/*/summary_metrics.csv`
   - 口径：固定 H2 checkpoint 的 runtime 组件消融
   - 含义：拆开 `refine_only`、`exact_empty_only`、`risk_trigger_only`、`viability_only`
   - 建议用途：appendix / discussion

### G. `A_hard` exact / sequential 诊断

1. `runs/model_compare_exact_hard_solver_fast2x1/summary_metrics.csv`
2. `runs/model_compare_exact_hard_solver_diag_medium2x2/summary_metrics.csv`

- 口径：小规模 exact solver 诊断
- 含义：比较 `sequential`、`exact`、`rescue`
- 建议用途：支撑“approximation-induced dead-end”与 exact projected `A_hard`

## 2. 口径说明：single-checkpoint / training-seed / formal compare

### `single-checkpoint compare`

- 含义：固定一个或几个具体 checkpoint，直接比较其评测表现
- 典型目录：
  - `final_formal_h2_vs_h1_multiseed3x3`
  - `final_formal_h2_vs_h1_multiseed5x5`
  - `h1_h2_cross_eval_multiseed3x3`

### `training-seed compare`

- 含义：每个 training seed 各自有一个 best checkpoint，先分别聚合，再比较训练种子间差异
- 典型文件：
  - `formal_compare_multiseed5x5/per_checkpoint_metrics.csv`
  - `formal_compare_with_dual_multiseed5x5/per_checkpoint_metrics.csv`

### `formal compare`

- 含义：在统一评测口径下，对多个 training seeds 再做 aggregate，得到最终方法级 summary
- 典型文件：
  - `formal_compare_multiseed5x5/summary_metrics.csv`
  - `formal_compare_with_dual_multiseed5x5/summary_metrics.csv`

## 3. 当前已确认的结论

1. shield 的当前语义是 `allowed-action filtering`，不是外部 planner 直接替 actor 选动作。
   - 代码依据：`src/hrvdn/shield.py`
   - 论文中可以写得比较确定

2. `A_hard` 是 always-on 底座，`safe` 只是“只停在 `A_hard` 层”，`recursive` 是在其上按 gate 升级到 `A_rec`。
   - 代码与文档一致

3. `threshold_only_progressive` 是当前最适合作为正文主结果的 progressive 分支。
   - 正式 `3x5x5` 口径下：
     - `search_rate ≈ 0.9987`
     - `collision_count ≈ 92.57`
     - `guarantee_broken_rate ≈ 0.332`
     - `dead_end_rec_rate ≈ 0.444`
   - 相比 `non_progressive`：
     - `guarantee_broken_rate` 更低（`0.332` vs `0.343`）
     - `dead_end_rec_rate` 更低（`0.444` vs `0.464`）
     - 但 `collision_count` 不更低（`92.57` vs `90.79`）

4. `safeearly_progressive` 不是当前更稳的主结果。
   - `search_rate` 可到 `1.0`
   - 但 `collision_count ≈ 94.73`，`guarantee_broken_rate ≈ 0.354`，`dead_end_rec_rate ≈ 0.467`
   - 整体不优于 `threshold_only_progressive`

5. H2 的 runtime layer 有价值，但 H2 learned checkpoint 还不是稳定主正结果。
   - `h1_ckpt_h2_shield` 在 `h1_h2_cross_eval_multiseed3x3` 中最强：
     - `search_rate = 1.0`
     - `collision_count ≈ 76.44`
     - `guarantee_broken_rate ≈ 0.316`
     - `dead_end_rec_rate ≈ 0.129`
   - 但 `h2_ckpt_h1_shield` 并没有打赢 `h1_ckpt_h1_shield`

6. exact / sequential 的 `A_hard` 诊断差异是真实存在的。
   - `model_compare_exact_hard_solver_diag_medium2x2` 中：
     - `seq_empty_exact_nonempty_rate` 大约在 `0.14` 到 `0.37`
     - `seq_nonempty_exact_empty_rate` 大约在 `0.24` 到 `0.42`
     - `seq_exact_jaccard` 大约在 `0.59` 到 `0.72`
   - 这足以支持“approximation-induced dead-end”这一写法

## 4. 当前属于 mixed / 仍有争议的结论

1. `threshold_only_progressive` 是否“整体优于” `non_progressive`
   - 不能这样写
   - 更准确的写法是：它在部分安全/可行性指标上更优，但 `collision_count` 和 runtime 不占优

2. H2 是否已经优于 H1
   - 不能这样写
   - `final_formal_h2_vs_h1_multiseed3x3` 与 `5x5` 呈 mixed
   - H2 常见模式是：
     - `guarantee_broken_rate` 更低
     - `dead_end_rec_rate` 更低
     - runtime 更低
     - 但 `search_rate` / `collision_count` 不稳定

3. dual scheduling 是否已经形成稳定收益
   - 不能这样写
   - 当前 `threshold_only_dual_progressive` 虽然 runtime 更低，但 `collision_count ≈ 108.49`，`guarantee_broken_rate ≈ 0.386`，`dead_end_rec_rate ≈ 0.471`
   - 相比 `threshold_only_progressive` 明显不占优

4. `safeearly_progressive` 是否说明 late-stage H2 curriculum 更好
   - 当前证据不支持
   - 反而更像一个负例：切入 H2 late stage 并没有稳定把 runtime stronger layer 转成更优 learned policy

## 5. 当前还缺什么，暂时不能下最终结论

1. 缺一个最终统一的主表重聚合脚本/表格
   - 尤其是 `formal_compare_multiseed5x5` 的 `perf_*` 指标
   - `progressive_mechanism_20260428/asset_inventory.md` 已明确指出旧 summary 与 raw 的 perf aggregate 有不一致

2. 缺一个可以直接进正文的 progressive 机制图
   - 现在已有 `stage_metrics.csv`
   - 但最好再补一张清晰可发图

3. 缺一组与正文主环境更完全同口径的 exact-diagnostic 汇总
   - 当前 exact-solver 诊断主要在小规模设置
   - 适合作为 appendix，但若能补一组主环境同口径摘要会更强

4. 缺一个最终决定：正文是否写风险变体细节
   - 代码默认 `risk_base`
   - 早期理论文本仍更接近 `clear + region + hist`
   - 当前初稿建议先弱化 risk 配方创新

5. 缺一个最终的 H2/dual 止损决定
   - 若后续没有小而强的新证据，正文应把它们稳稳降级到 mechanism / appendix

## 6. 各分支在文中的建议定位

### `threshold_only_progressive`

- 建议定位：正文主结果 / 最稳妥的主正分支
- 推荐表述：
  - “在 hard-safe 始终保留前提下，训练期 threshold curriculum 在部分安全与 future-feasibility 指标上带来相对稳定的改进”
- 不推荐表述：
  - “全面优于 non-progressive”

### `safeearly_progressive`

- 建议定位：主线对照 / 消融
- 推荐表述：
  - “用于说明 late-stage H2 升级并不自动带来更优 learned policy”
- 不推荐表述：
  - “更强版本的 progressive”

### `H=2`

- 建议定位：边界结果 / 机制证据 / appendix 候选
- 推荐表述：
  - “H2 作为 stronger runtime layer 显示出候选价值，但当前闭环训练后收益仍是 mixed”
- 关键证据：
  - validate-only `eta≈0.55`
  - `2x2` cross-eval
  - matched gate-rate / compute-budget

### dual scheduling

- 建议定位：边界结果 / future work 过渡
- 推荐表述：
  - “dual 当前更像降低介入成本的调度尝试，尚未在主安全指标上打赢 threshold-only progressive”
- 不推荐表述：
  - “完整成功的第二主创新”

## 7. 当前可以直接在稿中引用的几个数字

1. progressive 主线正式 compare
   - `non_progressive`：
     - `search_rate ≈ 0.9973`
     - `collision_count ≈ 90.79`
     - `guarantee_broken_rate ≈ 0.343`
     - `dead_end_rec_rate ≈ 0.464`
   - `threshold_only_progressive`：
     - `search_rate ≈ 0.9987`
     - `collision_count ≈ 92.57`
     - `guarantee_broken_rate ≈ 0.332`
     - `dead_end_rec_rate ≈ 0.444`
   - `safeearly_progressive`：
     - `search_rate = 1.0`
     - `collision_count ≈ 94.73`
     - `guarantee_broken_rate ≈ 0.354`
     - `dead_end_rec_rate ≈ 0.467`

2. cross-eval 关键格子
   - `h1_ckpt_h1_shield`：
     - `collision_count ≈ 86.89`
     - `guarantee_broken_rate ≈ 0.374`
   - `h1_ckpt_h2_shield`：
     - `collision_count ≈ 76.44`
     - `guarantee_broken_rate ≈ 0.316`
   - `h2_ckpt_h1_shield`：
     - `collision_count ≈ 90.67`
     - `guarantee_broken_rate ≈ 0.363`
   - `h2_ckpt_h2_shield`：
     - `search_rate ≈ 0.978`
     - `collision_count ≈ 96.11`
     - `guarantee_broken_rate ≈ 0.350`

3. H2 validate-only 候选点
   - `h2_risk_threshold_scan_medium2x2`
   - `recursive_risk_rescue_h2_eta55`：
     - `search_rate = 1.0`
     - `collision_count ≈ 72.25`
     - `guarantee_broken_rate ≈ 0.300`
     - `dead_end_rec_rate ≈ 0.133`

## 8. 当前需要特别提醒的口径冲突 / 阻塞点

1. `formal_compare_multiseed5x5/summary_metrics.csv` 的部分 `perf_*` 汇总与 raw/per-seed 重聚合不完全一致。
   - 处理建议：
     - 任务/安全指标可继续引用
     - runtime 指标优先引用 `progressive_mechanism_20260428/summary_metrics.csv` 或 raw 重聚合结果

2. 风险函数文字叙事尚未统一。
   - 理论文档更像 `v1`
   - 代码默认更像 `risk_base`
   - 处理建议：
     - 初稿主线先写 threshold curriculum / progressive conservativeness
     - 风险分量细节放到 appendix 或实验设置补充

3. H2 与 dual 没有达到“可无争议放进主表”的成熟度。
   - 处理建议：
     - 先写一个保守版初稿框架
     - 正文主线不押宝 H2 / dual
