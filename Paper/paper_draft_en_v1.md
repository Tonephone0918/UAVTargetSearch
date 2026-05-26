# English Paper Draft v1

## Title

Layered Allowed-Action Shielding for Multi-UAV Cooperative Search: Grounded `A_hard` Semantics and a Conservativeness Curriculum

## Abstract

Multi-UAV cooperative search requires a team of agents to discover targets and maintain coverage while respecting dynamic threats and inter-agent safety constraints. This paper organizes the safety module for this problem as a layered allowed-action shield. Rather than treating the shield as an external planner that takes over action selection, we define it as an allowed-action filtering module: the actor first proposes an action preference, the shield constructs an admissible action set, and the actor re-selects only within that set when the original action is not admissible.

The framework is built on grounded `A_hard` semantics and uses an exact/projected `A_hard` view as a semantic reference. We organize the safety layers as `A_hard`, `A_rec`, and `A_H^{viable}`. `A_hard` enforces one-step hard safety, `A_rec` further imposes one-step recursive feasibility, and `A_H^{viable}` represents finite-horizon future feasibility. Under this hierarchy, progressive shielding is interpreted as a conservativeness curriculum: the hard-safe layer remains always on, while training adjusts when and how strongly the stricter layers intervene.

The current evidence supports a conservative conclusion. The `threshold_only_progressive` variant is the most reliable positive result, but it should be described as a mixed but useful improvement rather than a uniform win. Compared with `non_progressive`, it improves guarantee violation and recursive dead-end rates while maintaining similar search performance, but it has higher collision count and higher runtime. The `safeearly_progressive`, H2, and dual-scheduling results are treated as ablations or boundary evidence. Together, these results suggest that stronger runtime filtering does not necessarily translate into a better learned policy.

## 1. Introduction

Multi-UAV cooperative search combines target discovery, coverage, dynamic threats, and coupled inter-agent constraints. A locally reasonable action may still be unsafe when executed jointly with other agents, and an action that is safe in the current step may push the system into a state with no safe continuation. This makes the safety problem more subtle than one-step collision avoidance.

This paper focuses on the semantics of shielding in this setting. We do not present shielding itself as a new concept. Instead, we clarify how a shield should be integrated with a learned actor in a multi-UAV search task. The shield computes an allowed action set; it does not replace the actor with a planner. The actor remains responsible for expressing preferences and re-selects within the admissible set when needed.

We use this view to build a layered allowed-action framework. The layers are `A_hard`, `A_rec`, and `A_H^{viable}`. `A_hard` provides the always-on hard-safe layer. `A_rec` removes actions that immediately lead to a lack of safe continuation. `A_H^{viable}` extends this idea to a finite look-ahead horizon. This hierarchy lets us interpret progressive shielding as a conservativeness curriculum rather than a hard-safety on/off warmup.

The key experimental finding is intentionally modest. `threshold_only_progressive` provides mixed but useful improvements: it reduces guarantee violations and recursive dead-end rates relative to `non_progressive`, while keeping search performance close, but it does not improve collision count or runtime. This is the strongest claim supported by the current evidence. The H2 and dual-scheduling results are not treated as additional success claims; they instead provide boundary evidence that stronger or more complex runtime filtering is not monotonically equivalent to better learned policies.

The paper makes four restrained contributions:

1. It formulates a layered allowed-action shielding view grounded in `A_hard` semantics for multi-UAV cooperative search.
2. It clarifies the relationship between `A_hard`, `A_rec`, and `A_H^{viable}`, while preserving the distinction between actor preference and shield filtering.
3. It uses an exact/projected `A_hard` reference to distinguish true dead-ends from approximation-induced dead-ends.
4. It analyzes progressive conservativeness curricula and presents evidence for a filtering-learning mismatch: stronger runtime filtering need not imply a better learned policy.

## 2. Related Work

### 2.1 Shielded RL and Safe Action Filtering

Safe reinforcement learning studies how to optimize policies while satisfying safety constraints. Common lines of work include constrained policy optimization, risk-sensitive objectives, control barrier functions, and runtime shielding [@garcia2015safeRLsurvey; @achiam2017cpo; @ames2017cbf; @alshiekh2018shielding]. Shielding is especially relevant when unsafe actions can be filtered before execution, because part of the safety burden is moved from reward shaping or post-hoc evaluation to the executed action interface.

Our work shares the idea of runtime filtering, but it does not claim shielding as a new mechanism. The difference is semantic and structural: the shield returns an allowed action set, while the actor still provides preferences over admissible actions. This distinction avoids framing the shield as a planner takeover.

### 2.2 Safe Multi-agent Reinforcement Learning

Safe multi-agent reinforcement learning introduces coupled constraints across agents. In multi-UAV search, safety depends not only on each UAV's local motion but also on collisions, swaps, crowding, and whether the team remains able to continue safely. This connects to safe MARL, multi-robot control, and centralized-training/decentralized-execution learning backbones [@gu2023safeMARL; @lowe2017maddpg; @yu2022mappo].

Our work does not aim to solve general decentralized multi-agent shielding. Instead, it studies a specific multi-UAV cooperative search setting and organizes its safety constraints through a grounded layered allowed-action framework.

### 2.3 Look-ahead, MPC-like Shielding, and Viability

Look-ahead safety filters, MPC-like shielding, and viability-based methods evaluate whether an action admits a safe future continuation, not only whether it is immediately safe [@aubin1991viability; @mayne2000mpc; @wabersich2021predictiveSafetyFilter]. This is closely related to `A_rec` and `A_H^{viable}` in our framework.

The difference is that our shield returns a set of admissible actions, not a single optimized action. Even when finite-horizon feasibility is used, the actor-shield division remains: the shield shrinks the action space, and the actor chooses within the remaining set.

### 2.4 Dynamic, Adaptive, and Progressive Shielding

Dynamic and adaptive shielding methods adjust the strength or frequency of intervention based on training stage, risk, or runtime state [@waga2022dynamicShielding; @xiao2023modelBasedDynamicShielding]. These works motivate the broader question of when stricter safety intervention helps learning rather than only improving runtime filtering.

Our progressive setting should not be read as a hard-safety off/on schedule. The hard-safe layer remains active throughout training. What changes is the degree to which stricter recursive or finite-horizon layers are injected. This makes the curriculum a conservativeness curriculum rather than a safety switch.

### 2.5 UAV Cooperative Search, MAPPO, and Action Masking

Multi-UAV cooperative search combines partial observability, multi-agent coordination, target discovery, and coverage control. Recent surveys of multi-UAV deep reinforcement learning emphasize scalability and cooperation challenges [@frattolillo2023multiUAVSurvey]. MAPPO-style centralized training with decentralized execution provides a common learning backbone [@yu2022mappo], and action masking is often used in discrete action spaces [@huang2022invalidActionMasking].

Our distinction from ordinary action masking is that we focus on the semantic origin and hierarchy of the allowed set. The mask is not merely a collection of local invalid-action rules; it is organized as `A_hard -> A_rec -> A_H^{viable}` and interpreted through an exact/projected `A_hard` reference.

## 3. Method

### 3.1 Problem Setting

Let `s_t` denote the global state and let each UAV `i` observe `o_t^i`. The actor produces action preferences, and the joint action is `a_t = (a_t^1, ..., a_t^n)`. The environment transition is denoted by `f(s_t, a_t)`. The hard-safe state set includes map boundaries, threat avoidance, inter-agent distance constraints, and swap constraints.

The one-step safe joint action set can be written as

\[
A^{safe}(s_t)=\{a_t \in A \mid f(s_t,a_t)\in \mathcal S_{safe}\}.
\]

One-step safety is necessary but not sufficient for sustained safe execution, because a currently safe action can still lead to a future state with no safe continuation.

### 3.2 Shielding as Allowed-Action Filtering

The shield returns an allowed action set

\[
\mathcal A_t^{allow}(s_t)\subseteq A^{safe}(s_t).
\]

The actor first proposes an action. If the action belongs to the allowed set, it is executed. Otherwise, the actor re-selects within the allowed set. Thus, the shield constrains the action space without taking over the policy.

### 3.3 Layered Allowed Sets

We organize the allowed action sets as

\[
A_{hard}(s_t), \qquad A_{rec}(s_t), \qquad A_H^{viable}(s_t).
\]

`A_hard` is the always-on one-step hard-safe layer. `A_rec` further requires that the next state admits at least one hard-safe continuation. `A_H^{viable}` extends this requirement to a finite horizon. Abstracting away approximation error, the layers satisfy

\[
A_H^{viable}(s_t)\subseteq A_{rec}(s_t)\subseteq A_{hard}(s_t).
\]

This containment relation is the basis for the conservativeness curriculum: progressive training does not turn hard safety on or off; it changes when stricter subsets are used.

### 3.4 Exact/Projected `A_hard`

For a candidate action `a_i` of agent `i`, the exact/projected hard-safe set can be written as

\[
A_{hard,i}^{\star}(s_t)
=
\{a_i \mid \exists a_{-i}, (a_i,a_{-i})\in A^{safe}(s_t)\}.
\]

This object is a semantic reference for joint feasibility projected onto an individual agent's action. The online implementation is an engineered approximation, typically based on sequential construction. Exact diagnostics and rescue are used to understand approximation errors; they are not presented as an online exact solver used at every step.

### 3.5 Dead-end Diagnosis

The exact/projected view distinguishes true dead-ends from approximation-induced dead-ends. A true dead-end occurs when the exact projected set is empty. An approximation-induced dead-end occurs when the exact projected set is non-empty, but the sequential approximation returns an empty or overly restrictive set. This distinction supports the diagnostic interpretation of false-empty and false-nonempty events.

### 3.6 Progressive Conservativeness Curriculum

The progressive curriculum changes the timing and strength of stricter filtering layers. In the main comparison, `non_progressive` uses fixed `recursive/H=1/eta=0.35`. `threshold_only_progressive` uses `safe/H=1/eta=0.90` early and switches to `recursive/H=1/eta=0.35` in the mid and late stages. `safeearly_progressive` shares the early and mid stages but switches to `recursive/H=2/eta=0.55` in the late stage.

### 3.7 Theory-Evidence Alignment

The theoretical hierarchy explains what the shield is allowed to remove, while the experiments evaluate what happens after policies are trained under different conservativeness schedules. The main table tests learned-policy outcomes. The stage-level mechanism figure tests whether the intended early/mid/late interventions actually occur. Matched gate-rate and compute-budget evidence supports the cautious statement that the threshold-only effect is not simply gate more or compute more. H2 and dual results provide boundary evidence for filtering-learning mismatch. Exact/projected `A_hard` diagnostics support the semantics of dead-end and approximation errors.

## 4. Experimental Setup

The experiments use a multi-UAV cooperative search environment with a `20 x 20` grid, `10` UAVs, `10` targets, `5` dynamic threats, and a maximum episode length of `120` steps. The UAV safety distance is `1`, and the threat safety distance is `2`.

MAPPO is used as the learning backbone. The paper does not treat MAPPO as a contribution; it is used to evaluate how different shielding curricula affect the learned policy.

The mainline experiment evaluates progressive shielding as a conservativeness curriculum. The main comparison contains three variants:

- `non_progressive`: fixed `recursive/H=1/eta=0.35`;
- `threshold_only_progressive`: early `safe/H=1/eta=0.90`, mid/late `recursive/H=1/eta=0.35`;
- `safeearly_progressive`: same early/mid stages, late `recursive/H=2/eta=0.55`.

Task, safety, and gate metrics are taken from `runs/formal_progressive_seed_compare_20260426/formal_compare_multiseed5x5/summary_metrics.csv`, aggregated over `3` training seeds, `5` evaluation seeds per checkpoint, and `5` episodes per evaluation seed. Runtime metrics are taken from `runs/progressive_mechanism_20260428/summary_metrics.csv`, using the re-aggregated mechanism summary to avoid inconsistencies in older profiling aggregation. `episode_return` is not used as a unified ranking metric across directories because reward normalization differs across historical baselines and later `risk_base` experiments.

## 5. Results and Discussion

### 5.1 Main Progressive Comparison

The main result table is reproduced from `codex1_workspace/progressive_final_main_table.csv`. The submission-ready LaTeX table is available at `Paper/tables/progressive_main_table.tex` and should be used as Table 1 (`\label{tab:progressive-main}`) in the LaTeX manuscript; the Markdown table below is kept for draft readability. Task, safety, and gate metrics use the formal comparison, while runtime metrics use the re-aggregated mechanism summary.

| model | search_rate | coverage_ratio | collision_count | guarantee_broken_rate | dead_end_rec_rate | recursive_gate_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `non_progressive` | 0.9973 | 0.9989 | 90.79 | 0.3433 | 0.4640 | 0.2684 | 197.84 | 167.40 |
| `threshold_only_progressive` | 0.9987 | 0.9979 | 92.57 | 0.3324 | 0.4437 | 0.2760 | 238.13 | 202.96 |
| `safeearly_progressive` | 1.0000 | 0.9984 | 94.73 | 0.3543 | 0.4670 | 0.2657 | 192.30 | 162.81 |

These results support a conservative conclusion: threshold-only progressive scheduling provides a mixed but useful improvement in safety and future-feasibility indicators, rather than a full dominance over the non-progressive baseline. It reduces `guarantee_broken_rate` and `dead_end_rec_rate`, while search performance remains close. However, it has higher `collision_count` and higher runtime.

`safeearly_progressive` achieves the highest `search_rate`, but it does not improve `collision_count`, `guarantee_broken_rate`, or `dead_end_rec_rate` over `threshold_only_progressive`. It is therefore better interpreted as a late-stage stronger-layer ablation, not as a stronger successful variant.

### 5.2 Stage-level Mechanism Analysis

The stage-level mechanism figure is available at `Paper/figures/progressive_stage_mechanism.png` and `Paper/figures/progressive_stage_mechanism.pdf`, with a caption draft in `Paper/figures/progressive_stage_mechanism_caption.md`. It should be used as Figure 1 (`\label{fig:progressive-stage-mechanism}`) in the LaTeX manuscript. The submission-ready stage table is available at `Paper/tables/progressive_stage_mechanism_table.tex` and can be used as Table 2 (`\label{tab:progressive-stage-mechanism}`) or moved to the appendix depending on space. These assets use aggregate evaluation rows from `runs/progressive_mechanism_20260428/stage_metrics.csv` and visualize recursive gate rate, recursive dead-end rate, and runtime.

| model | stage | shield mode | horizon | threshold | recursive_gate_rate | dead_end_rec_rate | perf_shield_time_ms | perf_recursive_time_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `non_progressive` | fixed | recursive | 1 | 0.35 | 0.2449 | 0.4549 | 175.87 | 147.49 |
| `threshold_only_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 31.41 | 0.00 |
| `threshold_only_progressive` | mid | recursive | 1 | 0.35 | 0.2477 | 0.4495 | 177.11 | 149.05 |
| `threshold_only_progressive` | late | recursive | 1 | 0.35 | 0.2473 | 0.4494 | 179.65 | 148.72 |
| `safeearly_progressive` | early | safe | 1 | 0.90 | 0.0000 | 0.0000 | 30.91 | 0.00 |
| `safeearly_progressive` | mid | recursive | 1 | 0.35 | 0.2457 | 0.4397 | 178.46 | 150.32 |
| `safeearly_progressive` | late | recursive | 2 | 0.55 | 0.0507 | 0.1441 | 97.22 | 67.18 |

The early stages stay at the hard-safe layer with no recursive gate. `threshold_only_progressive` switches to `A_rec` in the mid and late stages. `safeearly_progressive` activates an H2 stronger layer in the late stage and reduces stage-level gate and dead-end rates, but this runtime filtering pattern does not translate into a uniformly better final learned policy.

Matched gate-rate and compute-budget results further support a cautious interpretation: the threshold-only effect should not be reduced to gate more or compute more. However, the matched analysis is not a complete frontier sweep and should not be described as eliminating all confounds.

### 5.3 Boundary Results

H2 is a natural extension of `A_H^{viable}` and can act as a runtime stronger-layer candidate. Existing fixed-checkpoint, matched, and cross-evaluation results suggest that H2 can reduce recursive dead-end metrics in some settings. They do not establish H2 as a better learned-policy training regime. The appendix-ready H2 table is available at `Paper/tables/appendix_h2_boundary_table.tex` (`\label{tab:appendix-h2-boundary}`).

Dual scheduling can change runtime behavior and reduce some runtime cost, but current evidence does not show stable improvement over `threshold_only_progressive` on the main safety metrics. H2 and dual scheduling are therefore treated as boundary evidence for filtering-learning mismatch, not as additional main success claims. The appendix-ready dual table is available at `Paper/tables/appendix_dual_boundary_table.tex` (`\label{tab:appendix-dual-boundary}`).

Exact/projected `A_hard` diagnostics support the semantic foundation of the method. They help explain false-empty and false-nonempty events under sequential approximation, but they remain supporting diagnostics rather than the main experimental result. The appendix-ready exact-hard diagnostic table is available at `Paper/tables/appendix_exact_hard_diagnostic_table.tex` (`\label{tab:appendix-exact-hard}`), and the appendix drafting note is `Paper/appendix_evidence_note.md`.

## 6. Limitations

First, `threshold_only_progressive` is not a uniformly better method. It improves guarantee violation and recursive dead-end rates, but collision count and runtime are worse than `non_progressive`.

Second, `safeearly_progressive`, H2, and dual scheduling are mixed or boundary results. They should not be presented as stable success branches.

Third, runtime metrics use the re-aggregated mechanism summary, while task, safety, and gate metrics use the formal comparison summary. This source split must remain explicit.

Fourth, `episode_return` is not used as a unified cross-directory ranking metric because historical reward normalization differs.

Fifth, matched gate-rate and compute-budget analysis is not a complete frontier sweep. It supports a cautious mechanism interpretation but does not eliminate all confounds.

Sixth, exact/projected `A_hard` diagnostics are semantic and appendix-level support. They are not a large-scale proof that the online shield uses an exact solver at every step.

## 7. Conclusion

This paper organizes multi-UAV cooperative search shielding as a layered allowed-action framework grounded in `A_hard` semantics. The framework preserves the actor-shield division: the shield filters the action set, while the actor selects within the admissible set. The hierarchy `A_hard -> A_rec -> A_H^{viable}` provides a clean way to interpret progressive shielding as a conservativeness curriculum under an always-on hard-safe layer.

The current results support a restrained claim. Threshold-only progressive scheduling yields mixed but useful improvements in guarantee violation and recursive dead-end rates, while not dominating collision count or runtime. Stronger runtime filtering through H2 or more complex dual scheduling does not automatically produce a better learned policy. This boundary is central to the paper's mechanism analysis and motivates future work on more systematic conservativeness frontiers.
