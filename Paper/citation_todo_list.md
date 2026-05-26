# Citation TODO List

Updated: `2026-05-20`

This file records citation gaps that should be checked before submission. It is intentionally conservative: uncertain references should not be promoted into the paper until author, title, venue, year, and the relevant claim are verified.

## Already Seeded References

Seed bibliography:

- `Paper/references_seed.bib`

Currently covered topics:

- Safe RL survey and constrained policy optimization: `garcia2015safeRLsurvey`, `achiam2017cpo`
- Runtime shielding and safe action filtering: `alshiekh2018shielding`
- Control barrier functions: `ames2017cbf`
- Viability, MPC, and predictive safety filters: `aubin1991viability`, `mayne2000mpc`, `wabersich2021predictiveSafetyFilter`
- Safe MARL / multi-robot control: `gu2023safeMARL`
- CTDE and MAPPO-style backbones: `lowe2017maddpg`, `yu2022mappo`
- Discrete invalid action masking: `huang2022invalidActionMasking`
- Dynamic or model-based shielding: `waga2022dynamicShielding`, `xiao2023modelBasedDynamicShielding`
- Multi-UAV DRL survey: `frattolillo2023multiUAVSurvey`

## Remaining Citation Gaps

1. UAV cooperative target search with dynamic threats

Recommended search keywords:

- `"multi-UAV cooperative target search" "reinforcement learning"`
- `"multi-UAV search" "dynamic threats" "reinforcement learning"`
- `"multi-agent reinforcement learning" "UAV" "target search" "coverage"`

Needed claim:

- Multi-UAV cooperative search commonly combines partial observability, target discovery, coverage, dynamic obstacles or threats, and inter-agent coordination.

2. UAV or multi-robot shielding with coupled collision/swap constraints

Recommended search keywords:

- `"multi-agent shielding" "collision avoidance" reinforcement learning`
- `"multi-robot" "shielding" "reinforcement learning" safety`
- `"safe multi-agent reinforcement learning" "runtime shielding"`

Needed claim:

- Multi-agent shielding must handle coupled constraints and joint-action feasibility, not only local invalid actions.

3. Curriculum or progressive safety constraints

Recommended search keywords:

- `"curriculum" "safe reinforcement learning" "constraints"`
- `"adaptive shielding" "training" "reinforcement learning"`
- `"progressive safety constraints" "reinforcement learning"`

Needed claim:

- Safety intervention can be scheduled or adapted over training, but the present paper specifically frames this as a conservativeness curriculum under an always-on hard-safe layer.

4. Action masking in multi-agent discrete control

Recommended search keywords:

- `"invalid action masking" "multi-agent reinforcement learning"`
- `"action masking" "MAPPO" "multi-agent"`
- `"action mask" "discrete action" "policy gradient"`

Needed claim:

- Action masks are common in discrete RL, but this paper differs by assigning layered safety semantics to the mask.

## Do Not Cite Without Verification

- Do not add broad UAV survey claims unless the cited paper explicitly discusses cooperative multi-UAV learning or search/coverage.
- Do not cite dynamic shielding papers as evidence that the present progressive curriculum is already established in the exact same form.
- Do not cite H2, dual scheduling, or exact/projected `A_hard` as external prior work; those are internal boundary or theory-support materials in this project.
