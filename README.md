# HRVDN UAV Target Search Reproduction

A runnable reproduction-oriented codebase for multi-UAV cooperative target search with:

- 2D grid map and multi-altitude UAVs.
- Three cognitive maps (STM/TPM/DPM) + N+/N- counters.
- Three-stage TPM update (detection, communication fusion, revisit compensation).
- Shared-reward VDN with per-agent recurrent Q-network (MLP -> GRU -> MLP).
- MAPPO baseline with shared actor and centralized critic.
- Dense / sparse / hybrid reward training, including dense-to-sparse phase switching, target reset, and replay reward recalculation.
- Evaluation metrics and ablation entrypoints.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m hrvdn.main --dense-epochs 2 --sparse-epochs 2
```

The reproduction code now uses a paper-aligned fixed-dimensional observation:
local maps + current altitude + previous action, without UAV-count one-hot IDs.
Because of this change, old checkpoints created by the previous code version are
not compatible and should be retrained.

Paper-like environment settings can be changed directly from the command line:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --map-size 20 \
  --n-uavs 10 \
  --n-targets 10 \
  --n-threats 5 \
  --dense-epochs 600 \
  --sparse-epochs 2800
```

Train MAPPO:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --algo mappo \
  --map-size 20 \
  --n-uavs 10 \
  --n-targets 10 \
  --n-threats 5 \
  --dense-epochs 600 \
  --sparse-epochs 2800
```

MAPPO checkpoints and TensorBoard logs default to:

- `checkpoints/mappo`
- `runs/mappo`

Start TensorBoard:

```bash
tensorboard --logdir runs/mappo
```

The current MAPPO baseline uses a shared actor plus centralized critic.
Evaluation and replay can override `--n-uavs` without rebuilding the network,
so it is suitable as a controllable generalization baseline for later
algorithm modifications.

## Validation + Visualization

1) Validate a trained checkpoint:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --skip-train \
  --validate-checkpoint checkpoints/best.pt \
  --n-uavs 15 \
  --eval-episodes 20
```

Validate a MAPPO checkpoint with a different number of UAVs:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --skip-train \
  --validate-checkpoint checkpoints/mappo/best.pt \
  --n-uavs 15 \
  --eval-episodes 20
```

2) Generate an HTML training report from TensorBoard logs:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --skip-train \
  --report \
  --report-logdir runs/hrvdn \
  --report-output runs/hrvdn/report.html
```

3) Train + validate + generate report in one command:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --dense-epochs 600 \
  --sparse-epochs 2800 \
  --validate-checkpoint checkpoints/best.pt \
  --eval-episodes 20 \
  --report
```

## Dynamic Search Replay

Generate a dynamic, step-by-step HTML replay of the whole search process:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --skip-train \
  --animate-checkpoint checkpoints/best.pt \
  --n-uavs 15 \
  --animate-output runs/hrvdn/search_replay.html
```

Animate a MAPPO checkpoint:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --skip-train \
  --animate-checkpoint checkpoints/mappo/best.pt \
  --n-uavs 15 \
  --animate-output runs/mappo/search_replay.html
```

Optional: limit replay steps

```bash
PYTHONPATH=src python -m hrvdn.main \
  --skip-train \
  --animate-checkpoint checkpoints/best.pt \
  --animate-output runs/hrvdn/search_replay.html \
  --animate-max-steps 80
```

Generate a replay directly from a simple built-in greedy policy without training:

```bash
PYTHONPATH=src python -m hrvdn.main \
  --skip-train \
  --animate-baseline greedy \
  --animate-output runs/hrvdn/greedy_search_replay.html
```

## Ablations

```bash
PYTHONPATH=src python -m hrvdn.main --ablation reward
PYTHONPATH=src python -m hrvdn.main --ablation height
PYTHONPATH=src python -m hrvdn.main --ablation compensation
PYTHONPATH=src python -m hrvdn.main --ablation energy
```

## MAPPO训练
CUDA_VISIBLE_DEVICES=3 nohup PYTHONPATH=src .venv/bin/python -m hrvdn.main   --algo hrvdn   --normalize-dpm-reward   --dense-epochs 600   --sparse-epochs 2800   --checkpoint-dir checkpoints/hrvdn_normdpm   --tensorboard-dir runs/hrvdn_normdpm  --n-uavs 5 > hrvdn.log 2>&1 &

Formal Baselines：
```python
CUDA_VISIBLE_DEVICES=0 nohup ./.venv/bin/python -m src.hrvdn.main \
  --algo mappo \
  --device cuda \
  --dense-epochs 2000 \
  --sparse-epochs 0 \
  --normalize-dpm-reward \
  --shield-mode off \
  --shield-profile-enabled \
  --checkpoint-dir checkpoints/baseline_mappo_off_normdpm_dense2000 \
  --tensorboard-dir runs/baseline_mappo_off_normdpm_dense2000 \
  > logs/baseline_mappo_off_normdpm_dense2000.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup ./.venv/bin/python -m src.hrvdn.main \
  --algo mappo \
  --device cuda \
  --dense-epochs 2000 \
  --sparse-epochs 0 \
  --normalize-dpm-reward \
  --shield-mode safe \
  --shield-profile-enabled \
  --checkpoint-dir checkpoints/baseline_mappo_safe_normdpm_dense2000 \
  --tensorboard-dir runs/baseline_mappo_safe_normdpm_dense2000 \
  > logs/baseline_mappo_safe_normdpm_dense2000.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup ./.venv/bin/python -m src.hrvdn.main \
  --algo mappo \
  --device cuda \
  --dense-epochs 2000 \
  --sparse-epochs 0 \
  --normalize-dpm-reward \
  --shield-mode recursive \
  --shield-profile-enabled \
  --shield-risk-score-enabled \
  --shield-risk-variant risk_base \
  --shield-risk-threshold -1.0 \
  --no-shield-legacy-recursive-gate \
  --checkpoint-dir checkpoints/baseline_mappo_recursive_full_riskbase_normdpm_dense2000 \
  --tensorboard-dir runs/baseline_mappo_recursive_full_riskbase_normdpm_dense2000 \
  > logs/baseline_mappo_recursive_full_riskbase_normdpm_dense2000.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup ./.venv/bin/python -m src.hrvdn.main \
  --algo mappo \
  --device cuda \
  --dense-epochs 2000 \
  --sparse-epochs 0 \
  --normalize-dpm-reward \
  --shield-mode recursive \
  --shield-profile-enabled \
  --shield-risk-score-enabled \
  --shield-risk-variant risk_base \
  --shield-risk-threshold 0.35 \
  --no-shield-legacy-recursive-gate \
  --checkpoint-dir checkpoints/baseline_mappo_recursive_risk_riskbase_normdpm_dense2000 \
  --tensorboard-dir runs/baseline_mappo_recursive_risk_riskbase_normdpm_dense2000 \
  > logs/baseline_mappo_recursive_risk_riskbase_normdpm_dense2000.log 2>&1 &
```

Risk Sweep：
```python
recursive(risk) with eta = 0.25
CUDA_VISIBLE_DEVICES=0 nohup ./.venv/bin/python -m src.hrvdn.main \
  --algo mappo \
  --device cuda \
  --dense-epochs 2000 \
  --sparse-epochs 0 \
  --normalize-dpm-reward \
  --shield-mode recursive \
  --shield-profile-enabled \
  --shield-risk-score-enabled \
  --shield-risk-variant risk_base \
  --shield-risk-threshold 0.25 \
  --no-shield-legacy-recursive-gate \
  --checkpoint-dir checkpoints/baseline_mappo_recursive_risk_eta025_riskbase_normdpm_dense2000 \
  --tensorboard-dir runs/baseline_mappo_recursive_risk_eta025_riskbase_normdpm_dense2000 \
  > logs/baseline_mappo_recursive_risk_eta025_riskbase_normdpm_dense2000.log 2>&1 &


recursive(risk) with eta = 0.35
CUDA_VISIBLE_DEVICES=1 nohup ./.venv/bin/python -m src.hrvdn.main \
  --algo mappo \
  --device cuda \
  --dense-epochs 2000 \
  --sparse-epochs 0 \
  --normalize-dpm-reward \
  --shield-mode recursive \
  --shield-profile-enabled \
  --shield-risk-score-enabled \
  --shield-risk-variant risk_base \
  --shield-risk-threshold 0.35 \
  --no-shield-legacy-recursive-gate \
  --checkpoint-dir checkpoints/baseline_mappo_recursive_risk_eta035_riskbase_normdpm_dense2000 \
  --tensorboard-dir runs/baseline_mappo_recursive_risk_eta035_riskbase_normdpm_dense2000 \
  > logs/baseline_mappo_recursive_risk_eta035_riskbase_normdpm_dense2000.log 2>&1 &


recursive(risk) with eta = 0.45
CUDA_VISIBLE_DEVICES=2 nohup ./.venv/bin/python -m src.hrvdn.main \
  --algo mappo \
  --device cuda \
  --dense-epochs 2000 \
  --sparse-epochs 0 \
  --normalize-dpm-reward \
  --shield-mode recursive \
  --shield-profile-enabled \
  --shield-risk-score-enabled \
  --shield-risk-variant risk_base \
  --shield-risk-threshold 0.45 \
  --no-shield-legacy-recursive-gate \
  --checkpoint-dir checkpoints/baseline_mappo_recursive_risk_eta045_riskbase_normdpm_dense2000 \
  --tensorboard-dir runs/baseline_mappo_recursive_risk_eta045_riskbase_normdpm_dense2000 \
  > logs/baseline_mappo_recursive_risk_eta045_riskbase_normdpm_dense2000.log 2>&1 &


简单的阈值扫描批量版:
etas=(0.25 0.35 0.45)
gpus=(0 1 2)

for i in "${!etas[@]}"; do
  eta="${etas[$i]}"
  gpu="${gpus[$i]}"
  tag="${eta/./}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup ./.venv/bin/python -m src.hrvdn.main \
    --algo mappo \
    --device cuda \
    --dense-epochs 2000 \
    --sparse-epochs 0 \
    --normalize-dpm-reward \
    --shield-mode recursive \
    --shield-profile-enabled \
    --shield-risk-score-enabled \
    --shield-risk-variant risk_base \
    --shield-risk-threshold "$eta" \
    --no-shield-legacy-recursive-gate \
    --checkpoint-dir "checkpoints/baseline_mappo_recursive_risk_eta${tag}_riskbase_normdpm_dense2000" \
    --tensorboard-dir "runs/baseline_mappo_recursive_risk_eta${tag}_riskbase_normdpm_dense2000" \
    > "logs/baseline_mappo_recursive_risk_eta${tag}_riskbase_normdpm_dense2000.log" 2>&1 &
done
```
