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
  --checkpoint-dir checkpoints/mappo_run1 \
  --tensorboard-dir runs/mappo_run1 \
  --normalize-dpm-reward \
  --dense-epochs 600 \
  --sparse-epochs 2800
```

MAPPO checkpoints and TensorBoard logs default to:

- `checkpoints/mappo`
- `runs/mappo`

You can override both paths from the command line with
`--checkpoint-dir` and `--tensorboard-dir`.
If dense training becomes dominated by the pheromone term, you can enable
`--normalize-dpm-reward` to scale the DPM reward by swarm size and map area.

Start TensorBoard:

```bash
tensorboard --logdir runs/mappo
```

The current MAPPO baseline uses a shared actor plus centralized critic.
Evaluation and replay can override `--n-uavs` without rebuilding the network,
so it is suitable as a controllable generalization baseline for later
algorithm modifications.

Episodes now default to a fixed horizon and end at `--max-steps`, which is
closer to the paper's mission-period formulation. If you want the old
behavior, you can pass `--terminate-on-all-found`.

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
