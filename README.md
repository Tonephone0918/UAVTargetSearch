# HRVDN UAV Target Search Reproduction

A runnable reproduction-oriented codebase for multi-UAV cooperative target search with:

- 2D grid map and multi-altitude UAVs.
- Three cognitive maps (STM/TPM/DPM) + N+/N- counters.
- Three-stage TPM update (detection, communication fusion, revisit compensation).
- Shared-reward VDN with per-agent recurrent Q-network (MLP -> GRU -> MLP).
- Dense / sparse / hybrid reward training, including dense-to-sparse phase switching, target reset, and replay reward recalculation.
- Evaluation metrics and ablation entrypoints.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
PYTHONPATH=src python -m hrvdn.main --dense-epochs 2 --sparse-epochs 2
```

## Ablations

```bash
PYTHONPATH=src python -m hrvdn.main --ablation reward
PYTHONPATH=src python -m hrvdn.main --ablation height
PYTHONPATH=src python -m hrvdn.main --ablation compensation
PYTHONPATH=src python -m hrvdn.main --ablation energy
```
