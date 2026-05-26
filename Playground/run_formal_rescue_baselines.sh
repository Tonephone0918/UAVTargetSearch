#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
DENSE_EPOCHS="${DENSE_EPOCHS:-2000}"
SPARSE_EPOCHS="${SPARSE_EPOCHS:-0}"
RUN_SEQUENTIAL="${RUN_SEQUENTIAL:-0}"

run_train() {
  local gpu_id="$1"
  local run_name="$2"
  shift 2
  mkdir -p logs
  CUDA_VISIBLE_DEVICES="$gpu_id" nohup "$PYTHON_BIN" -m src.hrvdn.main \
    --algo mappo \
    --device "$DEVICE" \
    --dense-epochs "$DENSE_EPOCHS" \
    --sparse-epochs "$SPARSE_EPOCHS" \
    --normalize-dpm-reward \
    --shield-profile-enabled \
    --shield-dead-end-policy fail_closed \
    --checkpoint-dir "checkpoints/${run_name}" \
    --tensorboard-dir "runs/${run_name}" \
    "$@" \
    > "logs/${run_name}.log" 2>&1 &
  echo "started ${run_name} on CUDA_VISIBLE_DEVICES=${gpu_id}"
}

# Rescue retrains are the default because the sequential formal baselines
# already exist and can usually be reused directly.
run_train 0 baseline_mappo_safe_rescue_normdpm_dense2000 \
  --shield-mode safe \
  --shield-hard-solver-mode sequential_with_exact_rescue

run_train 1 baseline_mappo_recursive_full_rescue_riskbase_normdpm_dense2000 \
  --shield-mode recursive \
  --shield-recursive-gate-mode full \
  --shield-hard-solver-mode sequential_with_exact_rescue \
  --shield-risk-variant risk_base

run_train 2 baseline_mappo_recursive_risk_rescue_riskbase_normdpm_dense2000 \
  --shield-mode recursive \
  --shield-recursive-gate-mode risk \
  --shield-hard-solver-mode sequential_with_exact_rescue \
  --shield-risk-variant risk_base \
  --shield-risk-threshold 0.35

if [[ "$RUN_SEQUENTIAL" == "1" ]]; then
  run_train 3 baseline_mappo_safe_seq_normdpm_dense2000 \
    --shield-mode safe \
    --shield-hard-solver-mode sequential

  run_train 0 baseline_mappo_recursive_full_seq_riskbase_normdpm_dense2000 \
    --shield-mode recursive \
    --shield-recursive-gate-mode full \
    --shield-hard-solver-mode sequential \
    --shield-risk-variant risk_base

  run_train 1 baseline_mappo_recursive_risk_seq_riskbase_normdpm_dense2000 \
    --shield-mode recursive \
    --shield-recursive-gate-mode risk \
    --shield-hard-solver-mode sequential \
    --shield-risk-variant risk_base \
    --shield-risk-threshold 0.35
fi
