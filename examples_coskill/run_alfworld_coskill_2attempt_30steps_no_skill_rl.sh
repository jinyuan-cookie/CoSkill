#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./examples_coskill/run_alfworld_coskill_2attempt_30steps_no_skill_rl.sh [ENGINE] [Hydra overrides...]

Reasoning-only RL ablation based on the two-attempt, 30-step configuration:
  - Reasoning Agent samples update the shared actor.
  - Skill Agent still edits skills during Attempt 0.
  - Attempt 1 still evaluates the private edited overlay.
  - Skill Agent samples are excluded from the PPO actor update.

Environment variables and Hydra overrides are forwarded to the standard
two-attempt script.
EOF
  exit 0
fi

export SKILL_AGENT_RL_TRAINING_ENABLED=false
export RUN_NAME="${RUN_NAME:-coskill_alfworld_qwen2_5_7b_2attempt_30steps_no_skill_rl}"

exec "${SCRIPT_DIR}/run_alfworld_coskill_2attempt_30steps.sh" "$@"
