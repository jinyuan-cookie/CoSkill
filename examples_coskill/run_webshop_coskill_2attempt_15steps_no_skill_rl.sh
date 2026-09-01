#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./examples_coskill/run_webshop_coskill_2attempt_15steps_no_skill_rl.sh [ENGINE] [Hydra overrides...]

Reasoning-only RL ablation for WebShop:
  - Reasoning Agent samples update the shared actor.
  - Skill Agent still proposes edits during Attempt 0.
  - Attempt 1 still evaluates the private edited overlay.
  - Skill Agent samples are excluded from the PPO actor update.
EOF
  exit 0
fi

export SKILL_AGENT_RL_TRAINING_ENABLED=false
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_PLAY_STEPS="${MAX_PLAY_STEPS:-15}"
export RUN_NAME="${RUN_NAME:-coskill_webshop_2attempt_15steps_no_skill_rl}"

exec "${SCRIPT_DIR}/run_webshop_coskill_2attempt_15steps.sh" "$@"
