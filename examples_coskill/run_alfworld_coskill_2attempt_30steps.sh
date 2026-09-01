#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./examples_coskill/run_alfworld_coskill_2attempt_30steps.sh [ENGINE] [Hydra overrides...]

Reasoning-only RL ablation:
  SKILL_AGENT_RL_TRAINING_ENABLED=false \
  RUN_NAME=coskill_alfworld_2attempt_30steps_no_skill_rl \
    ./examples_coskill/run_alfworld_coskill_2attempt_30steps.sh vllm

This variant reuses run_alfworld_coskill.sh with these defaults:
  data.max_prompt_length                         8192
  env.skill_agent.meta_attempts.num_attempts    2
  env.max_steps                                  30

It otherwise accepts the same environment variables and Hydra overrides as
the base script.
EOF
  exit 0
fi

# Faster two-attempt variant of run_alfworld_coskill.sh. All other settings
# continue to come from the base script and may be overridden in the same way.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export NUM_META_ATTEMPTS="${NUM_META_ATTEMPTS:-2}"
export MAX_PLAY_STEPS="${MAX_PLAY_STEPS:-30}"
export RUN_NAME="${RUN_NAME:-coskill_alfworldv2_2attempt_30steps}"

exec "${SCRIPT_DIR}/run_alfworld_coskill.sh" "$@"
