#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./examples_coskill/run_webshop_coskill_2attempt_15steps.sh [ENGINE] [Hydra overrides...]

This WebShop variant reuses run_webshop_coskill.sh with these defaults:
  data.max_prompt_length                         4096
  env.skill_agent.meta_attempts.num_attempts    2
  env.max_steps                                  15

It otherwise accepts the same environment variables and Hydra overrides as
the base WebShop script.
EOF
  exit 0
fi

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export NUM_META_ATTEMPTS="${NUM_META_ATTEMPTS:-2}"
export MAX_PLAY_STEPS="${MAX_PLAY_STEPS:-15}"
export RUN_NAME="${RUN_NAME:-coskill_webshop_2attempt_15steps}"

exec "${SCRIPT_DIR}/run_webshop_coskill.sh" "$@"
