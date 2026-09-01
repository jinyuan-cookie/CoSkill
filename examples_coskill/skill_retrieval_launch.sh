#!/usr/bin/env bash
# Launch skill retrieval server (embedding mode), 8-GPU by default.
# Run:  bash examples_coskill/skill_retrieval_launch.sh
# Then in training config set: env.skills_only_memory.skill_retrieval_service_url=http://127.0.0.1:8003/retrieve_batch

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/env.sh"
fi

# Use CUDA_VISIBLE_DEVICES and NUM_GPUS to match the available hardware.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
SKILLS_JSON="${SKILLS_JSON:-}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-${SKILL_EMBEDDING_MODEL_PATH:-}}"
PORT="${PORT:-8003}"

if [[ -z "${EMBEDDING_MODEL}" ]]; then
  echo "Set EMBEDDING_MODEL or SKILL_EMBEDDING_MODEL_PATH before starting the retrieval server." >&2
  exit 1
fi

# The trainer loads the current Skill Bank through /reload_skills by default.
LOAD_INITIAL_SKILLS="${LOAD_INITIAL_SKILLS:-0}"

ARGS=(
  --device cuda
  --port "$PORT"
  --num_gpus "$NUM_GPUS"
  --embedding_model_path "$EMBEDDING_MODEL"
)
if [[ "$LOAD_INITIAL_SKILLS" == "0" ]]; then
  ARGS+=(--no_load_initial_skills)
else
  if [[ -z "${SKILLS_JSON}" ]]; then
    echo "Set SKILLS_JSON when LOAD_INITIAL_SKILLS=1." >&2
    exit 1
  fi
  ARGS+=(--skills_json_path "$SKILLS_JSON")
fi

python examples_coskill/skill_retrieval_server.py "${ARGS[@]}" "$@"
