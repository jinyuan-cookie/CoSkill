#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/env.sh"
fi

SOPHON_API_URL="${SOPHON_API_URL:-}"
SOPHON_MODEL="${SOPHON_MODEL:-}"
MAX_API_REQUESTS_PER_MINUTE="${MAX_API_REQUESTS_PER_MINUTE:-60}"
WEBSHOP_USE_FULL="${WEBSHOP_USE_FULL:-0}"

for variable_name in SOPHON_API_URL SOPHON_MODEL; do
  if [[ -z "${!variable_name}" ]]; then
    echo "Required API setting is not configured: ${variable_name}" >&2
    exit 1
  fi
done

if [[ -z "${SOPHON_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "SOPHON_API_KEY is not set and no interactive terminal is available." >&2
    exit 1
  fi
  read -r -s -p "Enter SOPHON_API_KEY: " SOPHON_API_KEY
  echo
fi

if [[ -z "${SOPHON_API_KEY}" ]]; then
  echo "SOPHON_API_KEY cannot be empty." >&2
  exit 1
fi
export SOPHON_API_KEY

case "${WEBSHOP_USE_FULL}" in
  0|1)
    ;;
  *)
    echo "WEBSHOP_USE_FULL must be 0 or 1; got: ${WEBSHOP_USE_FULL}" >&2
    exit 1
    ;;
esac

ARGS=(
  --api-provider sophon
  --api-base-url "${SOPHON_API_URL}"
  --model "${SOPHON_MODEL}"
  --temperature 1
  --task-groups-per-batch 1
  --rollouts-per-task 8
  --max-steps 15
  --max-concurrent-api-calls 8
  --max-api-requests-per-minute "${MAX_API_REQUESTS_PER_MINUTE}"
)

if [[ "${WEBSHOP_USE_FULL}" == "1" ]]; then
  ARGS+=(--use-full)
fi
if [[ -n "${WEBSHOP_ITEMS_PATH:-}" ]]; then
  ARGS+=(--items-path "${WEBSHOP_ITEMS_PATH}")
fi
if [[ -n "${WEBSHOP_ATTRIBUTES_PATH:-}" ]]; then
  ARGS+=(--attributes-path "${WEBSHOP_ATTRIBUTES_PATH}")
fi

cd "${PROJECT_DIR}"

python3 initial_skill_bank/generate_webshop_initial_skill_bank.py \
  "${ARGS[@]}" \
  "$@"
