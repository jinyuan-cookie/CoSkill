#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./examples_coskill/run_alfworld_coskill.sh [ENGINE] [Hydra overrides...]

Examples:
  ./examples_coskill/run_alfworld_coskill.sh vllm

  INITIAL_SKILL_BANK_PATH=/path/to/alfworld_model_300.json \
    ./examples_coskill/run_alfworld_coskill.sh vllm \
    trainer.total_epochs=10

Required environment variables (set them in env.sh or export them):
  MODEL_PATH                 Actor model directory
  ALFWORLD_DATA              ALFWorld data directory
  TRAIN_DATA_PATH            Training parquet
  VAL_DATA_PATH              Validation parquet
  SKILL_EMBEDDING_MODEL_PATH Embedding model directory

Defaults:
  INITIAL_SKILL_BANK_PATH
                   <project>/initial_skill_bank/skill_bank/alfworld_gpt-5.5-2026-04-24_300.json
  SKILL_RETRIEVAL_SERVICE_URL
                   http://127.0.0.1:8003/retrieve_batch
  MAX_PROMPT_LENGTH 8192
  MAX_PLAY_STEPS    50
  NUM_META_ATTEMPTS 3
  SKILL_AGENT_RL_TRAINING_ENABLED
                   true

Reasoning Agent uses verl-agent GiGPO with gamma=0.95, step weight=1.0,
mean/std normalization, and exact observation grouping.
Attempt 0 retrieves global skills through SKILL_RETRIEVAL_SERVICE_URL.
Attempts 1/2 retrieve only from their local, edited skill overlays.
Set SKILL_AGENT_RL_TRAINING_ENABLED=false to keep Skill Agent editing and
meta-attempt evaluation while excluding Skill Agent samples from PPO updates.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Keep private tokens and machine-specific settings in env.sh when needed.
if [[ -f "${PROJECT_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/env.sh"
fi

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

ALFWORLD_DATA="${ALFWORLD_DATA:-}"
MODEL_PATH="${MODEL_PATH:-}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
SKILL_EMBEDDING_MODEL_PATH="${SKILL_EMBEDDING_MODEL_PATH:-}"

for variable_name in MODEL_PATH ALFWORLD_DATA TRAIN_DATA_PATH VAL_DATA_PATH SKILL_EMBEDDING_MODEL_PATH; do
  if [[ -z "${!variable_name}" ]]; then
    echo "Required environment variable is not set: ${variable_name}" >&2
    exit 1
  fi
done

export ALFWORLD_DATA MODEL_PATH
export PYTHONPATH="${PROJECT_DIR}/agent_system/environments/env_package/alfworld:${PROJECT_DIR}:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"

INITIAL_SKILL_BANK_PATH="${INITIAL_SKILL_BANK_PATH:-${PROJECT_DIR}/initial_skill_bank/skill_bank/alfworld_gpt-5.5-2026-04-24_300.json}"
SKILL_RETRIEVAL_SERVICE_URL="${SKILL_RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8003/retrieve_batch}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_PLAY_STEPS="${MAX_PLAY_STEPS:-50}"
NUM_META_ATTEMPTS="${NUM_META_ATTEMPTS:-3}"
SKILL_AGENT_RL_TRAINING_ENABLED="${SKILL_AGENT_RL_TRAINING_ENABLED:-true}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-80}"

case "${SKILL_AGENT_RL_TRAINING_ENABLED}" in
  true|false)
    ;;
  *)
    echo "SKILL_AGENT_RL_TRAINING_ENABLED must be true or false; got: ${SKILL_AGENT_RL_TRAINING_ENABLED}" >&2
    exit 1
    ;;
esac

RUN_NAME="${RUN_NAME:-coskill_alfworld}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
  echo "Training parquet not found: ${TRAIN_DATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${VAL_DATA_PATH}" ]]; then
  echo "Validation parquet not found: ${VAL_DATA_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Actor model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ALFWORLD_DATA}" ]]; then
  echo "ALFWorld data directory not found: ${ALFWORLD_DATA}" >&2
  exit 1
fi
if [[ ! -f "${INITIAL_SKILL_BANK_PATH}" ]]; then
  echo "Initial Skill Bank not found: ${INITIAL_SKILL_BANK_PATH}" >&2
  echo "Generate it first or set INITIAL_SKILL_BANK_PATH explicitly." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/checkpoints"

echo "Project:            ${PROJECT_DIR}"
echo "Actor model:        ${MODEL_PATH}"
echo "ALFWorld data:      ${ALFWORLD_DATA}"
echo "Training parquet:   ${TRAIN_DATA_PATH}"
echo "Validation parquet: ${VAL_DATA_PATH}"
echo "Initial Skill Bank: ${INITIAL_SKILL_BANK_PATH}"
echo "Retrieval service:  ${SKILL_RETRIEVAL_SERVICE_URL}"
echo "Embedding model:    ${SKILL_EMBEDDING_MODEL_PATH}"
echo "Max prompt length:   ${MAX_PROMPT_LENGTH}"
echo "Meta attempts:       ${NUM_META_ATTEMPTS}"
echo "Max play steps:      ${MAX_PLAY_STEPS}"
echo "Train Skill Agent:   ${SKILL_AGENT_RL_TRAINING_ENABLED}"
echo "Reasoning algorithm: GiGPO (gamma=0.95, mode=mean_std_norm)"
echo "Training log:       ${LOG_FILE}"

cd "${PROJECT_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    algorithm.gigpo.enable_similarity=false \
    data.train_files="${TRAIN_DATA_PATH}" \
    data.val_files="${VAL_DATA_PATH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=1.0 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps="${MAX_PLAY_STEPS}" \
    env.rollout.n="${GROUP_SIZE}" \
    env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path="${INITIAL_SKILL_BANK_PATH}" \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.skill_retrieval_service_url="${SKILL_RETRIEVAL_SERVICE_URL}" \
    +env.skills_only_memory.embedding_model_path="${SKILL_EMBEDDING_MODEL_PATH}" \
    +env.skills_only_memory.skill_text_for_retrieval=when_to_apply \
    +env.skills_only_memory.load_initial_skills=True \
    +env.skills_only_memory.similarity_threshold=null \
    +env.skills_only_memory.top_k_task=1 \
    +env.skills_only_memory.top_k_step=1 \
    +env.skills_only_memory.skill_gen_mode=task_step \
    +env.skill_agent.enabled=true \
    +env.skill_agent.max_history_steps=2 \
    +env.skill_agent.record_trajectories=true \
    +env.skill_agent.training.enabled="${SKILL_AGENT_RL_TRAINING_ENABLED}" \
    +env.skill_agent.training.adv_estimator=gigpo \
    +env.skill_agent.training.gamma=0.95 \
    +env.skill_agent.training.step_advantage_w=1.0 \
    +env.skill_agent.meta_attempts.enabled=true \
    +env.skill_agent.meta_attempts.num_attempts="${NUM_META_ATTEMPTS}" \
    +env.skill_agent.meta_attempts.promote_to_global_bank=true \
    +env.skill_agent.meta_attempts.max_promotions_per_task_group=2 \
    +env.skill_agent.meta_attempts.require_effective_edit=true \
    +env.skills_only_memory.max_concurrent=10 \
    +env.skills_only_memory.enable_dynamic_update=False \
    +env.skills_only_memory.update_save_traj=True \
    +env.skills_only_memory.update_source=train \
    +env.skills_only_memory.skill_update_group_success_rate_threshold=0.5 \
    +env.skills_only_memory.max_trajectories_for_skill_update=10 \
    +env.skills_only_memory.record_retrieved_skills=True \
    +env.skills_only_memory.enable_dynamic_management=True \
    +env.skills_only_memory.management.baseline_ab_split=false \
    +env.skills_only_memory.management.utility_ema_beta=0.5 \
    +env.skills_only_memory.management.utility_ema_beta_task=0.5 \
    +env.skills_only_memory.management.utility_ema_beta_step=0.5 \
    +env.skills_only_memory.management.retrieval_top_2k=null \
    +env.skills_only_memory.management.retrieval_alpha=null \
    +env.skills_only_memory.management.retrieval_ucb_c=0.0 \
    +env.skills_only_memory.management.intrinsic_reward_enabled=false \
    +env.skills_only_memory.management.intrinsic_reward_coefficient=1 \
    +env.skills_only_memory.management.credit_use_baseline=true \
    +env.skills_only_memory.management.eviction_enabled=true \
    +env.skills_only_memory.management.eviction_interval=5 \
    +env.skills_only_memory.management.eviction_max_task_bundles=300 \
    +env.skills_only_memory.management.eviction_protect_recent_steps=10 \
    +env.skills_only_memory.management.eviction_policy=frequency_recency \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=coskill_alfworld \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.log_val_generations=10 \
    trainer.save_freq=200 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.val_before_train=True \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.ray_wait_register_center_timeout=3600 \
    ray_init.num_cpus="${RAY_NUM_CPUS}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
