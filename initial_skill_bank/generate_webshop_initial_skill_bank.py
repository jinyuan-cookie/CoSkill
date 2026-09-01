#!/usr/bin/env python3
"""Generate an initial hierarchical WebShop skill bank before PPO training.

The Reasoning model interacts with groups of repeated WebShop goals without
retrieval.  A Reflection model then converts representative trajectories from
each goal group into one task skill and grounded child step skills.  Common API
rate limiting, reflection, persistence, and output naming are shared with the
AlfWorld initializer so both datasets produce the same training JSON schema.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from initial_skill_bank.generate_alfworld_initial_skill_bank import (
    AlfWorldInitialSkillBankGenerator,
    GenerationConfig,
    OpenAICompatiblePolicy,
    RateLimitedChatClient,
    RequestsPerMinuteLimiter,
    SophonChatCompletionAdapter,
    _audit_path_for_skill_bank,
    _extract_action,
    _move_generated_file,
    _normalise_base_url,
    _skill_bank_output_path,
)


def _extract_webshop_task(observation: str) -> str:
    """Extract the shopping instruction without importing the training stack."""
    text = str(observation or "").strip()
    separator = " [SEP] "
    if separator in text and "Instruction:" in text:
        parts = text.split(separator)
        for index, part in enumerate(parts):
            if part.strip() == "Instruction:" and index + 1 < len(parts):
                return parts[index + 1].strip()
    marker = "Your task is to:"
    marker_index = text.find(marker)
    if marker_index >= 0:
        task = text[marker_index + len(marker):]
        return task.split("\n\n", 1)[0].strip()
    return text[:500]


def _available_actions(info: Dict[str, Any]) -> List[str]:
    """Convert WebShop's action dictionary to executable textual actions."""
    available = (info or {}).get("available_actions") or {}
    actions: List[str] = []
    if isinstance(available, dict):
        if available.get("has_search_bar"):
            actions.append("search[<your query>]")
        actions.extend(
            f"click[{value}]"
            for value in (available.get("clickables") or [])
            if str(value).strip()
        )
    elif isinstance(available, (list, tuple)):
        actions.extend(str(value) for value in available if str(value).strip())
    return actions or ["search[<your query>]"]


def _build_webshop_reasoning_prompt(
    task: str,
    observation: str,
    available_actions: Sequence[str],
    history: Sequence[Dict[str, str]],
    step: int,
    history_length: int,
) -> str:
    """Build a retrieval-free WebShop policy prompt for offline collection."""
    recent = list(history[-max(0, history_length):]) if history_length else []
    if recent:
        history_text = "\n".join(
            f"Observation: {item['observation']}\nAction: {item['action']}"
            for item in recent
        )
    else:
        history_text = "(none)"
    actions_text = "\n".join(f"- {action}" for action in available_actions)
    return f"""You are an expert autonomous agent operating in the WebShop environment.

Shopping goal: {task}
Current step: {step}

Recent interaction history:
{history_text}

Current observation:
{observation}

Admissible actions:
{actions_text}

Reason step by step inside <think>...</think>, then choose exactly one admissible
action inside <action>...</action>. A WebShop action must use either
search[query] or click[value]. Do not put any other text inside the action tags."""


def _task_group_uid(goal_indices: Optional[Sequence[Any]], index: int, task: str) -> str:
    """Prefer the stable WebShop session index; fall back to the exact goal."""
    if goal_indices is not None and index < len(goal_indices):
        return f"webshop-goal-{goal_indices[index]}"
    return f"webshop-goal-text::{task}"


class WebShopInitialSkillBankGenerator(AlfWorldInitialSkillBankGenerator):
    """WebShop interaction adapter for the shared hierarchical initializer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.audit["mode"] = "offline_webshop_task_group_initialization"
        self.audit["success_definition"] = "terminal task_score == 1.0"

    def _collect_rollout_batch(self) -> List[Dict[str, Any]]:
        observations, infos = self.envs.reset()
        count = len(observations)
        tasks = [_extract_webshop_task(obs) for obs in observations]
        goal_indices = getattr(self.envs, "_last_goal_indices", None)
        task_uids = [
            _task_group_uid(goal_indices, index, tasks[index])
            for index in range(count)
        ]
        histories: List[List[Dict[str, str]]] = [[] for _ in range(count)]
        turns: List[List[Dict[str, Any]]] = [[] for _ in range(count)]
        episode_rewards = [0.0] * count
        final_task_scores = [0.0] * count
        successes = [False] * count
        done = [False] * count

        for step in range(1, self.config.max_steps + 1):
            active_indices = [index for index, value in enumerate(done) if not value]
            if not active_indices:
                break
            prompts = [
                _build_webshop_reasoning_prompt(
                    task=tasks[index],
                    observation=str(observations[index]),
                    available_actions=_available_actions(infos[index]),
                    history=histories[index],
                    step=step,
                    history_length=self.config.history_length,
                )
                for index in active_indices
            ]
            responses = self.reasoning_policy.complete_batch(prompts)
            # Finished workers are still present in the fixed vector environment.
            # Their actions are ignored by collection and kept harmless.
            actions = ["search[noop]"] * count
            raw_by_index: Dict[int, str] = {}
            for index, response in zip(active_indices, responses):
                actions[index] = _extract_action(response)
                raw_by_index[index] = response

            next_observations, rewards, dones, step_infos = self.envs.step(actions)
            for index in active_indices:
                info = dict(step_infos[index] or {})
                reward = float(rewards[index])
                task_score = float(info.get("task_score", 0.0) or 0.0)
                won = bool(info.get("won", False))
                turns[index].append({
                    "observation": str(observations[index]),
                    "action": actions[index],
                    "raw_model_response": raw_by_index[index],
                    "reward": reward,
                    "task_score": task_score,
                })
                histories[index].append({
                    "observation": str(observations[index]),
                    "action": actions[index],
                })
                episode_rewards[index] += reward
                final_task_scores[index] = task_score
                successes[index] = successes[index] or won
                done[index] = bool(dones[index])
            observations = next_observations
            infos = [dict(info or {}) for info in step_infos]

        trajectories: List[Dict[str, Any]] = []
        for index in range(count):
            trajectories.append({
                "traj_uid": f"{task_uids[index]}::rollout-{index}",
                "group_uid": task_uids[index],
                "attempt_idx": 0,
                "task": tasks[index],
                "task_type": "webshop",
                "episode_reward": episode_rewards[index],
                "episode_length": len(turns[index]),
                "task_score": final_task_scores[index],
                "succeeded": successes[index],
                "full_dialogue": True,
                "refined_trajectory": {
                    "task": tasks[index],
                    "turns": [
                        {
                            "observation": turn["observation"],
                            "action": turn["action"],
                        }
                        for turn in turns[index]
                    ],
                },
            })
        return trajectories


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interact with repeated WebShop goals through an external LLM API "
            "and create an initial hierarchical Skill Bank."
        )
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        required=not bool(os.getenv("OPENAI_BASE_URL")),
    )
    parser.add_argument(
        "--api-provider",
        choices=("openai", "sophon"),
        default="openai",
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.getenv("SOPHON_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SKILL_LLM_API_KEY")
            or "EMPTY"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL"),
        required=not bool(os.getenv("OPENAI_MODEL")),
    )
    parser.add_argument("--reflection-model", default=None)
    parser.add_argument("--dataset-name", default="webshop")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-task-bundles", type=int, default=300)
    parser.add_argument("--task-groups-per-batch", type=int, default=1)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--max-step-skills-per-task", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--batches-per-pass", type=int, default=1500)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--reasoning-max-tokens", type=int, default=512)
    parser.add_argument("--reflection-max-tokens", type=int, default=2048)
    parser.add_argument("--max-concurrent-api-calls", type=int, default=8)
    parser.add_argument("--max-api-requests-per-minute", type=int, default=60)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-retries", type=int, default=10)
    parser.add_argument("--disable-api-cache", action="store_true")
    parser.add_argument("--num-cpus-per-env-worker", type=float, default=0.1)
    parser.add_argument("--num-products", type=int, default=None)
    parser.add_argument("--human-goals", action="store_true")
    parser.add_argument(
        "--use-full",
        action="store_true",
        help="Use full WebShop item/attribute files instead of the 1k small files.",
    )
    parser.add_argument("--items-path", type=Path, default=None)
    parser.add_argument("--attributes-path", type=Path, default=None)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.target_task_bundles <= 300:
        parser.error("--target-task-bundles must be between 1 and 300")
    if not 1 <= args.task_groups_per_batch <= 10:
        parser.error("--task-groups-per-batch must be between 1 and 10")
    if not 2 <= args.rollouts_per_task <= 10:
        parser.error("--rollouts-per-task must be between 2 and 10")
    if not 1 <= args.max_step_skills_per_task <= 8:
        parser.error("--max-step-skills-per-task must be between 1 and 8")
    if args.max_steps < 1 or args.batches_per_pass < 1 or args.max_passes < 1:
        parser.error("--max-steps, --batches-per-pass, and --max-passes must be positive")
    if args.max_concurrent_api_calls < 1 or args.max_api_requests_per_minute < 1:
        parser.error("API concurrency and requests-per-minute limits must be positive")
    if args.num_products is not None and args.num_products < 1:
        parser.error("--num-products must be positive when supplied")


def _webshop_data_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = (
        PROJECT_ROOT
        / "agent_system/environments/env_package/webshop/webshop/data"
    )
    items_name = "items_shuffle.json" if args.use_full else "items_shuffle_1000.json"
    attrs_name = "items_ins_v2.json" if args.use_full else "items_ins_v2_1000.json"
    items_path = (args.items_path or data_dir / items_name).expanduser().resolve()
    attributes_path = (args.attributes_path or data_dir / attrs_name).expanduser().resolve()
    return items_path, attributes_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    items_path, attributes_path = _webshop_data_paths(args)
    for label, path in (("WebShop items", items_path), ("WebShop attributes", attributes_path)):
        if not path.is_file():
            raise SystemExit(f"{label} file not found: {path}")

    auto_named_output = args.output is None
    output = (
        _skill_bank_output_path(args.dataset_name, args.model, args.target_task_bundles)
        if auto_named_output
        else args.output.expanduser().resolve()
    )
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Pass --overwrite to replace it.")
    audit_output = (
        args.audit_output.expanduser().resolve()
        if args.audit_output
        else _audit_path_for_skill_bank(output)
    )

    from agent_system.memory.skill_updater import SkillUpdater
    from agent_system.memory.skills_only_memory import SkillsOnlyMemory

    if args.api_provider == "sophon":
        base_client = SophonChatCompletionAdapter(
            endpoint=args.api_base_url,
            api_key=args.api_key,
            timeout=args.api_timeout,
            cache=not args.disable_api_cache,
        )
    else:
        from openai import OpenAI

        base_client = OpenAI(
            api_key=args.api_key,
            base_url=_normalise_base_url(args.api_base_url),
            timeout=args.api_timeout,
        )
    shared_client = RateLimitedChatClient(
        base_client,
        RequestsPerMinuteLimiter(args.max_api_requests_per_minute),
    )
    print(
        "[WebShopInitialSkillBank] API limits: "
        f"concurrent={args.max_concurrent_api_calls}, "
        f"requests_per_minute={args.max_api_requests_per_minute}"
    )

    policy = OpenAICompatiblePolicy(
        base_url=args.api_base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.reasoning_max_tokens,
        max_concurrent=args.max_concurrent_api_calls,
        timeout=args.api_timeout,
        retries=args.api_retries,
        client=shared_client,
    )
    updater = SkillUpdater(
        skill_gen_mode="task_step",
        skill_llm_api_key=args.api_key,
        skill_llm_model=args.reflection_model or args.model,
        max_completion_tokens=args.reflection_max_tokens,
        max_concurrent=args.max_concurrent_api_calls,
        success_max_step_skills=args.max_step_skills_per_task,
        chat_client=shared_client,
    )
    memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")

    from agent_system.environments.env_package.webshop import build_webshop_envs

    envs = build_webshop_envs(
        seed=args.seed,
        env_num=args.task_groups_per_batch,
        group_n=args.rollouts_per_task,
        resources_per_worker={
            "num_cpus": args.num_cpus_per_env_worker,
            "num_gpus": 0,
        },
        is_train=True,
        env_kwargs={
            "observation_mode": "text",
            "num_products": args.num_products,
            "human_goals": bool(args.human_goals),
            "file_path": str(items_path),
            "attr_path": str(attributes_path),
        },
    )
    config = GenerationConfig(
        output_path=output,
        audit_path=audit_output,
        target_task_bundles=args.target_task_bundles,
        task_groups_per_batch=args.task_groups_per_batch,
        rollouts_per_task=args.rollouts_per_task,
        max_step_skills_per_task=args.max_step_skills_per_task,
        max_steps=args.max_steps,
        history_length=args.history_length,
        batches_per_pass=args.batches_per_pass,
        max_passes=args.max_passes,
        seed=args.seed,
        max_concurrent_api_calls=args.max_concurrent_api_calls,
        max_api_requests_per_minute=args.max_api_requests_per_minute,
    )
    try:
        generator = WebShopInitialSkillBankGenerator(
            config=config,
            envs=envs,
            reasoning_policy=policy,
            skill_updater=updater,
            memory=memory,
        )
        generated_skills = generator.run()
    finally:
        envs.close()

    if auto_named_output:
        final_count = len(generated_skills.get("task_skills", []))
        final_output = _skill_bank_output_path(args.dataset_name, args.model, final_count)
        final_audit_output = (
            args.audit_output.expanduser().resolve()
            if args.audit_output
            else _audit_path_for_skill_bank(final_output)
        )
        if not args.overwrite:
            conflicts = [
                destination
                for source, destination in (
                    (output, final_output),
                    (audit_output, final_audit_output),
                )
                if source != destination and destination.exists()
            ]
            if conflicts:
                conflict_text = ", ".join(str(path) for path in conflicts)
                raise SystemExit(
                    f"Final generated output already exists: {conflict_text}. "
                    f"Pass --overwrite to replace it."
                )
        output = _move_generated_file(output, final_output, overwrite=args.overwrite)
        if audit_output != final_audit_output:
            audit_output = _move_generated_file(
                audit_output,
                final_audit_output,
                overwrite=args.overwrite,
            )
    print(f"[WebShopInitialSkillBank] Skill bank: {output}")
    print(f"[WebShopInitialSkillBank] Audit: {audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
