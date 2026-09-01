#!/usr/bin/env python3
"""Generate an initial hierarchical AlfWorld skill bank before PPO training.

This program is intentionally independent of ``RayPPOTrainer``.  It uses one
OpenAI-compatible API for both roles required during initialization:

1. sample Reasoning-Agent actions while interacting with AlfWorld;
2. sample representative success/failure evidence for each task and produce one
   task skill with grounded child step skills.

The resulting JSON can be loaded directly with
``env.skills_only_memory.skills_json_path`` during normal training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence


# Allow ``python initial_skill_bank/generate_...py`` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _normalise_base_url(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def _filename_component(value: str, fallback: str) -> str:
    """Convert dataset/model names into portable filename components."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    component = re.sub(r"-+", "-", component).strip("-._")
    return component or fallback


def _skill_bank_output_path(dataset: str, model: str, count: int) -> Path:
    """Return the default path: dataset + model + actual task-bundle count."""
    dataset_name = _filename_component(dataset, "dataset")
    model_name = _filename_component(model, "model")
    return PROJECT_ROOT / "initial_skill_bank/skill_bank" / f"{dataset_name}_{model_name}_{int(count)}.json"


def _audit_path_for_skill_bank(skill_path: Path) -> Path:
    return skill_path.with_name(f"{skill_path.stem}.audit.json")


def _move_generated_file(source: Path, destination: Path, overwrite: bool) -> Path:
    """Move a generated artifact without silently replacing an existing run."""
    if source == destination:
        return destination
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Generated output already exists: {destination}. Pass --overwrite to replace it. "
            f"The new artifact remains at {source}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    return destination


class RequestsPerMinuteLimiter:
    """Thread-safe sliding-window limiter shared by every external LLM call."""

    def __init__(
        self,
        max_requests: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if int(max_requests) < 1:
            raise ValueError("max_requests must be positive")
        if float(window_seconds) <= 0:
            raise ValueError("window_seconds must be positive")
        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._sleep = sleep
        self._request_starts = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        """Wait until one request-start slot is available."""
        # Sleeping while holding this short-lived lock deliberately elects one
        # waiter to advance the shared window. Network calls happen after the
        # lock is released, so up to max_concurrent requests can remain in flight.
        with self._lock:
            while True:
                now = self._clock()
                cutoff = now - self.window_seconds
                while self._request_starts and self._request_starts[0] <= cutoff:
                    self._request_starts.popleft()
                if len(self._request_starts) < self.max_requests:
                    self._request_starts.append(now)
                    return
                wait_seconds = self.window_seconds - (now - self._request_starts[0])
                self._sleep(max(wait_seconds, 0.001))


class RateLimitedChatClient:
    """OpenAI-like chat client wrapper with one shared request-rate budget."""

    def __init__(self, client: Any, limiter: RequestsPerMinuteLimiter) -> None:
        self._client = client
        self._limiter = limiter
        self.chat = self
        self.completions = self

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # The slot is acquired for every real attempt, so retries also count
        # toward the configured requests-per-minute ceiling.
        self._limiter.acquire()
        return self._client.chat.completions.create(*args, **kwargs)


def _extract_task(text: str) -> str:
    """Extract the short AlfWorld task from its initial observation."""
    value = (text or "").strip()
    marker = "Your task is to:"
    index = value.find(marker)
    if index < 0:
        return value[:500]
    task = value[index + len(marker):]
    end = task.find("\n\n")
    return (task if end < 0 else task[:end]).strip()


def _select_task_groups(
    trajectories: Sequence[Dict[str, Any]],
    max_groups: int,
    max_trajectories_per_group: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Select task groups and sample representative evidence for reflection.

    Statistics are computed from every collected rollout, while the reflection
    evidence follows three cases: one success for all-success groups, one
    success plus one failure for mixed groups, and one failure for all-failure
    groups.
    """
    # Retain the legacy argument for callers, but do not truncate a sampled group:
    # rollouts_per_task already controls how many attempts the environment collects.
    del max_trajectories_per_group
    sampler = rng if rng is not None else random.Random(0)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[str(trajectory.get("group_uid", ""))].append(trajectory)
    selected: List[Dict[str, Any]] = []
    for group_uid, items in grouped.items():
        if not items:
            continue
        # Rollout indices encode the original collection order.  Keep that order
        # while sampling so the selected evidence can be audited reproducibly.
        group_trajectories = sorted(items, key=lambda item: str(item.get("traj_uid", "")))
        successful = [item for item in group_trajectories if item.get("succeeded")]
        failed = [item for item in group_trajectories if not item.get("succeeded")]
        if successful and failed:
            sampled = [sampler.choice(successful), sampler.choice(failed)]
            selection_mode = "one_success_one_failure"
        elif successful:
            sampled = [sampler.choice(successful)]
            selection_mode = "one_success"
        else:
            sampled = [sampler.choice(failed)]
            selection_mode = "one_failure"
        original_order = {
            str(item.get("traj_uid", "")): index
            for index, item in enumerate(group_trajectories)
        }
        sampled.sort(key=lambda item: original_order[str(item.get("traj_uid", ""))])
        success_count = len(successful)
        selected.append({
            "group_uid": group_uid,
            "group_success_rate": success_count / len(group_trajectories),
            "group_size": len(group_trajectories),
            "success_count": success_count,
            "failure_count": len(failed),
            "reflection_selection_mode": selection_mode,
            "sampled_trajectory_count": len(sampled),
            "sampled_success_count": sum(1 for item in sampled if item.get("succeeded")),
            "sampled_failure_count": sum(1 for item in sampled if not item.get("succeeded")),
            "sampled_traj_uids": [str(item.get("traj_uid", "")) for item in sampled],
            "trajectories": deepcopy(sampled),
        })
    selected.sort(key=lambda group: (float(group["group_success_rate"]), str(group["group_uid"])))
    return selected[: max(0, int(max_groups))]


# Compatibility for callers that imported the old helper name.  Its semantics
# now match _select_task_groups: no mixed-outcome requirement or balancing.
_select_contrastive_task_groups = _select_task_groups


def _detect_task_type(task: str) -> str:
    text = (task or "").lower()
    if "clean" in text:
        return "pick_clean_then_place_in_recep"
    if "heat" in text:
        return "pick_heat_then_place_in_recep"
    if "cool" in text:
        return "pick_cool_then_place_in_recep"
    if "look at" in text and ("lamp" in text or "light" in text):
        return "look_at_obj_in_light"
    if "two" in text or "both" in text:
        return "pick_two_obj_and_place"
    return "pick_and_place"


def _task_uid(info: Dict[str, Any], task: str) -> str:
    """Prefer AlfWorld's game file because all rollouts of one task share it."""
    return str(
        info.get("extra.gamefile")
        or info.get("gamefile")
        or info.get("task_id")
        or task
    )


def _build_reasoning_prompt(
    task: str,
    observation: str,
    admissible_actions: Sequence[str],
    history: Sequence[Dict[str, str]],
    step: int,
    history_length: int,
) -> str:
    recent = list(history[-max(0, history_length):]) if history_length else []
    if recent:
        history_text = "\n".join(
            f"Observation: {item['observation']}\nAction: {item['action']}"
            for item in recent
        )
    else:
        history_text = "(none)"
    actions = "\n".join(f"- {action}" for action in admissible_actions if action != "help")
    return f"""You are an expert agent operating in the ALFRED embodied environment.

Task: {task}
Current step: {step}

Recent interaction history:
{history_text}

Current observation:
{observation}

Admissible actions:
{actions}

Reason step by step inside <think>...</think>, then choose exactly one admissible
action inside <action>...</action>. Do not put any other text inside the action tags."""


def _extract_action(response: str) -> str:
    match = re.search(r"<action>\s*(.*?)\s*</action>", response or "", re.I | re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip().lower()
    # Keep formatting failures visible to the environment instead of silently
    # replacing the model's decision with an oracle-chosen admissible action.
    return re.sub(r"\s+", " ", (response or "")[-30:]).strip().lower() or "look"


class OpenAICompatiblePolicy:
    """Concurrent external-API policy used only for offline data collection."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.8,
        max_completion_tokens: int = 512,
        max_concurrent: int = 32,
        timeout: float = 120.0,
        retries: int = 10,
        client: Optional[Any] = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=_normalise_base_url(base_url),
                timeout=timeout,
            )
        self.client = client
        self.model = model
        self.temperature = float(temperature)
        self.max_completion_tokens = int(max_completion_tokens)
        self.max_concurrent = max(1, int(max_concurrent))
        self.retries = max(1, int(retries))

    def _complete_one(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # external services expose backend-specific exceptions
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(5 * (2 ** attempt), 60))
        print(
            f"[ReasoningPolicy] Skipping completion after {self.retries} failed attempts: "
            f"{last_error}",
            file=sys.stderr,
        )
        return ""

    def complete_batch(self, prompts: Sequence[str]) -> List[str]:
        if not prompts:
            return []
        outputs = [""] * len(prompts)
        workers = min(self.max_concurrent, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._complete_one, prompt): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                outputs[futures[future]] = future.result()
        return outputs


class SophonChatCompletionAdapter:
    """Expose Sophon's custom HTTP endpoint through an OpenAI-like client API."""

    def __init__(self, endpoint: str, api_key: str, timeout: float, cache: bool = True) -> None:
        self.endpoint = endpoint.strip()
        self.api_key = api_key.strip()
        self.timeout = float(timeout)
        self.cache = bool(cache)
        # SkillUpdater and OpenAICompatiblePolicy both call
        # ``client.chat.completions.create``.
        self.chat = self
        self.completions = self

    @staticmethod
    def _content_from_response(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"])
            if choice.get("text") is not None:
                return str(choice["text"])
        data = payload.get("data")
        if isinstance(data, dict):
            nested = SophonChatCompletionAdapter._content_from_response(data)
            if nested:
                return nested
        message = payload.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message["content"])
        for key in ("content", "result", "output"):
            if payload.get(key) is not None and not isinstance(payload[key], (dict, list)):
                return str(payload[key])
        return ""

    def create(
        self,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float = 1.0,
        **_: Any,
    ) -> Any:
        import requests

        response = requests.post(
            self.endpoint,
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": list(messages),
                "temperature": float(temperature),
                "n": 1,
                "stream": False,
                "cache": self.cache,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        response_payload = response.json()
        content = self._content_from_response(response_payload)
        if not content:
            raise RuntimeError(
                "Sophon response did not contain chat content; "
                f"top-level keys={list(response_payload.keys())}"
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


@dataclass
class GenerationConfig:
    output_path: Path
    audit_path: Path
    target_task_bundles: int = 300
    task_groups_per_batch: int = 10
    rollouts_per_task: int = 10
    max_step_skills_per_task: int = 8
    max_steps: int = 50
    history_length: int = 2
    batches_per_pass: int = 356
    max_passes: int = 3
    seed: int = 0
    max_concurrent_api_calls: int = 8
    max_api_requests_per_minute: int = 60


class AlfWorldInitialSkillBankGenerator:
    """Environment interaction and all-rollout skill generation orchestration."""

    def __init__(
        self,
        config: GenerationConfig,
        envs: Any,
        reasoning_policy: Any,
        skill_updater: Any,
        memory: Any,
    ) -> None:
        self.config = config
        self.envs = envs
        self.reasoning_policy = reasoning_policy
        self.skill_updater = skill_updater
        self.memory = memory
        self._trajectory_sampler = random.Random(int(config.seed))
        self.audit: Dict[str, Any] = {
            "mode": "offline_alfworld_task_group_initialization",
            "target_task_bundles": min(300, max(1, int(config.target_task_bundles))),
            "api_limits": {
                "max_concurrent_calls": int(config.max_concurrent_api_calls),
                "max_requests_per_minute": int(config.max_api_requests_per_minute),
            },
            "batches": [],
        }

    def _collect_rollout_batch(self) -> List[Dict[str, Any]]:
        observations, _, infos = self.envs.reset()
        count = len(observations)
        tasks = [_extract_task(obs) for obs in observations]
        task_uids = [_task_uid(infos[i], tasks[i]) for i in range(count)]
        histories: List[List[Dict[str, str]]] = [[] for _ in range(count)]
        turns: List[List[Dict[str, Any]]] = [[] for _ in range(count)]
        episode_rewards = [0.0] * count
        successes = [False] * count
        done = [False] * count

        for step in range(1, self.config.max_steps + 1):
            active_indices = [index for index, value in enumerate(done) if not value]
            if not active_indices:
                break
            prompts = [
                _build_reasoning_prompt(
                    tasks[index],
                    observations[index],
                    self.envs.get_admissible_commands[index],
                    histories[index],
                    step,
                    self.config.history_length,
                )
                for index in active_indices
            ]
            responses = self.reasoning_policy.complete_batch(prompts)
            actions = ["look"] * count
            raw_by_index: Dict[int, str] = {}
            for index, response in zip(active_indices, responses):
                actions[index] = _extract_action(response)
                raw_by_index[index] = response

            next_observations, _, rewards, dones, step_infos = self.envs.step(actions)
            for index in active_indices:
                reward = float(rewards[index])
                won = bool(float(step_infos[index].get("won", 0.0) or 0.0))
                turn = {
                    "observation": str(observations[index]),
                    "action": actions[index],
                    "raw_model_response": raw_by_index[index],
                    "reward": reward,
                }
                turns[index].append(turn)
                histories[index].append({
                    "observation": str(observations[index]),
                    "action": actions[index],
                })
                episode_rewards[index] += reward
                successes[index] = successes[index] or won
                done[index] = bool(dones[index])
            observations = next_observations

        trajectories: List[Dict[str, Any]] = []
        for index in range(count):
            trajectories.append({
                "traj_uid": f"{task_uids[index]}::rollout-{index}",
                "group_uid": task_uids[index],
                "attempt_idx": 0,
                "task": tasks[index],
                "task_type": _detect_task_type(tasks[index]),
                "episode_reward": episode_rewards[index],
                "episode_length": len(turns[index]),
                "succeeded": successes[index],
                "full_dialogue": True,
                "refined_trajectory": {
                    "task": tasks[index],
                    "turns": [
                        {"observation": turn["observation"], "action": turn["action"]}
                        for turn in turns[index]
                    ],
                },
            })
        return trajectories

    def _save(self) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory.save_skills(str(self.config.output_path))
        with self.config.audit_path.open("w", encoding="utf-8") as handle:
            json.dump(self.audit, handle, indent=2, ensure_ascii=False)

    def run(self) -> Dict[str, Any]:
        # Initial banks are deliberately capped at 300 complete task bundles.
        target = min(300, max(1, int(self.config.target_task_bundles)))
        for pass_index in range(1, max(1, self.config.max_passes) + 1):
            added_this_pass = 0
            for batch_index in range(1, max(1, self.config.batches_per_pass) + 1):
                current = len(self.memory.skills.get("task_skills", []))
                if current >= target:
                    break
                trajectories = self._collect_rollout_batch()
                remaining = target - current
                groups = _select_task_groups(
                    trajectories,
                    max_groups=min(self.config.task_groups_per_batch, remaining),
                    max_trajectories_per_group=self.config.rollouts_per_task,
                    rng=self._trajectory_sampler,
                )
                existing_group_uids = {
                    str(skill.get("initialization_group_uid"))
                    for skill in self.memory.skills.get("task_skills", [])
                    if skill.get("initialization_group_uid") is not None
                }
                groups = [
                    group for group in groups
                    if str(group.get("group_uid")) not in existing_group_uids
                ][:remaining]
                metadata: Dict[str, Any] = {}
                added = {"task_skills": 0, "step_skills": 0}
                if groups:
                    pairs, metadata = self.skill_updater.analyze_task_groups(
                        groups,
                        current_skills=self.memory.skills,
                        return_metadata=True,
                    )
                    before = len(self.memory.skills.get("task_skills", []))
                    added = self.memory.add_hierarchical_skill_pairs(pairs, created_at_step=0)
                    after = len(self.memory.skills.get("task_skills", []))
                    added_this_pass += after - before

                successes = sum(1 for item in trajectories if item.get("succeeded"))
                batch_audit = {
                    "pass_index": pass_index,
                    "batch_index": batch_index,
                    "trajectory_count": len(trajectories),
                    "success_count": successes,
                    "task_groups": len(groups),
                    "task_group_uids": [group.get("group_uid") for group in groups],
                    "reflection_evidence": [
                        {
                            "group_uid": group.get("group_uid"),
                            "group_size": group.get("group_size"),
                            "success_count": group.get("success_count"),
                            "failure_count": group.get("failure_count"),
                            "group_success_rate": group.get("group_success_rate"),
                            "selection_mode": group.get("reflection_selection_mode"),
                            "sampled_traj_uids": group.get("sampled_traj_uids"),
                            "sampled_success_count": group.get("sampled_success_count"),
                            "sampled_failure_count": group.get("sampled_failure_count"),
                        }
                        for group in groups
                    ],
                    "added": added,
                    "task_bundle_count": len(self.memory.skills.get("task_skills", [])),
                    "reflection": metadata,
                }
                self.audit["batches"].append(batch_audit)
                if added.get("task_skills", 0):
                    self._save()
                print(
                    "[InitialSkillBank] "
                    f"pass={pass_index} batch={batch_index} "
                    f"success={successes}/{len(trajectories)} groups={len(groups)} "
                    f"bundles={len(self.memory.skills.get('task_skills', []))}/{target}"
                )

            if len(self.memory.skills.get("task_skills", [])) >= target:
                break
            if added_this_pass == 0:
                print("[InitialSkillBank] A complete pass added no bundles; stopping.")
                break

        final_count = len(self.memory.skills.get("task_skills", []))
        self.audit["final_task_bundles"] = final_count
        self.audit["final_step_skills"] = len(self.memory.skills.get("step_skills", []))
        self._save()
        if final_count == 0:
            raise RuntimeError(
                "No initial task bundle was generated. No task-local rollout group was available, "
                "or the reflection API returned no valid bundle."
            )
        return self.memory.skills


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interact with AlfWorld through an external LLM API and create an initial skill JSON."
    )
    parser.add_argument("--api-base-url", default=os.getenv("OPENAI_BASE_URL"), required=not bool(os.getenv("OPENAI_BASE_URL")))
    parser.add_argument(
        "--api-provider",
        choices=("openai", "sophon"),
        default="openai",
        help="Use 'sophon' for the ByteDance gateway chatCompletion endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.getenv("SOPHON_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SKILL_LLM_API_KEY")
            or "EMPTY"
        ),
        help="Defaults to SOPHON_API_KEY/OPENAI_API_KEY/SKILL_LLM_API_KEY, then EMPTY.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"), required=not bool(os.getenv("OPENAI_MODEL")))
    parser.add_argument("--reflection-model", default=None, help="Defaults to --model on the same API.")
    parser.add_argument(
        "--dataset-name",
        default="alfworld",
        help="Dataset component used in the automatic output filename. Defaults to alfworld.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Explicit Skill JSON output path. By default the final filename is "
            "<dataset>_<model>_<actual-task-bundle-count>.json under "
            "initial_skill_bank/skill_bank/."
        ),
    )
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-task-bundles", type=int, default=300)
    parser.add_argument("--task-groups-per-batch", type=int, default=1)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--max-step-skills-per-task", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--batches-per-pass", type=int, default=356)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--reasoning-max-tokens", type=int, default=512)
    parser.add_argument("--reflection-max-tokens", type=int, default=2048)
    parser.add_argument("--max-concurrent-api-calls", type=int, default=8)
    parser.add_argument(
        "--max-api-requests-per-minute",
        type=int,
        default=60,
        help=(
            "Global request-start limit shared by Reasoning and Reflection calls, "
            "including retries. Defaults to 60."
        ),
    )
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-retries", type=int, default=10)
    parser.add_argument("--disable-api-cache", action="store_true", help="Disable Sophon response caching.")
    parser.add_argument("--num-cpus-per-env-worker", type=float, default=0.1)
    parser.add_argument(
        "--alfworld-config",
        type=Path,
        default=PROJECT_ROOT / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
    if args.max_concurrent_api_calls < 1:
        parser.error("--max-concurrent-api-calls must be positive")
    if args.max_api_requests_per_minute < 1:
        parser.error("--max-api-requests-per-minute must be positive")
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

    # These dependencies are needed only for a real generation run; keeping the
    # imports here makes ``--help`` and the mock tests independent of AlfWorld,
    # NumPy, OpenAI, and Ray installations.
    from agent_system.memory.skill_updater import SkillUpdater
    from agent_system.memory.skills_only_memory import SkillsOnlyMemory

    base_client = None
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
        "[InitialSkillBank] API limits: "
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
    updater_kwargs = {
        "skill_gen_mode": "task_step",
        "skill_llm_api_key": args.api_key,
        "skill_llm_model": args.reflection_model or args.model,
        "max_completion_tokens": args.reflection_max_tokens,
        "max_concurrent": args.max_concurrent_api_calls,
        "success_max_step_skills": args.max_step_skills_per_task,
        "chat_client": shared_client,
    }
    updater = SkillUpdater(**updater_kwargs)
    memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")

    # Importing the environment starts Ray and loads AlfWorld, so keep it out of
    # module import paths used by unit tests and training.
    from agent_system.environments.env_package.alfworld import build_alfworld_envs

    envs = build_alfworld_envs(
        str(args.alfworld_config.expanduser().resolve()),
        seed=args.seed,
        env_num=args.task_groups_per_batch,
        group_n=args.rollouts_per_task,
        resources_per_worker={"num_cpus": args.num_cpus_per_env_worker, "num_gpus": 0},
        is_train=True,
        env_kwargs={},
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
        generator = AlfWorldInitialSkillBankGenerator(config, envs, policy, updater, memory)
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
                    f"Pass --overwrite to replace it. New artifacts remain at {output} and {audit_output}."
                )
        output = _move_generated_file(output, final_output, overwrite=args.overwrite)
        if audit_output != final_audit_output:
            audit_output = _move_generated_file(
                audit_output,
                final_audit_output,
                overwrite=args.overwrite,
            )
    print(f"[InitialSkillBank] Skill bank: {output}")
    print(f"[InitialSkillBank] Audit: {audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
