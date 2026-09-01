"""Pure helpers for initializing and promoting hierarchical skill bundles."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EDITABLE_FIELDS = ("title", "principle", "when_to_apply", "retrieval_obs")


def bundle_fingerprint(task_skill: Dict[str, Any], step_skills: Iterable[Dict[str, Any]]) -> str:
    """Return an ID-independent fingerprint for one task bundle version."""
    task = {key: str(task_skill.get(key, "")).strip() for key in EDITABLE_FIELDS}
    steps = [
        {key: str(step.get(key, "")).strip() for key in EDITABLE_FIELDS}
        for step in step_skills
    ]
    steps.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    payload = json.dumps({"task": task, "steps": steps}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_promotion_candidates(
    trajectory_sidecars: Sequence[Sequence[Dict[str, Any]]],
    max_per_group: int = 2,
    require_effective_edit: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate edited overlays and select the best candidates per task group."""
    accepted_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejected: List[Dict[str, Any]] = []
    for records in trajectory_sidecars:
        records = list(records or [])
        if not records:
            continue
        head = records[0]
        group_uid = str(head.get("uid", ""))
        traj_uid = str(head.get("traj_uid", ""))
        overlay = head.get("meta_attempt_overlay") or {}
        final = overlay.get("final") or {}
        tasks = list(final.get("task_skills") or [])
        steps = list(final.get("step_skills") or [])
        successes = list(head.get("attempt_successes") or [])
        rewards = list(head.get("attempt_rewards") or [])
        reason = None
        if len(successes) < 2:
            reason = "missing_attempt_successes"
        elif len(tasks) != 1:
            reason = "overlay_must_contain_one_task"
        elif not steps:
            reason = "empty_step_bundle"
        patches = list(overlay.get("patches") or [])
        effective_edits = sum(
            1 for patch in patches
            if patch.get("applied")
            and str((patch.get("effect") or {}).get("action", "")).upper() in {"INSERT", "UPDATE", "DELETE"}
        )
        if reason is None and require_effective_edit and effective_edits == 0:
            reason = "no_effective_edit"
        if reason is None:
            baseline = float(successes[0])
            edited_successes = [float(value) for value in successes[1:]]
            edited_rate = sum(edited_successes) / len(edited_successes)
            improvement = edited_rate - baseline
            if improvement <= 0:
                reason = "no_success_improvement"
        if reason is not None:
            rejected.append({"group_uid": group_uid, "traj_uid": traj_uid, "reason": reason})
            continue
        candidate = {
            "group_uid": group_uid,
            "traj_uid": traj_uid,
            "task_skill": deepcopy(tasks[0]),
            "step_skills": deepcopy(steps),
            "attempt_successes": [float(value) for value in successes],
            "attempt_rewards": [float(value) for value in rewards],
            "baseline_success": baseline,
            "edited_success_rate": edited_rate,
            "improvement": improvement,
            "effective_edits": effective_edits,
            "bundle_fingerprint": bundle_fingerprint(tasks[0], steps),
        }
        accepted_by_group[group_uid].append(candidate)

    accepted: List[Dict[str, Any]] = []
    for group_uid in sorted(accepted_by_group):
        candidates = accepted_by_group[group_uid]
        candidates.sort(key=lambda item: (
            -item["edited_success_rate"],
            -item["improvement"],
            -(
                sum(item["attempt_rewards"][1:]) / len(item["attempt_rewards"][1:])
                if len(item["attempt_rewards"]) >= 2
                else 0.0
            ),
            item["traj_uid"],
        ))
        accepted.extend(candidates[: max(0, int(max_per_group))])
        for candidate in candidates[max(0, int(max_per_group)):]:
            rejected.append({
                "group_uid": group_uid,
                "traj_uid": candidate["traj_uid"],
                "reason": "outside_group_top_k",
            })
    return accepted, rejected
