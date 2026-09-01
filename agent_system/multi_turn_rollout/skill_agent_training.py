"""Helpers for adding Skill-Agent samples to the shared actor PPO update."""

from copy import deepcopy
from typing import List

import torch

from verl import DataProto


_ACTOR_BATCH_KEYS = (
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "old_log_probs",
    "advantages",
)


def _actor_only_batch(
    batch: DataProto,
    *,
    use_kl_loss: bool,
    multi_turn: bool,
) -> DataProto:
    """Select only tensors consumed by ``update_actor``.

    Reasoning and Skill-Agent batches intentionally keep different non-tensor
    metadata.  Dropping sidecars here lets their policy tensors be concatenated
    without forcing unrelated schemas to match.
    """

    required: List[str] = list(_ACTOR_BATCH_KEYS)
    if use_kl_loss:
        required.append("ref_log_prob")
    missing = [key for key in required if key not in batch.batch]
    if missing:
        raise ValueError(f"actor training batch is missing required tensors: {missing}")

    tensors = {key: batch.batch[key] for key in required}
    if multi_turn:
        # Skill-Agent generations are one response per editor step.  Some
        # rollout backends do not emit loss_mask for those calls, in which case
        # the attention mask is the correct full-response training mask.
        tensors["loss_mask"] = (
            batch.batch["loss_mask"]
            if "loss_mask" in batch.batch
            else batch.batch["attention_mask"].clone()
        )
    return DataProto.from_dict(tensors=tensors, non_tensors={}, meta_info={})


def build_joint_actor_update_batch(
    reasoning_batch: DataProto,
    skill_batch: DataProto,
    *,
    use_kl_loss: bool,
    multi_turn: bool,
) -> DataProto:
    """Interleave Reasoning and Skill-Agent PPO rows for one shared update."""

    reasoning = _actor_only_batch(
        reasoning_batch, use_kl_loss=use_kl_loss, multi_turn=multi_turn
    )
    skill = _actor_only_batch(
        skill_batch, use_kl_loss=use_kl_loss, multi_turn=multi_turn
    )
    reasoning_rows = len(reasoning)
    skill_rows = len(skill)
    joint = DataProto.concat([reasoning, skill])

    # Avoid presenting all reasoning minibatches followed by all editor
    # minibatches.  Deterministic interleaving keeps both roles represented
    # throughout the shared optimizer update without changing their weights.
    order = []
    for index in range(max(reasoning_rows, skill_rows)):
        if index < reasoning_rows:
            order.append(index)
        if index < skill_rows:
            order.append(reasoning_rows + index)
    joint.reorder(torch.tensor(order, dtype=torch.long))

    joint.meta_info = deepcopy(reasoning_batch.meta_info)
    joint.meta_info.update({
        "multi_turn": bool(multi_turn),
        "joint_agent_update": True,
        "reasoning_rows": reasoning_rows,
        "skill_agent_rows": skill_rows,
        "global_token_num": torch.sum(joint.batch["attention_mask"], dim=-1).tolist(),
    })
    return joint
