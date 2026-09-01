import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from verl import DataProto


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = _load_module(
    "skill_agent_training_helpers",
    "agent_system/multi_turn_rollout/skill_agent_training.py",
)
rollout_utils = _load_module(
    "skill_agent_rollout_utils",
    "agent_system/multi_turn_rollout/utils.py",
)
core_gigpo = _load_module(
    "skill_agent_core_gigpo",
    "gigpo/core_gigpo.py",
)


def _actor_batch(markers, *, include_loss_mask):
    batch_size = len(markers)
    markers = torch.tensor(markers, dtype=torch.long)
    input_ids = torch.stack([markers, markers + 1, markers + 2, markers + 3], dim=1)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
        "position_ids": torch.arange(4).repeat(batch_size, 1),
        "responses": input_ids[:, -2:],
        "old_log_probs": torch.zeros(batch_size, 2),
        "ref_log_prob": torch.zeros(batch_size, 2),
        "advantages": torch.ones(batch_size, 2),
    }
    if include_loss_mask:
        tensors["loss_mask"] = torch.tensor([[0, 0, 1, 1]]).repeat(batch_size, 1)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={"role": np.asarray(["unused"] * batch_size, dtype=object)},
        meta_info={"temperature": 1.0},
    )


class SkillAgentTrainingTest(unittest.TestCase):
    def test_episode_stats_keep_verl_agent_cross_step_behavior(self):
        token_rewards = torch.tensor([[10.0], [10.0], [0.0]])
        response_mask = torch.ones_like(token_rewards)
        task_uids = np.asarray(["task", "task", "task"], dtype=object)
        traj_uids = np.asarray(["long", "long", "short"], dtype=object)

        advantages = core_gigpo.episode_norm_reward(
            token_rewards,
            response_mask,
            task_uids,
            traj_uids,
            remove_std=True,
            compute_mean_std_cross_steps=True,
        )

        torch.testing.assert_close(
            advantages[:, 0],
            torch.tensor([10.0 / 3.0, 10.0 / 3.0, -20.0 / 3.0]),
        )

    def test_episode_improvement_is_discounted_for_step_advantage(self):
        batch = DataProto.from_dict(
            tensors={"input_ids": torch.ones(3, 1, dtype=torch.long)},
            non_tensors={
                "rewards": np.asarray([0.0, 0.0, 10.0], dtype=np.float32),
                "traj_uid": np.asarray(["edit", "edit", "edit"], dtype=object),
                "active_masks": np.asarray([True, True, True]),
            },
        )

        step_returns = core_gigpo.compute_step_discounted_returns(batch, gamma=0.95)

        torch.testing.assert_close(
            step_returns,
            torch.tensor([9.025, 9.5, 10.0]),
        )

    def test_reasoning_step_returns_do_not_cross_attempt_trajectory_ids(self):
        batch = DataProto.from_dict(
            tensors={"input_ids": torch.ones(4, 1, dtype=torch.long)},
            non_tensors={
                "rewards": np.asarray([0.0, 10.0, 0.0, 20.0], dtype=np.float32),
                "traj_uid": np.asarray(
                    ["rollout:attempt:0", "rollout:attempt:0",
                     "rollout:attempt:1", "rollout:attempt:1"],
                    dtype=object,
                ),
                "active_masks": np.asarray([True, True, True, True]),
            },
        )

        step_returns = core_gigpo.compute_step_discounted_returns(batch, gamma=0.5)

        torch.testing.assert_close(
            step_returns,
            torch.tensor([5.0, 10.0, 10.0, 20.0]),
        )

    def test_step_advantage_still_groups_exact_observations(self):
        advantages, _ = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=torch.zeros(3, 1),
            step_rewards=torch.tensor([10.0, 0.0, 5.0]),
            response_mask=torch.ones(3, 1),
            anchor_obs=np.asarray(["same", "same", "unique"], dtype=object),
            index=np.asarray(["task", "task", "task"], dtype=object),
            traj_index=np.asarray(["a", "b", "c"], dtype=object),
            step_advantage_w=1.0,
            mode="mean_norm",
            enable_similarity=False,
        )

        torch.testing.assert_close(
            advantages[:, 0],
            torch.tensor([5.0, -5.0, 0.0]),
        )

    def test_joint_actor_batch_interleaves_roles_and_keeps_kl_fields(self):
        reasoning = _actor_batch([10, 20], include_loss_mask=True)
        skill = _actor_batch([100], include_loss_mask=False)

        joint = training.build_joint_actor_update_batch(
            reasoning,
            skill,
            use_kl_loss=True,
            multi_turn=True,
        )

        self.assertEqual(joint.batch["input_ids"][:, 0].tolist(), [10, 100, 20])
        self.assertIn("ref_log_prob", joint.batch)
        self.assertEqual(joint.batch["loss_mask"][0].tolist(), [0, 0, 1, 1])
        self.assertEqual(joint.batch["loss_mask"][1].tolist(), [1, 1, 1, 1])
        self.assertEqual(joint.non_tensor_batch, {})
        self.assertTrue(joint.meta_info["joint_agent_update"])
        self.assertEqual(joint.meta_info["reasoning_rows"], 2)
        self.assertEqual(joint.meta_info["skill_agent_rows"], 1)
        self.assertEqual(joint.meta_info["global_token_num"], [4, 4, 4])

    def test_joint_actor_batch_rejects_missing_advantages(self):
        reasoning = _actor_batch([10], include_loss_mask=True)
        skill = _actor_batch([100], include_loss_mask=True)
        skill.batch.pop("advantages")

        with self.assertRaisesRegex(ValueError, "advantages"):
            training.build_joint_actor_update_batch(
                reasoning,
                skill,
                use_kl_loss=False,
                multi_turn=True,
            )

    def test_adjust_batch_can_pad_batch_smaller_than_one_micro_batch(self):
        batch = DataProto.from_dict(
            tensors={"input_ids": torch.arange(3).reshape(1, 3)},
            non_tensors={"uid": np.asarray(["one"], dtype=object)},
        )
        config = SimpleNamespace(
            trainer=SimpleNamespace(n_gpus_per_node=2, nnodes=1),
            algorithm=SimpleNamespace(use_kl_in_reward=False),
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=2),
                ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=2),
                actor=SimpleNamespace(
                    use_kl_loss=False,
                    ppo_mini_batch_size=4,
                    ppo_micro_batch_size_per_gpu=2,
                ),
            ),
        )

        adjusted = rollout_utils.adjust_batch(config, batch)

        self.assertEqual(len(adjusted), 4)
        self.assertEqual(adjusted.batch["input_ids"].tolist(), [[0, 1, 2]] * 4)


if __name__ == "__main__":
    unittest.main()
