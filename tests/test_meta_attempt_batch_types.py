import unittest

import numpy as np
import torch

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl.trainer.ppo.metric_utils import compute_data_metrics


def _rollout_row(
    traj_uid,
    ppo_episode_reward=None,
    ppo_episode_length=None,
    base_traj_uid=None,
):
    row = {
        "traj_uid": traj_uid,
        "active_masks": True,
        "token_level_scores": torch.tensor([1.0, 0.0]),
        "token_level_rewards": torch.tensor([0.5, 0.0]),
        "advantages": torch.tensor([0.2, 0.0]),
        "returns": torch.tensor([0.3, 0.0]),
        "responses": torch.tensor([1, 0]),
        "attention_mask": torch.tensor([1, 1, 1, 0]),
    }
    if ppo_episode_reward is not None:
        row["ppo_episode_reward"] = ppo_episode_reward
        row["ppo_episode_length"] = ppo_episode_length
    if base_traj_uid is not None:
        row["base_traj_uid"] = base_traj_uid
    return row


class MetaAttemptBatchTypesTest(unittest.TestCase):
    def _gather(self, rows, episode_rewards, episode_lengths, outer_traj_uids=None):
        collector = TrajectoryCollector.__new__(TrajectoryCollector)
        batch_size = len(rows)
        return collector.gather_rollout_data(
            total_batch_list=[[row] for row in rows],
            episode_rewards=np.asarray(episode_rewards, dtype=np.float32),
            episode_lengths=np.asarray(episode_lengths, dtype=np.float32),
            success={"success_rate": np.zeros(batch_size, dtype=np.float32)},
            traj_uid=np.asarray(
                outer_traj_uids
                if outer_traj_uids is not None
                else [row["traj_uid"] for row in rows],
                dtype=object,
            ),
            tool_callings=np.zeros(batch_size, dtype=np.float32),
        )

    def test_ordinary_rollout_episode_fields_remain_numpy_float32(self):
        batch = self._gather(
            [_rollout_row("ordinary")],
            episode_rewards=[7.5],
            episode_lengths=[4.0],
        )

        self.assertIsInstance(batch.non_tensor_batch["episode_rewards"][0], np.float32)
        self.assertIsInstance(batch.non_tensor_batch["episode_lengths"][0], np.float32)
        self.assertEqual(batch.non_tensor_batch["episode_rewards"][0], np.float32(7.5))
        self.assertEqual(batch.non_tensor_batch["episode_lengths"][0], np.float32(4.0))

    def test_meta_reasoning_and_skill_rows_use_numpy_float32_without_changing_rewards(self):
        batch = self._gather(
            [
                _rollout_row("reasoning", ppo_episode_reward=3.0, ppo_episode_length=2),
                _rollout_row("skill", ppo_episode_reward=0.5, ppo_episode_length=2),
            ],
            episode_rewards=[10.0, 11.0],
            episode_lengths=[8.0, 9.0],
        )

        for field in (
            "ppo_episode_reward",
            "ppo_episode_length",
            "episode_rewards",
            "episode_lengths",
        ):
            self.assertTrue(all(isinstance(value, np.float32) for value in batch.non_tensor_batch[field]))
        np.testing.assert_allclose(
            list(batch.non_tensor_batch["episode_rewards"]),
            np.asarray([3.0, 0.5], dtype=np.float32),
        )
        np.testing.assert_allclose(
            list(batch.non_tensor_batch["episode_lengths"]),
            np.asarray([2.0, 2.0], dtype=np.float32),
        )

        # Exercise the original VERL reductions that call `.item()` on the
        # object-array reduction results. This is the production regression.
        metrics = compute_data_metrics(batch, use_critic=False)
        self.assertAlmostEqual(metrics["episode/reward/mean"], np.float32((3.0 + 0.5) / 2), places=5)
        self.assertAlmostEqual(metrics["episode/reward/max"], np.float32(3.0), places=5)
        self.assertAlmostEqual(metrics["episode/reward/min"], np.float32(0.5), places=5)
        self.assertAlmostEqual(metrics["episode/length/mean"], 2.0)
        self.assertAlmostEqual(metrics["episode/length/max"], 2.0)
        self.assertAlmostEqual(metrics["episode/length/min"], 2.0)

    def test_meta_reasoning_row_accepts_attempt_specific_trajectory_id(self):
        batch = self._gather(
            [
                _rollout_row(
                    "base:attempt:1",
                    ppo_episode_reward=10.0,
                    ppo_episode_length=2,
                    base_traj_uid="base",
                )
            ],
            episode_rewards=[10.0],
            episode_lengths=[2.0],
            outer_traj_uids=["base"],
        )

        self.assertEqual(batch.non_tensor_batch["traj_uid"][0], "base:attempt:1")
        self.assertEqual(batch.non_tensor_batch["base_traj_uid"][0], "base")


if __name__ == "__main__":
    unittest.main()
