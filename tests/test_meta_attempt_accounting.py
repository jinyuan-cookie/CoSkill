import importlib.util
from pathlib import Path
import unittest


def _load_accounting():
    path = Path(__file__).resolve().parents[1] / "agent_system" / "multi_turn_rollout" / "meta_attempt.py"
    spec = importlib.util.spec_from_file_location("meta_attempt_accounting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.annotate_meta_attempt_returns, module.build_reasoning_attempt_traj_uid


annotate_meta_attempt_returns, build_reasoning_attempt_traj_uid = _load_accounting()


class MetaAttemptAccountingTest(unittest.TestCase):
    def test_reasoning_attempt_trajectory_ids_keep_attempt_returns_separate(self):
        base_uid = "base-trajectory"

        attempt0 = build_reasoning_attempt_traj_uid(base_uid, 0)
        attempt1 = build_reasoning_attempt_traj_uid(base_uid, 1)

        self.assertEqual(attempt0, "base-trajectory:attempt:0")
        self.assertEqual(attempt1, "base-trajectory:attempt:1")
        self.assertNotEqual(attempt0, attempt1)

    def test_reasoning_episode_reward_is_copied_to_every_step_without_discounting(self):
        reasoning = [[
            {"traj_uid": "t0", "anchor_obs": "a", "attempt_idx": 0, "rewards": 0.0},
            {"traj_uid": "t0", "anchor_obs": "b", "attempt_idx": 0, "rewards": 0.0},
            {"traj_uid": "t0", "anchor_obs": "c", "attempt_idx": 0, "rewards": 10.0},
        ]]

        annotate_meta_attempt_returns(
            reasoning,
            [[]],
            [[10.0]],
        )

        self.assertEqual([row["ppo_episode_reward"] for row in reasoning[0]], [10.0, 10.0, 10.0])
        self.assertEqual([row["attempt_return"] for row in reasoning[0]], [10.0, 10.0, 10.0])
        self.assertEqual([row["ppo_episode_length"] for row in reasoning[0]], [3, 3, 3])
        self.assertTrue(all("reasoning_step_return" not in row for row in reasoning[0]))

    def test_reasoning_episode_rewards_do_not_cross_attempt_boundaries(self):
        reasoning = [[
            {"traj_uid": "t0", "anchor_obs": "a", "attempt_idx": 0, "rewards": 0.0},
            {"traj_uid": "t0", "anchor_obs": "b", "attempt_idx": 0, "rewards": 0.0},
            {"traj_uid": "t0", "anchor_obs": "a", "attempt_idx": 1, "rewards": 0.0},
            {"traj_uid": "t0", "anchor_obs": "c", "attempt_idx": 1, "rewards": 10.0},
        ]]

        annotate_meta_attempt_returns(
            reasoning,
            [[]],
            [[0.0, 10.0]],
        )

        self.assertEqual([row["ppo_episode_reward"] for row in reasoning[0]], [0.0, 0.0, 10.0, 10.0])

    def test_two_attempt_skill_credit_uses_single_edited_attempt(self):
        reasoning = [[
            {"traj_uid": "t0", "anchor_obs": "same", "attempt_idx": 0, "rewards": 1.0},
            {"traj_uid": "t0", "anchor_obs": "same", "attempt_idx": 1, "rewards": 4.0},
        ]]
        skill = [[
            {"traj_uid": "t0", "anchor_obs": "same", "rewards": 1.0},
        ]]

        annotate_meta_attempt_returns(
            reasoning, skill, [[1.0, 4.0]],
        )

        self.assertAlmostEqual(skill[0][0]["rewards"], 3.0)
        self.assertAlmostEqual(skill[0][0]["ppo_episode_reward"], 3.0)
        self.assertAlmostEqual(skill[0][0]["skill_baseline_episode_reward"], 1.0)
        self.assertAlmostEqual(skill[0][0]["skill_edited_episode_reward_mean"], 4.0)
        self.assertAlmostEqual(skill[0][0]["skill_episode_improvement_reward"], 3.0)

    def test_reasoning_uses_attempt_outcome_and_skill_uses_mean_episode_improvement(self):
        reasoning = [[
            {"traj_uid": "t0", "anchor_obs": "a", "attempt_idx": 0, "rewards": 1.0},
            {"traj_uid": "t0", "anchor_obs": "b", "attempt_idx": 0, "rewards": 2.0},
            {"traj_uid": "t0", "anchor_obs": "a", "attempt_idx": 1, "rewards": 3.0},
            {"traj_uid": "t0", "anchor_obs": "a", "attempt_idx": 2, "rewards": 4.0},
        ]]
        skill = [[
            {"traj_uid": "t0", "anchor_obs": "a", "rewards": 0.1},
            {"traj_uid": "t0", "anchor_obs": "b", "rewards": 0.2},
        ]]

        annotate_meta_attempt_returns(
            reasoning, skill, [[3.0, 3.0, 4.0]],
        )

        self.assertEqual([row["attempt_return"] for row in reasoning[0]], [3.0, 3.0, 3.0, 4.0])
        self.assertEqual([row["ppo_episode_reward"] for row in reasoning[0]], [3.0, 3.0, 3.0, 4.0])
        self.assertTrue(all("reasoning_step_return" not in row for row in reasoning[0]))
        self.assertEqual([row["ppo_episode_length"] for row in reasoning[0]], [2, 2, 1, 1])
        # D = mean(R_1, R_2) - R_0 = (3 + 4) / 2 - 3 = 0.5.
        self.assertEqual([row["skill_agent_env_reward"] for row in skill[0]], [0.1, 0.2])
        self.assertEqual([row["skill_baseline_episode_reward"] for row in skill[0]], [3.0, 3.0])
        self.assertEqual([row["skill_edited_episode_reward_mean"] for row in skill[0]], [3.5, 3.5])
        self.assertEqual([row["skill_episode_improvement_reward"] for row in skill[0]], [0.5, 0.5])
        self.assertEqual([row["rewards"] for row in skill[0]], [0.0, 0.5])
        self.assertEqual([row["ppo_episode_reward"] for row in skill[0]], [0.5, 0.5])
        self.assertEqual([row["ppo_episode_length"] for row in skill[0]], [2, 2])
        removed_fields = (
            "skill_observation_delta_reward",
            "skill_observation_group",
            "skill_step_reward",
            "skill_edit_return",
        )
        self.assertTrue(all(field not in row for row in skill[0] for field in removed_fields))

    def test_skill_episode_improvement_can_be_negative(self):
        reasoning = [[
            {"traj_uid": "t0", "anchor_obs": "a", "attempt_idx": 0, "rewards": 10.0},
            {"traj_uid": "t0", "anchor_obs": "b", "attempt_idx": 1, "rewards": 0.0},
        ]]
        skill = [[
            {"traj_uid": "t0", "anchor_obs": "a", "rewards": 10.0},
            {"traj_uid": "t0", "anchor_obs": "b", "rewards": 0.0},
        ]]

        annotate_meta_attempt_returns(
            reasoning, skill, [[10.0, 0.0]],
        )

        self.assertEqual([row["rewards"] for row in skill[0]], [0.0, -10.0])
        self.assertEqual([row["ppo_episode_reward"] for row in skill[0]], [-10.0, -10.0])


if __name__ == "__main__":
    unittest.main()
