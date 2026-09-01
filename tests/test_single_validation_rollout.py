from types import SimpleNamespace
from unittest import TestCase, skipIf
from unittest.mock import Mock, patch


try:
    import numpy as np
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import envs as alfworld_envs_module
    from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs, AlfworldWorker
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency availability is environment-specific
    IMPORT_ERROR = exc


class AttrDict(dict):
    __getattr__ = dict.__getitem__


@skipIf(IMPORT_ERROR is not None, f"rollout dependencies unavailable: {IMPORT_ERROR}")
class SingleValidationRolloutTest(TestCase):
    def _collector_config(self):
        return SimpleNamespace(
            env=AttrDict(
                max_steps=0,
                rollout=AttrDict(n=8),
                skill_agent=AttrDict(
                    enabled=True,
                    max_history_steps=4,
                    meta_attempts=AttrDict(enabled=True, num_attempts=3),
                ),
                skills_only_memory=AttrDict(skill_gen_mode="task_step"),
            )
        )

    def test_validation_bypasses_meta_attempts_and_skill_agent(self):
        collector = TrajectoryCollector.__new__(TrajectoryCollector)
        collector.config = self._collector_config()
        collector._meta_attempt_config = Mock(return_value={"enabled": True, "num_attempts": 3})
        collector._meta_attempt_multi_turn_loop = Mock(
            side_effect=AssertionError("validation must not enter meta-attempt rollout")
        )

        gen_batch = SimpleNamespace(batch=[object(), object()], non_tensor_batch={})
        envs = SimpleNamespace(
            retrieval_memory=None,
            with_skills_mask=None,
            reset=Mock(return_value=({"text": ["a", "b"], "image": None}, [{}, {}])),
            success_evaluator=Mock(return_value={"success_rate": np.zeros(2)}),
        )

        result = collector.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=object(),
            envs=envs,
            is_train=False,
        )

        self.assertIsNone(result[7])  # skill_agent_trajectories
        self.assertIsNone(result[8])  # skill_agent_batch_list
        collector._meta_attempt_multi_turn_loop.assert_not_called()

    def test_training_still_enters_meta_attempts(self):
        collector = TrajectoryCollector.__new__(TrajectoryCollector)
        collector.config = self._collector_config()
        meta_cfg = {"enabled": True, "num_attempts": 3}
        collector._meta_attempt_config = Mock(return_value=meta_cfg)
        sentinel = object()
        collector._meta_attempt_multi_turn_loop = Mock(return_value=sentinel)

        result = collector.vanilla_multi_turn_loop(
            gen_batch=object(), actor_rollout_wg=object(), envs=object(), is_train=True
        )

        self.assertIs(result, sentinel)
        collector._meta_attempt_multi_turn_loop.assert_called_once()

    def test_reset_retrieves_initial_child_step_skill(self):
        manager = AlfWorldEnvironmentManager.__new__(AlfWorldEnvironmentManager)
        manager.config = SimpleNamespace(
            env=AttrDict(
                use_skills_only_memory=True,
                rollout=AttrDict(n=8),
                skills_only_memory=AttrDict(
                    skill_gen_mode="task_step", top_k_task=1, top_k_step=1
                ),
            )
        )
        manager.val_rollout_always_skills = True
        manager.memory = SimpleNamespace(reset=Mock())
        manager.envs = SimpleNamespace(
            reset=Mock(return_value=(
                ["initial observation"],
                None,
                [{"extra.gamefile": "game-1"}],
            )),
            get_admissible_commands=[["look"]],
        )
        manager.extract_task = Mock(side_effect=lambda _obs: manager.tasks.append("task description"))
        manager.build_text_obs = Mock(return_value=["prompt with skills"])

        retrieval = Mock()
        retrieval.retrieve_task_skills_batch.return_value = [{
            "task_skills": [{"skill_id": "task-1", "step_skill_ids": ["step-1"]}],
            "query_text": "task description",
        }]
        retrieval.retrieve_child_step_skills_batch.return_value = [{
            "step_skills": [{"skill_id": "step-1", "parent_task_skill_id": "task-1"}],
            "query_text": "initial query",
        }]
        manager.retrieval_memory = retrieval

        obs, _ = manager.reset(kwargs=None)

        self.assertEqual(obs["text"], ["prompt with skills"])
        self.assertEqual(manager.retrieved_memories[0]["task_skills"][0]["skill_id"], "task-1")
        self.assertEqual(manager.retrieved_memories[0]["step_skills"][0]["skill_id"], "step-1")
        parent_ids, queries = retrieval.retrieve_child_step_skills_batch.call_args.args[:2]
        self.assertEqual(parent_ids, [["task-1"]])
        self.assertIn("initial observation", queries[0])

    def test_alfworld_worker_pool_uses_only_active_prefix(self):
        class RemoteMethod:
            def __init__(self, fn):
                self.fn = fn
                self.calls = 0

            def remote(self, *args):
                self.calls += 1
                return self.fn(*args)

        class Worker:
            def __init__(self, idx):
                self.reset = RemoteMethod(lambda: (
                    [f"obs-{idx}"],
                    {"admissible_commands": [["look"]], "extra.gamefile": [f"game-{idx}"]},
                ))
                self.restart = RemoteMethod(lambda: (
                    [f"restart-{idx}"],
                    {"admissible_commands": [["look"]], "extra.gamefile": [f"game-{idx}"]},
                ))
                self.step = RemoteMethod(lambda _action: (
                    [f"next-{idx}"],
                    [0.0],
                    [False],
                    {"admissible_commands": [["look"]], "won": [0.0]},
                ))
                self.getobs = RemoteMethod(lambda: f"image-{idx}")

        envs = AlfworldEnvs.__new__(AlfworldEnvs)
        envs.num_processes = 4
        envs.active_num_processes = 4
        envs.multi_modal = False
        envs.workers = [Worker(i) for i in range(4)]
        envs.prev_admissible_commands = [None] * 4
        envs.set_active_num_processes(2)

        with patch.object(alfworld_envs_module.ray, "get", side_effect=lambda values: values):
            text_obs, _, _ = envs.reset()
            next_obs, _, _, _, _ = envs.step(["look", "look"])
            restart_obs, _, _ = envs.restart()

        self.assertEqual(text_obs, ["obs-0", "obs-1"])
        self.assertEqual(next_obs, ["next-0", "next-1"])
        self.assertEqual(restart_obs, ["restart-0", "restart-1"])
        self.assertEqual([worker.reset.calls for worker in envs.workers], [1, 1, 0, 0])
        self.assertEqual([worker.restart.calls for worker in envs.workers], [1, 1, 0, 0])
        self.assertEqual([worker.step.calls for worker in envs.workers], [1, 1, 0, 0])
        self.assertEqual(len(envs.get_admissible_commands), 2)

        envs.set_active_num_processes(4)
        with self.assertRaises(ValueError):
            envs.set_active_num_processes(5)

    def test_alfworld_worker_restart_resets_current_batch_game(self):
        batch_env = Mock()
        batch_env.reset.return_value = (
            ["initial observation"],
            {
                "admissible_commands": [["look"]],
                "extra.gamefile": ["same-game.tw-pddl"],
            },
        )
        outer_env = SimpleNamespace(
            batch_env=batch_env,
            last_commands=["old command"],
            obs=["terminal observation"],
        )
        worker = AlfworldWorker.__new__(AlfworldWorker)
        worker.env = outer_env

        obs, infos = worker.restart()

        batch_env.reset.assert_called_once_with()
        self.assertEqual(outer_env.last_commands, [None])
        self.assertEqual(obs, ["initial observation"])
        self.assertEqual(infos["extra.gamefile"], ["same-game.tw-pddl"])
        self.assertEqual(infos["observation_text"], ["initial observation"])


if __name__ == "__main__":
    import unittest
    unittest.main()
