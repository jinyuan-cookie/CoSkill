import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _retrieval_steps(task_name, representative):
    return [
        {
            "attempt_idx": attempt_idx,
            "step": 0,
            "query_text": f"{task_name}\n\nCurrent observation: a long observation that is not persisted",
            "task_skills": [{
                "skill_id": f"task-{task_name}",
                "title": f"task skill {task_name}",
                "similarity": 0.9,
                "input_to_retrieval": "large repeated task-skill text",
            }],
            "step_skills": [{
                "skill_id": f"step-{representative}-{attempt_idx}",
                "title": f"step skill {representative}/{attempt_idx}",
                "similarity": 0.8,
                "input_to_retrieval": "large repeated step-skill text",
            }],
        }
        for attempt_idx in range(3)
    ]


class RetrievedSkillsRecordingTest(unittest.TestCase):
    def test_promotion_audit_is_reduced_to_counts_by_rejection_reason(self):
        summary = RayPPOTrainer._summarize_promotion_audit({
            "accepted": [
                {"traj_uid": "accepted-1", "edited_success_rate": 0.5, "improvement": 0.5},
                {"traj_uid": "accepted-2", "edited_success_rate": 1.0, "improvement": 1.0},
            ],
            "rejected": [
                {"traj_uid": "r1", "reason": "no_success_improvement"},
                {"traj_uid": "r2", "reason": "outside_group_top_k"},
                {"traj_uid": "r3", "reason": "no_success_improvement"},
            ],
        }, step=7)
        self.assertEqual(summary, {
            "step": 7,
            "accepted_count": 2,
            "accepted": [
                {"edited_success_rate": 0.5, "improvement": 0.5},
                {"edited_success_rate": 1.0, "improvement": 1.0},
            ],
            "rejected_count": 3,
            "total_candidates": 5,
            "rejected_by_reason": {
                "no_success_improvement": 2,
                "outside_group_top_k": 1,
            },
        })

    def test_validation_retrieval_log_is_disabled(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.config = SimpleNamespace(
                env={"skills_only_memory": {"record_retrieved_skills": True}},
                trainer={"default_local_dir": output_dir},
            )
            trainer._record_retrieved_skills(
                step=0,
                phase="validation",
                memories_list=[[{"query_text": "task", "task_skills": [], "step_skills": []}]],
            )
            self.assertFalse(
                (Path(output_dir) / "retrieved_skills_validation_step0.json").exists()
            )

    def test_training_log_keeps_one_representative_per_task_and_groups_attempts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.config = SimpleNamespace(
                env={"skills_only_memory": {"record_retrieved_skills": True, "skill_gen_mode": "task_step"}},
                trainer={"default_local_dir": output_dir},
            )
            memories = [
                {"query_text": "task A", "task_skills": [], "step_skills": []},
                {"query_text": "task A", "task_skills": [], "step_skills": []},
                {"query_text": "task B", "task_skills": [], "step_skills": []},
                {"query_text": "task B", "task_skills": [], "step_skills": []},
            ]
            per_step = np.empty(4, dtype=object)
            per_step[0] = _retrieval_steps("task A", "representative-A")
            per_step[1] = _retrieval_steps("task A", "discarded-A")
            per_step[2] = _retrieval_steps("task B", "representative-B")
            per_step[3] = _retrieval_steps("task B", "discarded-B")

            trainer._record_retrieved_skills(
                step=1,
                phase="train",
                memories_list=[memories],
                per_step_retrievals_list=[per_step],
                traj_index_for_record=[np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)],
                task_uid_for_record=[np.asarray([
                    "uid-A", "uid-A", "uid-A", "uid-A",
                    "uid-B", "uid-B", "uid-B", "uid-B",
                ], dtype=object)],
            )

            output = json.loads(
                (Path(output_dir) / "retrieved_skills_train_step1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(output["num_tasks"], 2)
            tasks = output["retrievals"][0]["tasks"]
            self.assertEqual([task["task_uid"] for task in tasks], ["uid-A", "uid-B"])
            self.assertEqual([task["num_rollouts_collapsed"] for task in tasks], [2, 2])
            self.assertEqual([attempt["attempt_idx"] for attempt in tasks[0]["attempts"]], [0, 1, 2])
            self.assertEqual(tasks[0]["task_query_text"], "task A")
            self.assertEqual(
                tasks[0]["attempts"][0]["step_retrievals"][0]["step_skills"][0]["skill_id"],
                "step-representative-A-0",
            )
            compact_step = tasks[0]["attempts"][0]["step_retrievals"][0]
            self.assertNotIn("query_text", compact_step)
            self.assertNotIn("input_to_retrieval", compact_step["step_skills"][0])

    def test_skill_agent_log_keeps_one_compact_representative_per_task(self):
        def trajectory(task_uid, traj_uid, marker):
            return [{
                "step": 0,
                "uid": task_uid,
                "traj_uid": traj_uid,
                "anchor_obs": f"observation-{marker}",
                "state": "large state that must not be persisted",
                "next_state": "large next state that must not be persisted",
                "reasoning_history": ["large history that must not be persisted"],
                "skill_agent_prompt": "large prompt that must not be persisted",
                "skill_agent_response": "large raw response that must not be persisted",
                "reasoning_action": "open door",
                "environment_feedback": {"reward": 1.0},
                "active_step_skills": [{
                    "skill_id": "step-1",
                    "title": "Open the door",
                    "principle": "large principle that must not be persisted",
                }],
                "decision": {"action": "KEEP", "parse_ok": True},
                "attempt_rewards": [0.0, 1.0, 1.0],
                "attempt_successes": [0.0, 1.0, 1.0],
                "skill_baseline_episode_reward": 0.0,
                "skill_edited_episode_reward_mean": 1.0,
                "skill_episode_improvement_reward": 1.0,
                "meta_attempt_overlay": {
                    "initial": {"task_skills": [], "step_skills": []},
                    "final": {"task_skills": [], "step_skills": []},
                    "patches": [],
                    "applied_edits": [],
                    "rejected_edits": [],
                },
            }]

        with tempfile.TemporaryDirectory() as output_dir:
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.config = SimpleNamespace(
                env={"skill_agent": {"record_trajectories": True}},
                trainer={"default_local_dir": output_dir},
            )
            trainer._record_skill_agent_trajectories(
                step=1,
                phase="train",
                trajectory_batches=[[
                    trajectory("uid-A", "traj-A1", "representative-A"),
                    trajectory("uid-A", "traj-A2", "discarded-A"),
                    trajectory("uid-B", "traj-B1", "representative-B"),
                ]],
            )

            output = json.loads(
                (Path(output_dir) / "skill_agent_trajectories_train_step1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(output["num_tasks"], 2)
            self.assertEqual([record["num_rollouts_collapsed"] for record in output["tasks"]], [2, 1])
            representative = output["tasks"][0]
            self.assertEqual(representative["representative_traj_uid"], "traj-A1")
            self.assertEqual(representative["skill_baseline_episode_reward"], 0.0)
            self.assertEqual(representative["skill_edited_episode_reward_mean"], 1.0)
            self.assertEqual(representative["skill_episode_improvement_reward"], 1.0)
            edit_step = representative["edit_steps"][0]
            self.assertEqual(edit_step["anchor_obs"], "observation-representative-A")
            for omitted in (
                "state", "next_state", "reasoning_history",
                "skill_agent_prompt", "skill_agent_response",
            ):
                self.assertNotIn(omitted, edit_step)
            self.assertNotIn("principle", edit_step["active_step_skills"][0])


if __name__ == "__main__":
    unittest.main()
