import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_modules():
    package = types.ModuleType("agent_system")
    package.__path__ = [str(ROOT / "agent_system")]
    memory_package = types.ModuleType("agent_system.memory")
    memory_package.__path__ = [str(ROOT / "agent_system" / "memory")]
    sys.modules.setdefault("agent_system", package)
    sys.modules.setdefault("agent_system.memory", memory_package)
    for name in ("base", "skill_bank_lifecycle", "skills_only_memory"):
        full_name = f"agent_system.memory.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name, ROOT / "agent_system" / "memory" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return (
        sys.modules["agent_system.memory.skill_bank_lifecycle"],
        sys.modules["agent_system.memory.skills_only_memory"].SkillsOnlyMemory,
    )


lifecycle, SkillsOnlyMemory = _load_modules()


class SkillBankLifecycleTest(unittest.TestCase):
    def test_saved_hierarchical_bank_loads_as_training_input(self):
        memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")
        memory.add_hierarchical_skill_pairs([{
            "task_skill": {
                "title": "Task", "principle": "plan", "when_to_apply": "task",
                "retrieval_obs": "task",
            },
            "step_skill": {
                "title": "Step", "principle": "act", "when_to_apply": "state",
                "retrieval_obs": "state",
            },
        }], created_at_step=0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "initial_skill_bank.json")
            memory.save_skills(path)
            loaded = SkillsOnlyMemory(
                skills_json_path=path,
                load_initial_skills=True,
                retrieval_mode="template",
            )
        self.assertEqual(loaded.skills, memory.skills)
        task = loaded.skills["task_skills"][0]
        step = loaded.skills["step_skills"][0]
        self.assertEqual(task["step_skill_ids"], [step["skill_id"]])
        self.assertEqual(step["parent_task_skill_id"], task["skill_id"])

    def _sidecar(self, uid, traj, successes, rewards, principle, action="UPDATE"):
        return [{
            "uid": uid,
            "traj_uid": traj,
            "attempt_successes": successes,
            "attempt_rewards": rewards,
            "meta_attempt_overlay": {
                "final": {
                    "task_skills": [{
                        "skill_id": "task_001", "title": "Task", "principle": "task",
                        "when_to_apply": "task", "retrieval_obs": "task", "step_skill_ids": ["step_001"],
                    }],
                    "step_skills": [{
                        "skill_id": "step_001", "title": "Step", "principle": principle,
                        "when_to_apply": "state", "retrieval_obs": "state",
                        "parent_task_skill_id": "task_001",
                    }],
                },
                "patches": [{"applied": True, "effect": {"action": action}}],
            },
        }]

    def test_promotion_requires_improvement_and_keeps_top_two(self):
        sidecars = [
            self._sidecar("g", "a", [0, 1, 1], [0, 4, 5], "best"),
            self._sidecar("g", "b", [0, 1, 0], [0, 3, 3], "second"),
            self._sidecar("g", "c", [0, 1, 0], [0, 2, 2], "third"),
            self._sidecar("g", "d", [1, 1, 1], [5, 5, 5], "unchanged"),
            self._sidecar("h", "e", [0, 1, 0], [0, 1, 1], "other-group"),
        ]
        accepted, rejected = lifecycle.select_promotion_candidates(sidecars, max_per_group=2)
        self.assertEqual([item["traj_uid"] for item in accepted], ["a", "b", "e"])
        reasons = {(item["traj_uid"], item["reason"]) for item in rejected}
        self.assertIn(("c", "outside_group_top_k"), reasons)
        self.assertIn(("d", "no_success_improvement"), reasons)

    def test_two_attempt_promotion_uses_the_only_edited_evaluation(self):
        sidecars = [
            self._sidecar("g", "improved", [0, 1], [0, 4], "better"),
            self._sidecar("g", "unchanged", [1, 1], [4, 4], "same"),
        ]

        accepted, rejected = lifecycle.select_promotion_candidates(sidecars)

        self.assertEqual([item["traj_uid"] for item in accepted], ["improved"])
        self.assertEqual(accepted[0]["attempt_successes"], [0.0, 1.0])
        self.assertEqual(accepted[0]["attempt_rewards"], [0.0, 4.0])
        self.assertEqual(accepted[0]["edited_success_rate"], 1.0)
        self.assertEqual(accepted[0]["improvement"], 1.0)
        self.assertIn(
            ("unchanged", "no_success_improvement"),
            {(item["traj_uid"], item["reason"]) for item in rejected},
        )

    def test_promotion_clones_ids_and_duplicate_is_rejected(self):
        memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")
        memory.add_hierarchical_skill_pairs([{
            "task_skill": {"title": "Task", "principle": "task", "when_to_apply": "task", "retrieval_obs": "task"},
            "step_skill": {"title": "Old", "principle": "old", "when_to_apply": "state", "retrieval_obs": "state"},
        }])
        source_task = memory.skills["task_skills"][0]
        candidate = {
            "task_skill": dict(source_task),
            "step_skills": [{"title": "New", "principle": "new", "when_to_apply": "state", "retrieval_obs": "state"}],
            "traj_uid": "t", "group_uid": "g", "attempt_successes": [0, 1, 1],
            "attempt_rewards": [0, 1, 1], "edited_success_rate": 1.0, "improvement": 1.0,
        }
        candidate["bundle_fingerprint"] = lifecycle.bundle_fingerprint(
            candidate["task_skill"], candidate["step_skills"]
        )
        first = memory.promote_hierarchical_bundle(candidate, created_at_step=5)
        second = memory.promote_hierarchical_bundle(candidate, created_at_step=6)
        self.assertTrue(first["added"])
        self.assertFalse(second["added"])
        self.assertEqual(second["reason"], "duplicate_bundle")
        self.assertEqual(len(memory.skills["task_skills"]), 2)
        self.assertNotEqual(memory.skills["task_skills"][0]["skill_id"], first["task_skill_id"])

    def test_bundle_eviction_uses_frequency_then_recency_and_cascades(self):
        memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")
        for title in ("A", "B", "C"):
            memory.add_hierarchical_skill_pairs([{
                "task_skill": {"title": title, "principle": title, "retrieval_obs": title},
                "step_skill": {"title": f"{title}-step", "principle": title, "retrieval_obs": title},
            }], created_at_step=0)
        memory.skills["task_skills"][0].update(retrieval_count=0, last_retrieval_step=3)
        memory.skills["task_skills"][1].update(retrieval_count=0, last_retrieval_step=2)
        memory.skills["task_skills"][2].update(retrieval_count=1, last_retrieval_step=1)
        result = memory.evict_task_bundles_frequency_recency(20, 2, protect_recent_steps=0)
        self.assertEqual(result["removed"][0]["title"], "B")
        self.assertEqual(len(memory.skills["task_skills"]), 2)
        self.assertEqual(len(memory.skills["step_skills"]), 2)

    def test_initialization_retrieval_switch_hides_partial_bank(self):
        memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")
        memory.add_hierarchical_skill_pairs([{
            "task_skill": {"title": "Task", "principle": "task", "retrieval_obs": "task"},
            "step_skill": {"title": "Step", "principle": "step", "retrieval_obs": "state"},
        }])
        memory._retrieval_disabled = True
        self.assertEqual(memory.retrieve_task_skills_batch(["task"], top_k=1)[0]["task_skills"], [])
        task_id = memory.skills["task_skills"][0]["skill_id"]
        self.assertEqual(
            memory.retrieve_child_step_skills_batch([[task_id]], ["state"], top_k=1)[0]["step_skills"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
