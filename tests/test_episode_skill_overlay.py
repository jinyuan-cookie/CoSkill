import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _load_overlay_classes():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("agent_system")
    package.__path__ = [str(root / "agent_system")]
    memory_package = types.ModuleType("agent_system.memory")
    memory_package.__path__ = [str(root / "agent_system" / "memory")]
    sys.modules.setdefault("agent_system", package)
    sys.modules.setdefault("agent_system.memory", memory_package)
    for name, path in (
        ("agent_system.memory.base", root / "agent_system" / "memory" / "base.py"),
        ("agent_system.memory.skills_only_memory", root / "agent_system" / "memory" / "skills_only_memory.py"),
        ("agent_system.memory.episode_skill_overlay", root / "agent_system" / "memory" / "episode_skill_overlay.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    memory_module = sys.modules["agent_system.memory.skills_only_memory"]
    overlay_module = sys.modules["agent_system.memory.episode_skill_overlay"]
    return memory_module.SkillsOnlyMemory, overlay_module.EpisodeSkillOverlay


SkillsOnlyMemory, EpisodeSkillOverlay = _load_overlay_classes()


def _decision(action, target=None, skill=None, parent=None, parse_ok=True):
    return {
        "decision": {
            "action": action,
            "target_skill_id": target,
            "parent_task_skill_id": parent,
            "skill": skill,
            "parse_ok": parse_ok,
        }
    }


class EpisodeSkillOverlayTest(unittest.TestCase):
    def setUp(self):
        self.memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")
        self.memory.add_hierarchical_skill_pairs([
            {
                "task_skill": {"title": "Kitchen", "principle": "k", "retrieval_obs": "kitchen"},
                "step_skill": {"title": "Use sink", "principle": "old sink", "retrieval_obs": "sink visible"},
            },
            {
                "task_skill": {"title": "Bedroom", "principle": "b", "retrieval_obs": "bedroom"},
                "step_skill": {"title": "Use bed", "principle": "bed", "retrieval_obs": "bed visible"},
            },
        ])
        self.kitchen_task = self.memory.skills["task_skills"][0]
        self.kitchen_step = self.memory.skills["step_skills"][0]
        self.overlay = EpisodeSkillOverlay(
            self.memory,
            {"task_skills": [self.kitchen_task], "step_skills": []},
        )

    def test_clone_is_private_and_scoped_to_selected_task(self):
        self.assertEqual([s["title"] for s in self.overlay.memory.skills["task_skills"]], ["Kitchen"])
        self.assertEqual([s["title"] for s in self.overlay.memory.skills["step_skills"]], ["Use sink"])

        self.overlay.memory.skills["step_skills"][0]["principle"] = "overlay only"
        self.assertEqual(self.memory.skills["step_skills"][0]["principle"], "old sink")

    def test_overlays_share_supplied_encoder_but_not_skill_state(self):
        shared_encoder = object()
        shared_child_cache = {}
        other_overlay = EpisodeSkillOverlay(
            self.memory,
            {"task_skills": [self.kitchen_task], "step_skills": []},
            shared_embedding_model=shared_encoder,
            shared_child_embedding_cache=shared_child_cache,
        )
        self.overlay = EpisodeSkillOverlay(
            self.memory,
            {"task_skills": [self.kitchen_task], "step_skills": []},
            shared_embedding_model=shared_encoder,
            shared_child_embedding_cache=shared_child_cache,
        )

        self.assertIs(self.overlay.memory._embedding_model, shared_encoder)
        self.assertIs(other_overlay.memory._embedding_model, shared_encoder)
        self.assertIs(self.overlay._shared_child_embedding_cache, shared_child_cache)
        self.assertIs(other_overlay._shared_child_embedding_cache, shared_child_cache)
        self.assertIsNot(self.overlay.memory.skills, other_overlay.memory.skills)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is required for embedding retrieval")
    def test_embedding_retrieval_batches_queries_and_reuses_bundle_cache(self):
        class FakeEncoder:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **_kwargs):
                self.calls.append(list(texts))

                def vector(text):
                    lowered = str(text).lower()
                    if "sink" in lowered:
                        return [1.0, 0.0]
                    if "fridge" in lowered:
                        return [0.0, 1.0]
                    return [0.70710678, 0.70710678]

                return [vector(text) for text in texts]

        memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="embedding")
        memory.skills = {
            "task_skills": [{
                "skill_id": "task-1",
                "title": "Kitchen",
                "step_skill_ids": ["step-sink", "step-fridge"],
            }],
            "step_skills": [
                {
                    "skill_id": "step-sink",
                    "parent_task_skill_id": "task-1",
                    "title": "Use sink",
                    "retrieval_obs": "sink visible",
                },
                {
                    "skill_id": "step-fridge",
                    "parent_task_skill_id": "task-1",
                    "title": "Use fridge",
                    "retrieval_obs": "fridge visible",
                },
            ],
        }
        encoder = FakeEncoder()
        shared_cache = {}
        overlays = [
            EpisodeSkillOverlay(
                memory,
                {"task_skills": [memory.skills["task_skills"][0]], "step_skills": []},
                shared_embedding_model=encoder,
                shared_child_embedding_cache=shared_cache,
            )
            for _ in range(2)
        ]

        first = EpisodeSkillOverlay.retrieve_many(
            overlays,
            ["Current observation: sink visible", "Current observation: fridge visible"],
            top_k_step=1,
        )
        self.assertEqual(len(encoder.calls), 1)
        # Two observation queries plus one copy of the shared two-skill bundle.
        self.assertEqual(len(encoder.calls[0]), 4)
        self.assertEqual(len(shared_cache), 1)
        self.assertEqual(first[0]["step_skills"][0]["skill_id"], "step-sink")
        self.assertEqual(first[1]["step_skills"][0]["skill_id"], "step-fridge")

        EpisodeSkillOverlay.retrieve_many(
            overlays,
            ["Current observation: sink visible", "Current observation: fridge visible"],
            top_k_step=1,
        )
        self.assertEqual(len(encoder.calls), 2)
        # The second call encodes only the two observations; child embeddings
        # come from the task-bundle cache.
        self.assertEqual(len(encoder.calls[1]), 2)

        overlays[0].apply({
            "step": 0,
            "decision": {
                "parse_ok": True,
                "action": "UPDATE",
                "target_skill_id": "step-sink",
                "skill": {"retrieval_obs": "updated sink visible"},
            },
        })
        EpisodeSkillOverlay.retrieve_many(
            overlays,
            ["Current observation: sink visible", "Current observation: fridge visible"],
            top_k_step=1,
        )
        self.assertEqual(len(encoder.calls), 3)
        # Only the edited overlay has a new content key, so its child bundle is
        # encoded once while the unchanged overlay keeps using the old cache.
        self.assertEqual(len(encoder.calls[2]), 4)
        self.assertEqual(len(shared_cache), 2)

    def test_ordered_insert_update_delete_keep_and_invalid_are_local(self):
        task_id = self.kitchen_task["skill_id"]
        step_id = self.kitchen_step["skill_id"]
        self.overlay.apply_all([
            _decision("UPDATE", step_id, {"principle": "updated sink"}),
            _decision("INSERT", parent=task_id, skill={"title": "Turn tap", "principle": "turn it", "retrieval_obs": "tap visible"}),
            _decision("DELETE", step_id),
            _decision("KEEP", target="overlay_step_001"),
            _decision("KEEP", parse_ok=False),
        ])

        overlay_steps = self.overlay.memory.skills["step_skills"]
        self.assertEqual([s["title"] for s in overlay_steps], ["Turn tap"])
        self.assertEqual(self.overlay.memory.skills["task_skills"][0]["step_skill_ids"], ["overlay_step_001"])
        self.assertEqual(self.memory.skills["step_skills"][0]["principle"], "old sink")
        self.assertEqual(len(self.overlay.applied_edits), 4)
        self.assertEqual(self.overlay.rejected_edits[0]["reason"], "invalid_json")

        retrieved = self.overlay.retrieve("Current observation: tap visible", top_k_step=2)
        self.assertTrue(retrieved["overlay"])
        self.assertEqual([s["title"] for s in retrieved["step_skills"]], ["Turn tap"])

    def test_invalid_target_and_insert_parent_do_not_change_overlay(self):
        before = self.overlay.snapshot()["final"]
        self.assertFalse(self.overlay.apply(_decision("UPDATE", "not-in-overlay", {"principle": "bad"})))
        self.assertFalse(self.overlay.apply(_decision("INSERT", parent="not-in-overlay", skill={"title": "bad"})))
        self.assertEqual(self.overlay.snapshot()["final"], before)
        self.assertEqual(
            [entry["reason"] for entry in self.overlay.rejected_edits],
            ["update_target_not_in_overlay", "insert_parent_not_in_overlay"],
        )

    def test_update_with_non_object_skill_payload_is_rejected(self):
        before = self.overlay.snapshot()["final"]

        applied = self.overlay.apply(
            _decision("UPDATE", self.kitchen_step["skill_id"], "replace the old skill")
        )

        self.assertFalse(applied)
        self.assertEqual(self.overlay.snapshot()["final"], before)
        self.assertEqual(
            self.overlay.rejected_edits[-1]["reason"],
            "update_missing_skill_content",
        )

    def test_non_string_nested_skill_fields_are_rejected(self):
        before = self.overlay.snapshot()["final"]
        task_id = self.kitchen_task["skill_id"]
        step_id = self.kitchen_step["skill_id"]

        update_applied = self.overlay.apply(
            _decision("UPDATE", step_id, {"retrieval_obs": ["sink visible"]})
        )
        insert_applied = self.overlay.apply(
            _decision(
                "INSERT",
                parent=task_id,
                skill={"title": "Turn tap", "principle": {"text": "turn it"}},
            )
        )

        self.assertFalse(update_applied)
        self.assertFalse(insert_applied)
        self.assertEqual(self.overlay.snapshot()["final"], before)
        self.assertEqual(
            [entry["reason"] for entry in self.overlay.rejected_edits[-2:]],
            ["update_invalid_skill_field_type", "insert_invalid_skill_field_type"],
        )

    def test_skill_to_text_ignores_malformed_legacy_fields(self):
        skill = {
            "title": "Use sink",
            "principle": {"text": "invalid"},
            "when_to_apply": 123,
            "retrieval_obs": ["sink visible"],
        }

        self.assertEqual(self.memory._skill_to_text(skill), "Use sink")
        self.assertEqual(self.memory._skill_to_text(skill, mode="principle"), "")
        self.assertEqual(self.memory._skill_to_text(["not", "a", "skill"]), "")


if __name__ == "__main__":
    unittest.main()
