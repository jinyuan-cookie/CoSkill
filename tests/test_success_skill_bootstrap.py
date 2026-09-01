import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _load_skill_updater():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("agent_system")
    package.__path__ = [str(root / "agent_system")]
    memory_package = types.ModuleType("agent_system.memory")
    memory_package.__path__ = [str(root / "agent_system" / "memory")]
    sys.modules.setdefault("agent_system", package)
    sys.modules.setdefault("agent_system.memory", memory_package)
    openai_module = types.ModuleType("openai")
    openai_module.AzureOpenAI = type("AzureOpenAI", (), {})
    openai_module.OpenAI = type("OpenAI", (), {})
    sys.modules.setdefault("openai", openai_module)
    spec = importlib.util.spec_from_file_location(
        "agent_system.memory.skill_updater",
        root / "agent_system" / "memory" / "skill_updater.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_system.memory.skill_updater"] = module
    spec.loader.exec_module(module)
    return module.SkillUpdater


SkillUpdater = _load_skill_updater()


def _updater_without_client(max_steps=8):
    updater = SkillUpdater.__new__(SkillUpdater)
    updater.success_max_step_skills = max_steps
    updater.api_type = "skill_batch"
    updater.model = "test-reflector"
    updater.update_history = []
    updater.max_concurrent = 1
    return updater


class SuccessSkillBootstrapTest(unittest.TestCase):
    def test_caller_provided_chat_client_is_used_for_reflection(self):
        client = object()
        updater = SkillUpdater(chat_client=client, skill_llm_model="custom-model")
        self.assertIs(updater.client, client)
        self.assertEqual(updater.model, "custom-model")
        self.assertEqual(updater.api_type, "custom_chat_client")

    def test_task_group_analysis_resolves_any_trajectory_index_and_turn(self):
        updater = _updater_without_client(max_steps=2)
        updater._reflect_batch_http = lambda prompts: ["""SKILL_BUNDLE:
        {
          "task_skill": {"title": "Use evidence", "principle": "Follow the successful route.", "when_to_apply": "relocation"},
          "step_skills": [{
            "source_trajectory_index": 3,
            "source_turn": 1,
            "title": "Use visible target",
            "principle": "Take the visible object.",
            "when_to_apply": "the target is visible"
          }]
        }"""]
        trajectory = lambda observation, succeeded: {
            "task": "Your task is to: put the apple in the fridge.",
            "succeeded": succeeded,
            "refined_trajectory": {
                "task": "put the apple in the fridge",
                "turns": [{"observation": observation, "action": "take apple"}],
            },
        }
        groups = [{
            "group_uid": "g",
            "group_success_rate": 0.5,
            "trajectories": [
                trajectory("first", True),
                trajectory("second", True),
                trajectory("failed", False),
            ],
        }]

        pairs, metadata = updater.analyze_task_groups(groups, {}, return_metadata=True)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["step_skill"]["retrieval_obs"], "failed")
        self.assertEqual(pairs[0]["step_skill"]["source_outcome"], "failure")
        self.assertEqual(pairs[0]["task_skill"]["utility"], 0.5)
        self.assertEqual(metadata["mode"], "task_group_initialization")

    def test_task_group_prompt_contains_every_outcome_labeled_trajectory(self):
        updater = _updater_without_client(max_steps=2)
        make_trajectory = lambda observation, succeeded: {
            "task": "put the apple in the fridge",
            "succeeded": succeeded,
            "refined_trajectory": {
                "task": "put the apple in the fridge",
                "turns": [{"observation": observation, "action": "look"}],
            },
        }
        prompt = updater._build_task_group_skill_prompt({
            "trajectories": [
                make_trajectory("success observation", True),
                make_trajectory("failure observation", False),
            ]
        })

        self.assertIn("TRAJECTORY 1 [SUCCESS]", prompt)
        self.assertIn("TRAJECTORY 2 [FAILURE]", prompt)
        self.assertIn("success observation", prompt)
        self.assertIn("failure observation", prompt)

    def test_success_bundle_parser_accepts_task_and_multiple_steps(self):
        updater = _updater_without_client()
        task, steps = updater._parse_success_skill_response(
        """SKILL_BUNDLE:
        {
          "task_skill": {
            "title": "Move an object safely",
            "principle": "Locate, pick up, and place the requested object.",
            "when_to_apply": "Object relocation tasks"
          },
          "step_skills": [
            {
              "source_turn": 1,
              "title": "Inspect likely container",
              "principle": "Open the likely container before searching elsewhere.",
              "when_to_apply": "The target location is unknown"
            },
            {
              "source_turn": 2,
              "title": "Pick up target",
              "principle": "Take the visible target before navigating away.",
              "when_to_apply": "The target is visible and inventory is empty"
            }
          ]
        }"""
    )

        self.assertEqual(task["title"], "Move an object safely")
        self.assertEqual([step["source_turn"] for step in steps], [1, 2])


    def test_success_analysis_uses_real_turn_observations_and_builds_pairs(self):
        updater = _updater_without_client(max_steps=2)
        raw = """SKILL_BUNDLE:
    {
      "task_skill": {
        "title": "Complete pick and place",
        "principle": "Find the object and put it in the requested receptacle.",
        "when_to_apply": "Pick-and-place tasks"
      },
      "step_skills": [
        {
          "source_turn": 1,
          "title": "Open the cabinet",
          "principle": "Open the closed cabinet to reveal its contents.",
          "when_to_apply": "A closed cabinet may contain the target"
        },
        {
          "source_turn": 2,
          "title": "Take the apple",
          "principle": "Pick up the apple once it is visible.",
          "when_to_apply": "The apple is visible"
        },
        {
          "source_turn": 3,
          "title": "Ignored by cap",
          "principle": "This skill exceeds the configured cap.",
          "when_to_apply": "Never"
        }
      ]
    }"""
        updater._reflect_batch_endpoint = lambda: "http://reflection/reflect_batch"
        updater._reflect_batch_http = lambda prompts: [raw]
        trajectory = {
        "task": "Your task is to: put the apple in the fridge.",
        "task_type": "pick_and_place",
        "refined_trajectory": {
            "task": "put the apple in the fridge",
            "turns": [
                {"observation": "Cabinet 1 is closed.", "action": "open cabinet 1"},
                {"observation": "An apple is in cabinet 1.", "action": "take apple"},
                {"observation": "You carry an apple.", "action": "go to fridge"},
            ],
        },
        }

        pairs, metadata = updater.analyze_successes(
            [trajectory], current_skills={}, return_metadata=True
        )

        self.assertEqual(len(pairs), 2)
        self.assertIs(pairs[0]["task_skill"], pairs[1]["task_skill"])
        self.assertEqual(pairs[0]["task_skill"]["step_skill_ids"], [])
        self.assertEqual(
            [pair["step_skill"]["retrieval_obs"] for pair in pairs],
            ["Cabinet 1 is closed.", "An apple is in cabinet 1."],
        )
        self.assertEqual(metadata["bundle_sizes"], [2])


    def test_success_bundle_without_valid_steps_is_not_emitted(self):
        updater = _updater_without_client()
        raw = (
            'SKILL_BUNDLE: {"task_skill": {"title": "x", "principle": "y", '
            '"when_to_apply": "z"}, "step_skills": []}'
        )
        updater._reflect_batch_endpoint = lambda: "http://reflection/reflect_batch"
        updater._reflect_batch_http = lambda prompts: [raw]
        trajectory = {
            "task": "Your task is to: put the apple in the fridge.",
            "refined_trajectory": {
                "task": "put the apple in the fridge",
                "turns": [{"observation": "Apple visible.", "action": "take apple"}],
            },
        }

        pairs, metadata = updater.analyze_successes(
            [trajectory], current_skills={}, return_metadata=True
        )

        self.assertEqual(pairs, [])
        self.assertEqual(metadata["bundle_sizes"], [0])


if __name__ == "__main__":
    unittest.main()
