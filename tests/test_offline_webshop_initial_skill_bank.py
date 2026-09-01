import json
import tempfile
import unittest
from pathlib import Path

from initial_skill_bank.generate_alfworld_initial_skill_bank import GenerationConfig
from initial_skill_bank.generate_webshop_initial_skill_bank import (
    WebShopInitialSkillBankGenerator,
    _available_actions,
    _build_parser,
    _webshop_data_paths,
)


class _FakeWebShopEnvs:
    def __init__(self):
        self._last_goal_indices = None

    @staticmethod
    def _info():
        return {
            "available_actions": {
                "has_search_bar": True,
                "clickables": ["back to search"],
            },
            "won": False,
            "task_score": 0.0,
        }

    def reset(self):
        self._last_goal_indices = [501, 501, 777, 777]
        observations = [
            "WebShop [SEP] Instruction: [SEP] buy a red cotton shirt [SEP] Search",
            "WebShop [SEP] Instruction: [SEP] buy a red cotton shirt [SEP] Search",
            "WebShop [SEP] Instruction: [SEP] buy a waterproof hiking bag [SEP] Search",
            "WebShop [SEP] Instruction: [SEP] buy a waterproof hiking bag [SEP] Search",
        ]
        return observations, [self._info() for _ in observations]

    def step(self, actions):
        won = [action == "search[correct]" for action in actions]
        infos = []
        for succeeded in won:
            info = self._info()
            info.update(won=succeeded, task_score=1.0 if succeeded else 0.0)
            infos.append(info)
        return (
            ["WebShop [SEP] done"] * len(actions),
            [10.0 if succeeded else 0.0 for succeeded in won],
            [True] * len(actions),
            infos,
        )


class _AlternatingPolicy:
    def __init__(self):
        self.prompts = []

    def complete_batch(self, prompts):
        self.prompts.extend(prompts)
        return [
            "<think>search precisely</think><action>search[correct]</action>"
            if index % 2 == 0
            else "<think>search broadly</think><action>search[wrong]</action>"
            for index in range(len(prompts))
        ]


class _FakeUpdater:
    def __init__(self):
        self.groups = []

    def analyze_task_groups(self, groups, current_skills, return_metadata):
        del current_skills
        self.groups.extend(groups)
        pairs = []
        for group in groups:
            trajectory = group["trajectories"][0]
            pairs.append({
                "task_skill": {
                    "title": "Plan the purchase",
                    "principle": "Translate all requested attributes into a precise search.",
                    "when_to_apply": trajectory["task"],
                    "retrieval_obs": trajectory["task"],
                    "initialization_group_uid": group["group_uid"],
                    "utility": group["group_success_rate"],
                },
                "step_skill": {
                    "title": "Search with constraints",
                    "principle": "Include product type and requested attributes.",
                    "when_to_apply": "the initial search page",
                    "retrieval_obs": trajectory["refined_trajectory"]["turns"][0]["observation"],
                    "source_trajectory_index": 1,
                    "source_turn": 1,
                },
            })
        metadata = {"mode": "fake-webshop", "num_task_groups": len(groups)}
        return (pairs, metadata) if return_metadata else pairs


class _FakeMemory:
    def __init__(self):
        self.skills = {"task_skills": [], "step_skills": []}

    def add_hierarchical_skill_pairs(self, pairs, created_at_step):
        del created_at_step
        added_task = added_step = 0
        by_group = {
            task["initialization_group_uid"]: task
            for task in self.skills["task_skills"]
        }
        for pair in pairs:
            task = by_group.get(pair["task_skill"]["initialization_group_uid"])
            if task is None:
                task = dict(pair["task_skill"])
                task.update(
                    skill_id=f"task_skill_{len(self.skills['task_skills']) + 1}",
                    step_skill_ids=[],
                )
                self.skills["task_skills"].append(task)
                by_group[task["initialization_group_uid"]] = task
                added_task += 1
            step = dict(pair["step_skill"])
            step.update(
                skill_id=f"step_skill_{len(self.skills['step_skills']) + 1}",
                parent_task_skill_id=task["skill_id"],
            )
            self.skills["step_skills"].append(step)
            task["step_skill_ids"].append(step["skill_id"])
            added_step += 1
        return {"task_skills": added_task, "step_skills": added_step}

    def save_skills(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.skills, handle, indent=2)


class OfflineWebShopInitialSkillBankTest(unittest.TestCase):
    def test_cli_defaults_match_webshop_collection(self):
        args = _build_parser().parse_args([
            "--api-base-url", "https://gateway.example/chatCompletion",
            "--model", "test-model",
        ])

        self.assertEqual(args.dataset_name, "webshop")
        self.assertEqual(args.task_groups_per_batch, 1)
        self.assertEqual(args.rollouts_per_task, 8)
        self.assertEqual(args.max_steps, 15)
        self.assertEqual(args.api_retries, 10)
        self.assertFalse(args.use_full)
        items, attrs = _webshop_data_paths(args)
        self.assertEqual(items.name, "items_shuffle_1000.json")
        self.assertEqual(attrs.name, "items_ins_v2_1000.json")

    def test_available_actions_preserve_webshop_syntax(self):
        actions = _available_actions({
            "available_actions": {
                "has_search_bar": True,
                "clickables": ["item 1", "back to search"],
            }
        })

        self.assertEqual(
            actions,
            ["search[<your query>]", "click[item 1]", "click[back to search]"],
        )

    def test_grouped_webshop_rollouts_generate_loadable_hierarchy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "webshop_model_2.json"
            audit = Path(temp_dir) / "webshop_model_2.audit.json"
            config = GenerationConfig(
                output_path=output,
                audit_path=audit,
                target_task_bundles=2,
                task_groups_per_batch=2,
                rollouts_per_task=2,
                max_steps=2,
                batches_per_pass=1,
                max_passes=1,
            )
            policy = _AlternatingPolicy()
            updater = _FakeUpdater()
            generator = WebShopInitialSkillBankGenerator(
                config=config,
                envs=_FakeWebShopEnvs(),
                reasoning_policy=policy,
                skill_updater=updater,
                memory=_FakeMemory(),
            )

            result = generator.run()

            self.assertEqual(len(result["task_skills"]), 2)
            self.assertEqual(len(result["step_skills"]), 2)
            self.assertEqual(
                [group["group_uid"] for group in updater.groups],
                ["webshop-goal-501", "webshop-goal-777"],
            )
            self.assertTrue(all(group["group_size"] == 2 for group in updater.groups))
            self.assertTrue(all(group["success_count"] == 1 for group in updater.groups))
            self.assertTrue(all(group["failure_count"] == 1 for group in updater.groups))
            self.assertTrue(all(len(group["trajectories"]) == 2 for group in updater.groups))
            self.assertTrue(output.is_file())
            self.assertTrue(audit.is_file())
            audit_data = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(
                audit_data["mode"],
                "offline_webshop_task_group_initialization",
            )
            self.assertTrue(policy.prompts)
            self.assertTrue(all("Relevant Experience" not in prompt for prompt in policy.prompts))
            self.assertTrue(all("Admissible actions" in prompt for prompt in policy.prompts))


if __name__ == "__main__":
    unittest.main()
