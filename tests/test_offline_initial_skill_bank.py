import json
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from initial_skill_bank.generate_alfworld_initial_skill_bank import (
    AlfWorldInitialSkillBankGenerator,
    GenerationConfig,
    OpenAICompatiblePolicy,
    RateLimitedChatClient,
    RequestsPerMinuteLimiter,
    SophonChatCompletionAdapter,
    _audit_path_for_skill_bank,
    _build_parser,
    _move_generated_file,
    _select_task_groups,
    _skill_bank_output_path,
)


class _FakeAlfWorldEnvs:
    def __init__(self):
        self.get_admissible_commands = [
            ["finish", "wait"],
            ["finish", "wait"],
            ["finish", "wait"],
            ["finish", "wait"],
        ]

    def reset(self):
        observations = [
            "Room A.\nYour task is to: put apple on table",
            "Room A.\nYour task is to: put apple on table",
            "Room B.\nYour task is to: heat potato",
            "Room B.\nYour task is to: heat potato",
        ]
        infos = [
            {"extra.gamefile": "task-a"},
            {"extra.gamefile": "task-a"},
            {"extra.gamefile": "task-b"},
            {"extra.gamefile": "task-b"},
        ]
        return observations, None, infos

    def step(self, actions):
        won = [action == "finish" for action in actions]
        infos = [{"won": float(value)} for value in won]
        return ["done"] * 4, None, [10.0 if value else 0.0 for value in won], [True] * 4, infos


class _AlternatingPolicy:
    def __init__(self):
        self.prompts = []

    def complete_batch(self, prompts):
        self.prompts.extend(prompts)
        return [
            "<think>solve</think><action>finish</action>" if index % 2 == 0
            else "<think>fail</think><action>wait</action>"
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
            task = {
                "title": f"strategy-{group['group_uid']}",
                "principle": "use the evidence from all sampled attempts",
                "when_to_apply": trajectory["task"],
                "retrieval_obs": trajectory["task"],
                "initialization_group_uid": group["group_uid"],
                "utility": group["group_success_rate"],
            }
            step = {
                "title": "first decision",
                "principle": "finish when the goal is ready",
                "when_to_apply": "at the initial state",
                "retrieval_obs": trajectory["refined_trajectory"]["turns"][0]["observation"],
                "source_trajectory_index": 1,
                "source_turn": 1,
            }
            pairs.append({"task_skill": task, "step_skill": step})
        metadata = {"mode": "fake", "num_task_groups": len(groups)}
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
            raw_task = dict(pair["task_skill"])
            group_uid = raw_task["initialization_group_uid"]
            task = by_group.get(group_uid)
            if task is None:
                raw_task["skill_id"] = f"task_skill_{len(self.skills['task_skills']) + 1}"
                raw_task["step_skill_ids"] = []
                self.skills["task_skills"].append(raw_task)
                by_group[group_uid] = raw_task
                task = raw_task
                added_task += 1
            step = dict(pair["step_skill"])
            step["skill_id"] = f"step_skill_{len(self.skills['step_skills']) + 1}"
            step["parent_task_skill_id"] = task["skill_id"]
            self.skills["step_skills"].append(step)
            task["step_skill_ids"].append(step["skill_id"])
            added_step += 1
        return {"task_skills": added_task, "step_skills": added_step}

    def save_skills(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.skills, handle, indent=2)


class OfflineInitialSkillBankTest(unittest.TestCase):
    def test_generation_cli_uses_safe_parallel_defaults(self):
        args = _build_parser().parse_args([
            "--api-base-url", "https://gateway.example/chatCompletion",
            "--model", "test-model",
        ])

        self.assertEqual(args.task_groups_per_batch, 1)
        self.assertEqual(args.rollouts_per_task, 8)
        self.assertEqual(args.max_concurrent_api_calls, 8)
        self.assertEqual(args.max_api_requests_per_minute, 60)
        self.assertEqual(args.api_retries, 10)
        self.assertEqual(args.dataset_name, "alfworld")
        self.assertIsNone(args.output)

    def test_reasoning_policy_skips_completion_after_retries(self):
        class _FailingClient:
            def __init__(self):
                self.chat = self
                self.completions = self
                self.calls = 0

            def create(self, **kwargs):
                del kwargs
                self.calls += 1
                raise RuntimeError("empty response")

        client = _FailingClient()
        policy = OpenAICompatiblePolicy(
            base_url="https://gateway.example",
            api_key="secret",
            model="test-model",
            retries=3,
            client=client,
        )
        sleeps = []

        with patch(
            "initial_skill_bank.generate_alfworld_initial_skill_bank.time.sleep",
            side_effect=sleeps.append,
        ), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            result = policy._complete_one("prompt")

        self.assertEqual(result, "")
        self.assertEqual(client.calls, 3)
        self.assertEqual(sleeps, [5, 10])
        self.assertIn("Skipping completion after 3 failed attempts", stderr.getvalue())

    def test_default_output_name_uses_dataset_model_and_actual_count(self):
        path = _skill_bank_output_path("alfworld", "Qwen/Qwen 3:4B", 287)

        self.assertEqual(path.name, "alfworld_Qwen-Qwen-3-4B_287.json")
        self.assertEqual(
            _audit_path_for_skill_bank(path).name,
            "alfworld_Qwen-Qwen-3-4B_287.audit.json",
        )

    def test_generated_output_can_be_renamed_to_actual_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "alfworld_model_300.json"
            destination = Path(temp_dir) / "alfworld_model_17.json"
            source.write_text('{"task_skills": []}', encoding="utf-8")

            result = _move_generated_file(source, destination, overwrite=False)

            self.assertEqual(result, destination)
            self.assertFalse(source.exists())
            self.assertTrue(destination.is_file())

    def test_rate_limiter_is_shared_and_counts_every_request_attempt(self):
        now = [0.0]
        sleeps = []

        def clock():
            return now[0]

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        class _Delegate:
            def __init__(self):
                self.chat = self
                self.completions = self
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return kwargs

        delegate = _Delegate()
        client = RateLimitedChatClient(
            delegate,
            RequestsPerMinuteLimiter(
                2,
                window_seconds=60,
                clock=clock,
                sleep=sleep,
            ),
        )

        # These calls can represent two Reasoning requests and one Reflection
        # request because both roles receive this exact shared wrapper.
        client.chat.completions.create(model="m", messages=[])
        client.chat.completions.create(model="m", messages=[])
        client.chat.completions.create(model="m", messages=[])

        self.assertEqual(delegate.calls, 3)
        self.assertEqual(sleeps, [60.0])

    def test_sophon_adapter_uses_custom_endpoint_header_and_payload(self):
        captured = {}

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"choices": [{"message": {"content": "answer"}}]}}

        requests_module = types.ModuleType("requests")

        def post(url, headers, json, timeout):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return _Response()

        requests_module.post = post
        previous = sys.modules.get("requests")
        sys.modules["requests"] = requests_module
        try:
            adapter = SophonChatCompletionAdapter(
                endpoint="https://gateway/chatCompletion",
                api_key="secret",
                timeout=30,
                cache=True,
            )
            response = adapter.chat.completions.create(
                model="model-name",
                messages=[{"role": "user", "content": "hello"}],
                temperature=1,
            )
        finally:
            if previous is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = previous

        self.assertEqual(response.choices[0].message.content, "answer")
        self.assertEqual(captured["headers"]["api-key"], "secret")
        self.assertEqual(captured["json"]["model"], "model-name")
        self.assertEqual(captured["json"]["n"], 1)
        self.assertFalse(captured["json"]["stream"])
        self.assertTrue(captured["json"]["cache"])

    def test_groups_sample_representative_evidence_for_each_outcome_case(self):
        trajectories = []
        for group_uid, outcomes in (
            ("mixed", [1, 1, 0]),
            ("all-success", [1, 1]),
            ("all-failure", [0, 0]),
        ):
            for index, outcome in enumerate(outcomes):
                trajectories.append({
                    "group_uid": group_uid,
                    "traj_uid": f"{group_uid}-{index}",
                    "succeeded": bool(outcome),
                    "episode_reward": float(index),
                    "episode_length": index + 1,
                })
        groups = _select_task_groups(
            trajectories,
            max_groups=10,
            max_trajectories_per_group=1,
        )
        self.assertEqual(
            [group["group_uid"] for group in groups],
            ["all-failure", "mixed", "all-success"],
        )
        self.assertEqual(groups[0]["group_size"], 2)
        self.assertEqual(len(groups[0]["trajectories"]), 1)
        self.assertEqual(groups[0]["success_count"], 0)
        self.assertEqual(groups[0]["failure_count"], 2)
        self.assertEqual(groups[0]["reflection_selection_mode"], "one_failure")
        self.assertFalse(groups[0]["trajectories"][0]["succeeded"])

        self.assertEqual(groups[1]["group_size"], 3)
        self.assertEqual(len(groups[1]["trajectories"]), 2)
        self.assertEqual(groups[1]["success_count"], 2)
        self.assertEqual(groups[1]["failure_count"], 1)
        self.assertEqual(groups[1]["reflection_selection_mode"], "one_success_one_failure")
        self.assertEqual(
            sorted(item["succeeded"] for item in groups[1]["trajectories"]),
            [False, True],
        )

        self.assertEqual(groups[2]["group_size"], 2)
        self.assertEqual(len(groups[2]["trajectories"]), 1)
        self.assertEqual(groups[2]["success_count"], 2)
        self.assertEqual(groups[2]["failure_count"], 0)
        self.assertEqual(groups[2]["reflection_selection_mode"], "one_success")
        self.assertTrue(groups[2]["trajectories"][0]["succeeded"])

    def test_environment_to_loadable_hierarchical_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "initial_skill_bank.json"
            audit = Path(temp_dir) / "initial_skill_bank.audit.json"
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
            generator = AlfWorldInitialSkillBankGenerator(
                config=config,
                envs=_FakeAlfWorldEnvs(),
                reasoning_policy=policy,
                skill_updater=updater,
                memory=_FakeMemory(),
            )

            result = generator.run()

            self.assertEqual(len(result["task_skills"]), 2)
            self.assertEqual(len(result["step_skills"]), 2)
            self.assertTrue(output.is_file())
            self.assertTrue(audit.is_file())
            saved = json.loads(output.read_text(encoding="utf-8"))
            audit_data = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(saved, result)
            for task in saved["task_skills"]:
                self.assertEqual(len(task["step_skill_ids"]), 1)
                step = next(
                    item for item in saved["step_skills"]
                    if item["skill_id"] == task["step_skill_ids"][0]
                )
                self.assertEqual(step["parent_task_skill_id"], task["skill_id"])

            self.assertEqual(len(updater.groups), 2)
            for group in updater.groups:
                self.assertEqual(len(group["trajectories"]), 2)
                self.assertEqual(group["success_count"], 1)
                self.assertEqual(group["failure_count"], 1)
            evidence = audit_data["batches"][0]["reflection_evidence"]
            self.assertEqual(len(evidence), 2)
            self.assertTrue(all(item["selection_mode"] == "one_success_one_failure" for item in evidence))
            self.assertTrue(all(item["sampled_success_count"] == 1 for item in evidence))
            self.assertTrue(all(item["sampled_failure_count"] == 1 for item in evidence))
            self.assertTrue(policy.prompts)
            self.assertTrue(all("Relevant Experience" not in prompt for prompt in policy.prompts))


if __name__ == "__main__":
    unittest.main()
