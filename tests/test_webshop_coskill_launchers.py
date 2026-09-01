import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "examples_coskill"
ENV_MANAGER = PROJECT_ROOT / "agent_system" / "environments" / "env_manager.py"


class WebShopCoSkillLauncherTest(unittest.TestCase):
    def test_base_launcher_uses_webshop_and_coskill_configuration(self):
        script = (SCRIPT_DIR / "run_webshop_coskill.sh").read_text(encoding="utf-8")

        self.assertIn("env.env_name=Webshop", script)
        self.assertIn('env.webshop.use_small="${WEBSHOP_USE_SMALL}"', script)
        self.assertIn("algorithm.adv_estimator=gigpo", script)
        self.assertIn('+env.skills_only_memory.load_initial_skills=True', script)
        self.assertIn('+env.skill_agent.meta_attempts.num_attempts="${NUM_META_ATTEMPTS}"', script)
        self.assertIn('MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"', script)
        self.assertIn('MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"', script)
        self.assertIn('VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"', script)
        self.assertIn('MAX_PLAY_STEPS="${MAX_PLAY_STEPS:-15}"', script)
        self.assertIn("trainer.save_freq=250", script)
        self.assertIn('TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"', script)
        self.assertIn('VAL_DATA_PATH="${VAL_DATA_PATH:-}"', script)
        self.assertIn('MODEL_PATH="${MODEL_PATH:-}"', script)
        self.assertIn('SKILL_EMBEDDING_MODEL_PATH="${SKILL_EMBEDDING_MODEL_PATH:-}"', script)
        self.assertNotIn("/mnt/", script)
        self.assertNotIn("PREPARE_WEBSHOP_DATA", script)
        self.assertNotIn("prepare_webshop_data.py", script)
        self.assertNotIn("ALFWORLD_DATA", script)

    def test_two_attempt_and_no_skill_rl_wrappers_target_webshop_base(self):
        two_attempt = (
            SCRIPT_DIR / "run_webshop_coskill_2attempt_15steps.sh"
        ).read_text(encoding="utf-8")
        no_skill_rl = (
            SCRIPT_DIR / "run_webshop_coskill_2attempt_15steps_no_skill_rl.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('NUM_META_ATTEMPTS="${NUM_META_ATTEMPTS:-2}"', two_attempt)
        self.assertIn('MAX_PLAY_STEPS="${MAX_PLAY_STEPS:-15}"', two_attempt)
        self.assertIn('run_webshop_coskill.sh" "$@"', two_attempt)
        self.assertIn("SKILL_AGENT_RL_TRAINING_ENABLED=false", no_skill_rl)
        self.assertIn('run_webshop_coskill_2attempt_15steps.sh" "$@"', no_skill_rl)

    def test_small_webshop_catalogue_uses_matching_1k_search_index(self):
        source = ENV_MANAGER.read_text(encoding="utf-8")

        self.assertIn("num_products = 1000", source)
        self.assertIn("num_products = None", source)
        self.assertIn("'num_products': num_products", source)


if __name__ == "__main__":
    unittest.main()
