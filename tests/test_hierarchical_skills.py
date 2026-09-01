import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _load_skills_only_memory():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("agent_system")
    package.__path__ = [str(root / "agent_system")]
    memory_package = types.ModuleType("agent_system.memory")
    memory_package.__path__ = [str(root / "agent_system" / "memory")]
    sys.modules.setdefault("agent_system", package)
    sys.modules.setdefault("agent_system.memory", memory_package)
    for name, path in (("agent_system.memory.base", root / "agent_system" / "memory" / "base.py"), ("agent_system.memory.skills_only_memory", root / "agent_system" / "memory" / "skills_only_memory.py")):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["agent_system.memory.skills_only_memory"].SkillsOnlyMemory


SkillsOnlyMemory = _load_skills_only_memory()


class HierarchicalSkillsTest(unittest.TestCase):
    def setUp(self):
        self.memory = SkillsOnlyMemory(load_initial_skills=False, retrieval_mode="template")

    def test_step_retrieval_is_limited_to_selected_task_children(self):
        self.memory.add_hierarchical_skill_pairs([
            {"task_skill": {"title": "Kitchen", "principle": "k", "retrieval_obs": "kitchen"}, "step_skill": {"title": "Use sink", "principle": "s", "retrieval_obs": "sink visible"}},
            {"task_skill": {"title": "Bedroom", "principle": "b", "retrieval_obs": "bedroom"}, "step_skill": {"title": "Use bed", "principle": "b", "retrieval_obs": "bed visible"}},
        ])
        task_id = self.memory.skills["task_skills"][0]["skill_id"]
        result = self.memory.retrieve_child_step_skills_batch([[task_id]], ["Current observation: a visible sink"], top_k=5)[0]
        self.assertEqual(result["query_text"], "a visible sink")
        self.assertEqual([s["title"] for s in result["step_skills"]], ["Use sink"])

    def test_shared_deduplicated_step_survives_other_parent(self):
        self.memory.add_hierarchical_skill_pairs([
            {"task_skill": {"title": "A", "principle": "a", "retrieval_obs": "a"}, "step_skill": {"title": "Shared", "principle": "s", "retrieval_obs": "visible"}},
            {"task_skill": {"title": "B", "principle": "b", "retrieval_obs": "b"}, "step_skill": {"title": "Shared", "principle": "s", "retrieval_obs": "visible"}},
        ])
        self.assertEqual(len(self.memory.skills["step_skills"]), 1)
        self.memory.remove_skill(self.memory.skills["task_skills"][0]["skill_id"])
        self.assertEqual(len(self.memory.skills["step_skills"]), 1)

    def test_embedding_retrieval_uses_similarity_only_when_reranking_is_disabled(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is required for the embedding retrieval path")
        memory = SkillsOnlyMemory(
            load_initial_skills=False,
            retrieval_mode="embedding",
            retrieval_alpha=None,
            similarity_threshold=None,
        )
        memory.skills["task_skills"] = [
            {"skill_id": "task_near", "title": "Near", "retrieval_obs": "near", "utility": 0.0},
            {"skill_id": "task_far", "title": "Far", "retrieval_obs": "far", "utility": 100.0},
        ]
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        class FakeEmbeddingModel:
            def encode(self, texts, **kwargs):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        memory._compute_pool_embeddings = lambda pool: {
            "items": memory.skills[pool],
            "embeddings": embeddings,
        }
        memory._get_embedding_model = lambda: FakeEmbeddingModel()

        result = memory.retrieve_task_skills_batch(["query"], top_k=1)[0]
        self.assertEqual([skill["skill_id"] for skill in result["task_skills"]], ["task_near"])
        self.assertEqual(result["task_skills"][0]["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
