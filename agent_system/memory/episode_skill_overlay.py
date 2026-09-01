"""Per-trajectory, in-memory skill overlays for meta-attempt rollouts.

The overlay is deliberately isolated from ``SkillsOnlyMemory`` used by normal
retrieval: edits are evaluated inside one rollout and are never written to the
global skill bank.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .skills_only_memory import SkillsOnlyMemory


EDITABLE_FIELDS = ("title", "principle", "when_to_apply", "retrieval_obs")


class EpisodeSkillOverlay:
    """A cloned task-skill subtree plus ordered, local edit application."""

    def __init__(
        self,
        base_memory: SkillsOnlyMemory,
        initial_memories: Dict[str, Any],
        shared_embedding_model: Optional[Any] = None,
        shared_child_embedding_cache: Optional[Dict[Tuple, Any]] = None,
    ):
        self.initial_memories = deepcopy(initial_memories or {})
        selected_tasks = deepcopy(self.initial_memories.get("task_skills") or [])
        selected_ids = {s.get("skill_id") for s in selected_tasks if s.get("skill_id")}
        child_ids = {
            child_id
            for task in selected_tasks
            for child_id in task.get("step_skill_ids", [])
        }
        base_steps = base_memory.skills.get("step_skills", [])
        selected_steps = [deepcopy(s) for s in base_steps if s.get("skill_id") in child_ids]

        # Keep retrieval private to this overlay, but share the read-only encoder
        # and content-addressed child embedding cache across all overlays in the
        # same rollout. Each overlay still owns an independent editable skill pool.
        self.memory = SkillsOnlyMemory(
            skills_json_path=None,
            retrieval_mode=base_memory.retrieval_mode,
            embedding_model_path=base_memory.embedding_model_path,
            task_specific_top_k=base_memory.task_specific_top_k,
            device=base_memory.device,
            skill_retrieval_service_url=None,
            skill_text_for_retrieval=base_memory._skill_text_for_retrieval,
            load_initial_skills=False,
            similarity_threshold=base_memory.similarity_threshold,
            skill_retrieval_timeout=getattr(base_memory, "_retrieval_timeout", 60),
            retrieval_top_2k=getattr(base_memory, "_retrieval_top_2k", None),
            retrieval_alpha=getattr(base_memory, "_retrieval_alpha", None),
            retrieval_ucb_c=getattr(base_memory, "_retrieval_ucb_c", 0.5),
            eviction_enabled=False,
            log_load_summary=False,
        )
        self.memory.skills = {"task_skills": selected_tasks, "step_skills": selected_steps}
        if shared_embedding_model is not None:
            self.memory._embedding_model = shared_embedding_model
        # This cache is rollout-private but shared by all trajectory overlays.
        # Its content-addressed keys make unchanged copies of the same task
        # bundle reuse one child-skill embedding matrix. Edited bundles receive
        # a new key automatically and therefore cannot contaminate each other.
        self._shared_child_embedding_cache = (
            shared_child_embedding_cache
            if shared_child_embedding_cache is not None
            else {}
        )
        self.memory._normalize_hierarchy()
        self.memory._task_skill_embeddings_cache = None
        self.memory._step_skill_embeddings_cache = None
        self.applied_edits: List[Dict[str, Any]] = []
        self.rejected_edits: List[Dict[str, Any]] = []
        self.edit_audit: List[Dict[str, Any]] = []
        self._insert_counter = 0

    @property
    def task_skill_ids(self) -> List[str]:
        return [s["skill_id"] for s in self.memory.skills.get("task_skills", []) if s.get("skill_id")]

    def _step_by_id(self, skill_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return next((s for s in self.memory.skills.get("step_skills", []) if s.get("skill_id") == skill_id), None)

    def _task_by_id(self, skill_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return next((s for s in self.memory.skills.get("task_skills", []) if s.get("skill_id") == skill_id), None)

    def _next_overlay_step_id(self) -> str:
        existing = {s.get("skill_id") for s in self.memory.skills.get("step_skills", [])}
        while True:
            self._insert_counter += 1
            skill_id = f"overlay_step_{self._insert_counter:03d}"
            if skill_id not in existing:
                return skill_id

    def _reject(self, record: Dict[str, Any], reason: str) -> None:
        rejected = {"decision": deepcopy(record), "reason": reason}
        self.rejected_edits.append(rejected)
        self.edit_audit.append({"applied": False, **deepcopy(rejected)})

    def _record_applied(self, record: Dict[str, Any], effect: Dict[str, Any]) -> None:
        self.applied_edits.append(effect)
        self.edit_audit.append({
            "applied": True,
            "step": record.get("step"),
            "decision": deepcopy((record or {}).get("decision") or {}),
            "effect": deepcopy(effect),
        })

    @staticmethod
    def _editable_text_fields(skill_payload: Any) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """Return editable string fields, rejecting malformed nested values.

        Model output can be valid JSON with an object-shaped ``skill`` while
        individual fields are still arrays, objects, or numbers. Letting those
        values into the overlay breaks embedding retrieval later, where skill
        fields are treated as text.
        """
        if not isinstance(skill_payload, dict):
            return None, "missing_skill_content"

        fields: Dict[str, str] = {}
        for field in EDITABLE_FIELDS:
            if field not in skill_payload or skill_payload[field] is None:
                continue
            value = skill_payload[field]
            if not isinstance(value, str):
                return None, "invalid_skill_field_type"
            fields[field] = value
        return fields, None

    def apply(self, record: Dict[str, Any]) -> bool:
        """Apply one parsed Skill-Agent decision to this overlay only."""
        record = record or {}
        decision = (record or {}).get("decision") or {}
        if not isinstance(decision, dict):
            self._reject(record, "invalid_json")
            return False
        # The parser intentionally falls back to KEEP so a malformed model
        # response cannot break collection.  In a meta-attempt, however, that
        # fallback must remain an *unapplied* edit rather than a real KEEP
        # decision in the audit trail.
        if not decision.get("parse_ok", False):
            self._reject(record, "invalid_json")
            return False
        action = str(decision.get("action", "KEEP")).upper()
        if action == "KEEP":
            self._record_applied(record, {"action": "KEEP", "target_skill_id": decision.get("target_skill_id")})
            return True
        if action not in ("INSERT", "UPDATE", "DELETE"):
            self._reject(record, "unsupported_action")
            return False

        target_id = decision.get("target_skill_id")
        skill_payload = decision.get("skill") or {}
        if action == "INSERT":
            parent_id = decision.get("parent_task_skill_id") or target_id
            parent = self._task_by_id(parent_id)
            if parent is None:
                self._reject(record, "insert_parent_not_in_overlay")
                return False
            new_skill, payload_error = self._editable_text_fields(skill_payload)
            if payload_error == "invalid_skill_field_type":
                self._reject(record, "insert_invalid_skill_field_type")
                return False
            if payload_error or not any(value.strip() for value in new_skill.values()):
                self._reject(record, "insert_missing_skill_content")
                return False
            new_skill.update({
                "skill_id": self._next_overlay_step_id(),
                "parent_task_skill_id": parent_id,
                "overlay_only": True,
            })
            self.memory.skills.setdefault("step_skills", []).append(new_skill)
            parent.setdefault("step_skill_ids", []).append(new_skill["skill_id"])
            parent["step_skill_ids"] = list(dict.fromkeys(parent["step_skill_ids"]))
            effect = {"action": action, "target_skill_id": new_skill["skill_id"], "parent_task_skill_id": parent_id}
        elif action == "UPDATE":
            target = self._step_by_id(target_id)
            if target is None:
                self._reject(record, "update_target_not_in_overlay")
                return False
            changes, payload_error = self._editable_text_fields(skill_payload)
            if payload_error == "invalid_skill_field_type":
                self._reject(record, "update_invalid_skill_field_type")
                return False
            if payload_error:
                self._reject(record, "update_missing_skill_content")
                return False
            if not changes:
                self._reject(record, "update_missing_skill_content")
                return False
            target.update(changes)
            effect = {"action": action, "target_skill_id": target_id, "fields": sorted(changes)}
        else:  # DELETE
            target = self._step_by_id(target_id)
            if target is None:
                self._reject(record, "delete_target_not_in_overlay")
                return False
            self.memory.skills["step_skills"] = [s for s in self.memory.skills["step_skills"] if s.get("skill_id") != target_id]
            for task in self.memory.skills.get("task_skills", []):
                task["step_skill_ids"] = [sid for sid in task.get("step_skill_ids", []) if sid != target_id]
            effect = {"action": action, "target_skill_id": target_id}

        self.memory._task_skill_embeddings_cache = None
        self.memory._step_skill_embeddings_cache = None
        self._record_applied(record, effect)
        return True

    def apply_all(self, records: List[Dict[str, Any]]) -> List[bool]:
        """Apply records in trajectory order and return one status per record."""
        return [self.apply(record) for record in records]

    def retrieve(self, query_text: str, top_k_step: int) -> Dict[str, Any]:
        step_res = self.memory.retrieve_child_step_skills_batch(
            [self.task_skill_ids], [query_text], top_k=top_k_step
        )[0]
        return {
            "task_skills": [deepcopy(s) for s in self.memory.skills.get("task_skills", [])],
            "step_skills": step_res.get("step_skills", []),
            "query_text": step_res.get("query_text", query_text),
            "overlay": True,
        }

    def _child_embedding_cache_key(self) -> Tuple:
        """Content key for the current ordered child-step bundle."""
        return tuple(
            (
                str(skill.get("skill_id") or ""),
                self.memory._skill_to_text(skill),
            )
            for skill in self.memory.skills.get("step_skills", [])
        )

    @classmethod
    def retrieve_many(
        cls,
        overlays: List["EpisodeSkillOverlay"],
        query_texts: List[str],
        top_k_step: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve for all overlays with one batched encoder invocation.

        Observation texts are encoded together. On the first call, child-skill
        texts for every unique edited bundle are appended to that same encoder
        batch; later calls reuse the rollout-shared child embedding cache.
        """
        if len(overlays) != len(query_texts):
            raise ValueError("overlays and query_texts must have the same length")
        if not overlays:
            return []
        if any(overlay.memory.retrieval_mode != "embedding" for overlay in overlays):
            return [
                overlay.retrieve(query, top_k_step)
                for overlay, query in zip(overlays, query_texts)
            ]

        import numpy as np

        shared_cache = overlays[0]._shared_child_embedding_cache
        # Make the cache shared even for callers that constructed overlays
        # without explicitly passing one. Production passes it explicitly.
        for overlay in overlays:
            overlay._shared_child_embedding_cache = shared_cache

        normalized_queries = [
            overlay.memory._observation_only(query)
            for overlay, query in zip(overlays, query_texts)
        ]
        bundle_keys = [overlay._child_embedding_cache_key() for overlay in overlays]
        missing_bundles = {}
        for overlay, bundle_key in zip(overlays, bundle_keys):
            if bundle_key and bundle_key not in shared_cache:
                missing_bundles.setdefault(
                    bundle_key,
                    (overlay.memory, list(overlay.memory.skills.get("step_skills", []))),
                )

        missing_texts = []
        missing_ranges = {}
        for bundle_key, (memory, skills) in missing_bundles.items():
            start = len(missing_texts)
            missing_texts.extend(
                memory._skill_to_text(skill) for skill in skills
            )
            missing_ranges[bundle_key] = (start, len(missing_texts))

        # One encoder call per environment step: all observations plus any
        # previously unseen child-bundle documents.
        model = overlays[0].memory._get_embedding_model(log_loading=False)
        encoded = model.encode(
            normalized_queries + missing_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        encoded = np.asarray(encoded)
        if encoded.ndim == 1:
            encoded = np.expand_dims(encoded, axis=0)
        query_embeddings = encoded[: len(normalized_queries)]
        encoded_missing = encoded[len(normalized_queries):]
        for bundle_key, (start, end) in missing_ranges.items():
            shared_cache[bundle_key] = encoded_missing[start:end]

        outputs = []
        for query_idx, (overlay, query, bundle_key) in enumerate(
            zip(overlays, normalized_queries, bundle_keys)
        ):
            memory = overlay.memory
            items = memory.skills.get("step_skills", [])
            if not items or top_k_step <= 0:
                selected_steps = []
            else:
                child_embeddings = shared_cache[bundle_key]
                scores = np.asarray(child_embeddings @ query_embeddings[query_idx]).ravel()
                eligible = list(range(len(items)))
                eligible.sort(key=lambda index: (-scores[index], index))
                if memory._retrieval_alpha is not None:
                    limit = (
                        memory._retrieval_top_2k
                        if memory._retrieval_top_2k is not None
                        else max(2 * top_k_step, top_k_step + 1)
                    )
                    selected_indices = memory._apply_simutil_ucb(
                        "step_skills", eligible[:limit], scores, top_k_step
                    )
                else:
                    selected_indices = eligible[:top_k_step]
                selected_steps = []
                for index in selected_indices:
                    similarity = float(scores[index])
                    if (
                        memory.similarity_threshold is not None
                        and similarity < memory.similarity_threshold
                    ):
                        continue
                    skill = dict(items[index])
                    skill["similarity"] = similarity
                    skill.update(memory._get_skill_ranking_meta(
                        "step_skills", index, similarity
                    ))
                    selected_steps.append(skill)
            outputs.append({
                "task_skills": [
                    deepcopy(skill)
                    for skill in memory.skills.get("task_skills", [])
                ],
                "step_skills": selected_steps,
                "query_text": query,
                "overlay": True,
            })
        return outputs

    def snapshot(self) -> Dict[str, Any]:
        return {
            "initial": deepcopy(self.initial_memories),
            "final": deepcopy(self.memory.skills),
            "patches": deepcopy(self.edit_audit),
            "applied_edits": deepcopy(self.applied_edits),
            "rejected_edits": deepcopy(self.rejected_edits),
        }
