# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from copy import deepcopy

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.multi_turn_rollout.meta_attempt import (
    annotate_meta_attempt_returns,
    build_reasoning_attempt_traj_uid,
)
from agent_system.environments import EnvironmentManagerBase
from agent_system.memory import EpisodeSkillOverlay
from typing import List, Dict, Any, Optional, Tuple
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


SKILL_AGENT_ACTIONS = ("INSERT", "UPDATE", "DELETE", "KEEP")


def _json_safe(value: Any) -> Any:
    """Convert rollout metadata to JSON-friendly values before recording it."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _parse_skill_agent_decision(response: str) -> Dict[str, Any]:
    """Best-effort parser; malformed output is safely treated as KEEP."""
    raw = (response or "").strip()
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(payload, dict):
        payload = {}

    action = str(payload.get("action", payload.get("operation", "KEEP"))).upper().strip()
    if action not in SKILL_AGENT_ACTIONS:
        action = "KEEP"
    return {
        "action": action,
        "target_skill_id": payload.get("target_skill_id"),
        "parent_task_skill_id": payload.get("parent_task_skill_id"),
        "skill": _json_safe(payload.get("skill")),
        "reason": str(payload.get("reason", "")),
        "parse_ok": bool(payload),
    }


def _compact_env_feedback(info: Any, reward: Any, done: Any) -> Dict[str, Any]:
    """Keep the editor prompt informative without serialising arbitrary env state."""
    info = info if isinstance(info, dict) else {"raw_info": info}
    selected = {
        key: info[key]
        for key in ("is_action_valid", "tool_calling", "won", "feedback", "message", "error", "result")
        if key in info
    }
    selected["reward"] = reward
    selected["done"] = done
    return _json_safe(selected)


def _build_skill_agent_prompt(
    current_state: str,
    next_state: str,
    active_step_skills: List[Dict[str, Any]],
    reasoning_action: str,
    feedback: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    """Prompt for the online skill editor. Edits are proposals in this first phase."""
    return (
        "You are the Skill Agent for an interactive task-solving system.\n"
        "The Reasoning Agent has just executed one action. Decide how the currently active step-level skill should be edited for future use.\n\n"
        "Choose exactly one action:\n"
        "- INSERT: propose a new step-level skill when no active skill adequately captures the useful behaviour.\n"
        "- UPDATE: improve one active step-level skill.\n"
        "- DELETE: remove one harmful, redundant, or invalid active step-level skill.\n"
        "- KEEP: retain the active skill unchanged.\n\n"
        "Return exactly one JSON object and no markdown:\n"
        '{"action":"INSERT|UPDATE|DELETE|KEEP","target_skill_id":"step id or null","parent_task_skill_id":"task id required for INSERT","skill":{"title":"...","principle":"...","when_to_apply":"..."},"reason":"..."}\n\n'
        f"Raw environment observation before the reasoning action:\n{current_state}\n\n"
        f"Raw environment observation after the action:\n{next_state}\n\n"
        f"Currently active step-level skills:\n{json.dumps(_json_safe(active_step_skills), ensure_ascii=False)}\n\n"
        f"Reasoning Agent action:\n{reasoning_action}\n\n"
        f"Environment feedback:\n{json.dumps(_json_safe(feedback), ensure_ascii=False)}\n\n"
        f"Earlier reasoning trajectory:\n{json.dumps(_json_safe(history), ensure_ascii=False)}\n\n"
        "This is proposal-only collection: do not assume an earlier proposal has already changed the skill bank."
    )


def _skill_input_to_retrieval(s: Dict[str, Any], mode: str = "full") -> str:
    """Text that was used as input (document side) for this skill in retrieval.
    mode: 'full' = title + principle + when_to_apply; 'when_to_apply' = only when_to_apply; 'principle' = only principle.
    """
    if mode == "when_to_apply":
        return (s.get("when_to_apply") or "").strip()
    if mode == "principle":
        return (s.get("principle") or "").strip()
    parts = [s.get("title", ""), s.get("principle", ""), s.get("when_to_apply", "")]
    return ". ".join(p for p in parts if p and str(p).strip()).strip(". ")


def _task_step_skill_row(s: Dict[str, Any]) -> Dict[str, Any]:
    """One row for task_skill or step_skill in snapshot (skill_id, title, input_to_retrieval, similarity, utility, ucb, retrieval_score)."""
    inp = (s.get("retrieval_obs") or "").strip() or _skill_input_to_retrieval(s, "full")
    row = {"title": s.get("title", ""), "input_to_retrieval": inp, "similarity": s.get("similarity")}
    if s.get("skill_id") is not None:
        row["skill_id"] = s["skill_id"]
    if "utility" in s:
        row["utility"] = s["utility"]
    if "ucb" in s:
        row["ucb"] = s["ucb"]
    if "retrieval_score" in s:
        row["retrieval_score"] = s["retrieval_score"]
    return row


def _snapshot_retrieved_memories(mem: Dict[str, Any], skill_text_mode: str = "full") -> Dict[str, Any]:
    """For JSON: query_text and per-skill task_skills, step_skills."""
    return {
        "query_text": mem.get("query_text", ""),
        "task_skills": [_task_step_skill_row(s) for s in mem.get("task_skills", [])],
        "step_skills": [_task_step_skill_row(s) for s in mem.get("step_skills", [])],
    }

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_query_texts = obs.get('query_text', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        obs_query_text = (obs_query_texts[item] if obs_query_texts is not None and item < len(obs_query_texts) else None) or ""
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'query_text': obs_query_text,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def _generate_skill_agent_trajectory_step(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        active_step_skills: List[List[Dict[str, Any]]],
        reasoning_history: List[List[Dict[str, Any]]],
        text_actions: List[str],
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        active_masks: np.ndarray,
        uid_batch: np.ndarray,
        traj_uid: np.ndarray,
        step: int,
        max_history_steps: int,
    ) -> Tuple[List[Dict[str, Any]], DataProto]:
        """Run the editor and return both records and its RL-ready generation batch."""
        batch_size = len(text_actions)
        current_texts = obs.get("text") or [""] * batch_size
        next_texts = next_obs.get("text") or [""] * batch_size
        current_states = obs.get("anchor")
        next_states = next_obs.get("anchor")
        if current_states is None:
            current_states = current_texts
        if next_states is None:
            next_states = next_texts
        prompts: List[str] = []
        feedbacks: List[Dict[str, Any]] = []
        for i in range(batch_size):
            feedback = _compact_env_feedback(infos[i], rewards[i], dones[i])
            feedbacks.append(feedback)
            prompts.append(
                _build_skill_agent_prompt(
                    current_state=str(current_states[i]),
                    next_state=str(next_states[i]),
                    active_step_skills=active_step_skills[i],
                    reasoning_action=text_actions[i],
                    feedback=feedback,
                    history=reasoning_history[i][-max_history_steps:] if max_history_steps > 0 else [],
                )
            )

        # The Skill Agent shares parameters with the Reasoning Agent, while using a
        # separate generation call and trajectory. The trainer later computes its
        # GiGPO advantages and includes these rows in the same actor optimizer update.
        skill_obs = {"text": prompts, "image": None, "anchor": None}
        skill_batch = self.preprocess_batch(gen_batch=gen_batch, obs=skill_obs)
        skill_non_tensor_keys = ["raw_prompt_ids"]
        if "raw_prompt" in skill_batch.non_tensor_batch:
            skill_non_tensor_keys.append("raw_prompt")
        if "tools_kwargs" in skill_batch.non_tensor_batch:
            skill_non_tensor_keys.append("tools_kwargs")
        skill_input = skill_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=skill_non_tensor_keys,
        )
        skill_input.meta_info = gen_batch.meta_info
        padded_input, pad_size = pad_dataproto_to_divisor(skill_input, actor_rollout_wg.world_size)
        padded_output = actor_rollout_wg.generate_sequences(padded_input)
        skill_output = unpad_dataproto(padded_output, pad_size=pad_size)
        # Match the Reasoning-Agent grouping so group-relative advantages can be
        # computed for the editor policy as well.
        skill_batch.non_tensor_batch["uid"] = uid_batch
        skill_batch.non_tensor_batch["traj_uid"] = traj_uid
        skill_batch = skill_batch.union(skill_output)
        skill_responses = self.tokenizer.batch_decode(skill_output.batch["responses"], skip_special_tokens=True)

        records: List[Dict[str, Any]] = []
        for i, response in enumerate(skill_responses):
            decision = _parse_skill_agent_decision(response)
            records.append({
                "step": int(step),
                "uid": str(uid_batch[i]),
                "traj_uid": str(traj_uid[i]),
                "active": bool(active_masks[i]),
                "state": str(current_states[i]),
                "next_state": str(next_states[i]),
                "active_step_skills": _json_safe(active_step_skills[i]),
                "reasoning_history": _json_safe(reasoning_history[i][-max_history_steps:] if max_history_steps > 0 else []),
                "reasoning_action": text_actions[i],
                "environment_feedback": feedbacks[i],
                "skill_agent_prompt": prompts[i],
                "skill_agent_response": response,
                "decision": decision,
                # Deliberately false in this first integration: the proposal is recorded,
                # not written back into SkillsOnlyMemory yet.
                "applied_to_skill_bank": False,
            })
        skill_batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
        skill_batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)
        skill_batch.non_tensor_batch["skill_agent_action"] = np.asarray(
            [record["decision"]["action"] for record in records], dtype=object
        )
        skill_batch.non_tensor_batch["skill_agent_target_skill_id"] = np.asarray(
            [record["decision"]["target_skill_id"] for record in records], dtype=object
        )
        skill_batch.non_tensor_batch["skill_agent_step"] = np.full(batch_size, step, dtype=np.int32)
        skill_batch.non_tensor_batch["agent_role"] = np.full(batch_size, "skill", dtype=object)
        return records, skill_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            per_step_retrieved: Optional[List[List[Dict]]] = None,
            skill_agent_trajectories: Optional[List[List[Dict]]] = None,
            envs: Optional[Any] = None,
            enable_dynamic_management: bool = False,
            with_skills_per_traj: Optional[np.ndarray] = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        wsm_arr = with_skills_per_traj
        if wsm_arr is None and envs is not None:
            _w = getattr(envs, "with_skills_mask", None)
            wsm_arr = np.asarray(_w, dtype=bool).copy() if _w is not None else None
        if wsm_arr is not None:
            wsm_arr = np.asarray(wsm_arr, dtype=bool).ravel()

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        # A/B rollout: split success for skill vs no-skill arms (logged as episode/success_rate_* via metric_utils)
        if (
            wsm_arr is not None
            and "success_rate" in success
            and wsm_arr.shape[0] == batch_size
        ):
            st = np.asarray(success["success_rate"], dtype=np.float64).ravel()
            if st.shape[0] == batch_size:
                skill_vals = st[wsm_arr]
                origin_vals = st[~wsm_arr]
                success_rate["success_rate_skill"] = np.array(
                    [float(np.mean(skill_vals)) if skill_vals.size > 0 else float("nan")],
                    dtype=np.float32,
                )
                success_rate["success_rate_origin"] = np.array(
                    [float(np.mean(origin_vals)) if origin_vals.size > 0 else float("nan")],
                    dtype=np.float32,
                )

        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                # Meta-attempt Reasoning rows use one GiGPO trajectory id per
                # attempt so discounted step returns cannot cross episode
                # boundaries. The outer rollout and Skill-Agent batch retain
                # the stable base trajectory id.
                row_base_traj_uid = data.get("base_traj_uid", data["traj_uid"])
                assert traj_uid[bs] == row_base_traj_uid, "data is not from the same trajectory"
                if data['active_masks']:
                    if 'ppo_episode_reward' in data:
                        # Meta-attempt rows carry a role-specific PPO return.
                        # Keep the usual rollout total as an audit field, but do
                        # not feed later-attempt rewards back into Attempt 0
                        # through the generic EpisodeRewardManager.
                        data['rollout_episode_rewards'] = episode_rewards[bs]
                        ppo_episode_reward = np.float32(data['ppo_episode_reward'])
                        ppo_episode_length = np.float32(
                            data.get('ppo_episode_length', episode_lengths[bs])
                        )
                        # collate_fn deliberately stores non-tensor values in an
                        # object array.  Use NumPy scalars here, just like the
                        # ordinary rollout branch below, so reductions in the
                        # unchanged VERL metric code still return values with
                        # an `.item()` method.
                        data['ppo_episode_reward'] = ppo_episode_reward
                        data['ppo_episode_length'] = ppo_episode_length
                        data['episode_rewards'] = ppo_episode_reward
                        # An attempt-local length keeps optional reward
                        # normalization aligned with the return above.
                        data['episode_lengths'] = ppo_episode_length
                    else:
                        data['episode_rewards'] = episode_rewards[bs]
                        data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    # trajectory index for intrinsic reward / utility (stable after balance_batch)
                    data['traj_index'] = bs

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        # Per-step retrieval for recording only: always trajectory-level (len = num_trajectories).
        # Trainer will pop this before adjust_batch so it never causes length mismatch.
        if per_step_retrieved is not None:
            # Must store as an object array with shape (n_traj,), where each cell
            # is one trajectory-level list[dict]. If we directly call np.array on
            # list_of_lists and all trajectories share the same step count, NumPy
            # creates a 2D (n_traj, L) matrix. A later .ravel() would flatten by
            # step instead of by trajectory, and JSON records would keep only the
            # global i-th step for sample_i.
            n_ps = len(per_step_retrieved)
            _ps_store = np.empty(n_ps, dtype=object)
            for _ii in range(n_ps):
                _ps_store[_ii] = list(per_step_retrieved[_ii])
            gen_batch_output.non_tensor_batch["per_step_retrieved_for_record"] = _ps_store
        if skill_agent_trajectories is not None:
            n_sa = len(skill_agent_trajectories)
            _sa_store = np.empty(n_sa, dtype=object)
            for _ii in range(n_sa):
                _sa_store[_ii] = list(skill_agent_trajectories[_ii])
            # This is deliberately trajectory-level. The trainer records and removes it
            # before balancing the row-level Reasoning/Skill PPO batches.
            gen_batch_output.non_tensor_batch["skill_agent_trajectory_for_record"] = _sa_store

        # When dynamic management is on: add trajectory-derived keys **expanded to row-level**
        # so adjust_batch (select_idxs + concat) and balance_batch never see length mismatch.
        if enable_dynamic_management:
            traj_idx = np.asarray(gen_batch_output.non_tensor_batch.get("traj_index")).ravel().astype(np.int64)
            num_rows = len(traj_idx)
            if success and "success_rate" in success:
                st = np.asarray(success["success_rate"])
                gen_batch_output.non_tensor_batch["success_per_traj"] = st[traj_idx]
            if wsm_arr is not None and wsm_arr.shape[0] == batch_size:
                gen_batch_output.non_tensor_batch["with_skills_mask"] = wsm_arr[traj_idx]
            if per_step_retrieved is not None:
                gen_batch_output.non_tensor_batch["per_step_retrieved_by_traj"] = np.array(
                    [per_step_retrieved[int(traj_idx[i])] for i in range(num_rows)], dtype=object
                )
        return gen_batch_output

    def _meta_attempt_config(self) -> Dict[str, Any]:
        env_cfg = getattr(self.config, "env", {}) or {}
        skill_cfg = env_cfg.get("skill_agent") or {}
        meta_cfg = skill_cfg.get("meta_attempts") or {}
        return meta_cfg if meta_cfg.get("enabled", False) else {}

    def _validate_meta_attempt_config(self, envs: EnvironmentManagerBase, meta_cfg: Dict[str, Any]) -> None:
        som_cfg = (self.config.env.get("skills_only_memory") or {})
        if not self.config.env.get("use_skills_only_memory", False):
            raise ValueError("skill_agent.meta_attempts requires env.use_skills_only_memory=True")
        if (som_cfg.get("skill_gen_mode") or "task_step").strip().lower() != "task_step":
            raise ValueError("skill_agent.meta_attempts currently requires skills_only_memory.skill_gen_mode=task_step")
        if int(som_cfg.get("top_k_task", som_cfg.get("top_k", 1))) != 1:
            raise ValueError("skill_agent.meta_attempts requires skills_only_memory.top_k_task=1")
        if getattr(envs, "retrieval_memory", None) is None:
            raise ValueError("skill_agent.meta_attempts requires SkillsOnlyMemory")
        if getattr(envs.retrieval_memory, "retrieval_mode", None) != "embedding":
            raise ValueError("skill_agent.meta_attempts requires embedding retrieval")
        if not som_cfg.get("embedding_model_path"):
            raise ValueError("skill_agent.meta_attempts requires skills_only_memory.embedding_model_path for local overlay retrieval")
        # The global bank may use a remote embedding service during Attempt 0.
        # EpisodeSkillOverlay deliberately creates a separate SkillsOnlyMemory
        # without that URL, so Attempts 1/2 always retrieve from the local,
        # trajectory-private edited bundle.
        if not hasattr(envs, "restart_attempt") or envs.__class__.restart_attempt is EnvironmentManagerBase.restart_attempt:
            raise ValueError(f"{envs.__class__.__name__} does not implement meta-attempt restart")
        management_cfg = som_cfg.get("management") or {}
        if management_cfg.get("baseline_ab_split", False):
            raise ValueError("skill_agent.meta_attempts is incompatible with baseline A/B skill split")
        if int(meta_cfg.get("num_attempts", 3)) < 2:
            raise ValueError("skill_agent.meta_attempts.num_attempts must be >= 2")

    def _annotate_meta_attempt_returns(
        self,
        total_batch_list: List[List[Dict]],
        total_skill_agent_batch_list: List[List[Dict]],
        attempt_rewards: np.ndarray,
    ) -> None:
        """Assign Reasoning outcomes and Skill-Agent episode improvement."""
        annotate_meta_attempt_returns(
            total_batch_list,
            total_skill_agent_batch_list,
            attempt_rewards,
        )

    def _meta_attempt_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        meta_cfg: Dict[str, Any],
    ):
        """First attempt edits a private skill overlay; later attempts evaluate it."""
        self._validate_meta_attempt_config(envs, meta_cfg)
        num_attempts = int(meta_cfg.get("num_attempts", 3))
        skill_cfg = (self.config.env.get("skill_agent") or {})
        skill_history_steps = max(0, int(skill_cfg.get("max_history_steps", 2)))
        som_cfg = self.config.env.get("skills_only_memory") or {}
        batch_size = len(gen_batch.batch)
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))
        if len(gen_batch.batch) != len(obs["text"]):
            raise ValueError("meta-attempt rollout requires gen_batch and environment batch sizes to match")
        initial_retrieved = deepcopy(getattr(envs, "retrieved_memories", None))
        if initial_retrieved is None:
            raise ValueError("meta-attempt rollout requires initial task-skill retrieval")
        task_step_counts = {}
        for memories in initial_retrieved:
            for task_skill in memories.get("task_skills", []):
                skill_id = str(task_skill.get("skill_id") or "<missing-id>")
                child_ids = {sid for sid in task_skill.get("step_skill_ids", []) if sid}
                task_step_counts[skill_id] = len(child_ids)
        print(
            "[MetaAttempt][Attempt 0] Retrieved task-skill child step counts: "
            + json.dumps(task_step_counts, ensure_ascii=False, sort_keys=True)
        )
        # Attempt 0 may use a remote service for global retrieval, while private
        # overlays must retrieve locally in Attempts 1/2. Load one local encoder
        # here and share that read-only instance across every trajectory overlay.
        shared_overlay_embedding_model = envs.retrieval_memory._get_embedding_model(log_loading=False)
        shared_child_embedding_cache = {}
        overlays = [
            EpisodeSkillOverlay(
                envs.retrieval_memory,
                mem,
                shared_embedding_model=shared_overlay_embedding_model,
                shared_child_embedding_cache=shared_child_embedding_cache,
            )
            for mem in initial_retrieved
        ]

        rollout_n = int(getattr(self.config.env.rollout, "n", 0) or 0)
        groups = [str(uuid.uuid4()) for _ in range((batch_size + max(rollout_n, 1) - 1) // max(rollout_n, 1))]
        uid_batch = np.asarray([groups[i // max(rollout_n, 1)] for i in range(batch_size)], dtype=object)
        base_traj_uid = np.asarray([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_skill_agent_batch_list = [[] for _ in range(batch_size)]
        skill_agent_trajectories = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        per_step_retrieved = [[] for _ in range(batch_size)]
        reasoning_history = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        attempt_rewards = np.zeros((batch_size, num_attempts), dtype=np.float32)
        attempt_successes = np.zeros((batch_size, num_attempts), dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            per_step_retrieved[i].append({
                "attempt_idx": 0,
                "step": 0,
                **_snapshot_retrieved_memories(initial_retrieved[i], som_cfg.get("skill_text_for_retrieval", "full")),
            })

        for attempt_idx in range(num_attempts):
            reasoning_traj_uid = np.asarray(
                [
                    build_reasoning_attempt_traj_uid(uid, attempt_idx)
                    for uid in base_traj_uid
                ],
                dtype=object,
            )
            if attempt_idx == 1:
                for overlay, edits in zip(overlays, skill_agent_trajectories):
                    for edit, applied in zip(edits, overlay.apply_all(edits)):
                        edit["applied_to_skill_overlay"] = applied
                envs.set_episode_skill_overlays(overlays)
            if attempt_idx >= 1:
                obs, infos = envs.restart_attempt()
                for i in range(batch_size):
                    per_step_retrieved[i].append({
                        "attempt_idx": attempt_idx,
                        "step": 0,
                        **_snapshot_retrieved_memories(
                            envs.retrieved_memories[i], som_cfg.get("skill_text_for_retrieval", "full")
                        ),
                    })
            is_done = np.zeros(batch_size, dtype=bool)
            phase = "edit_play" if attempt_idx == 0 else "eval_play"
            skill_version = "original" if attempt_idx == 0 else "edited"

            for turn_idx in range(self.config.env.max_steps):
                active_masks = np.logical_not(is_done)
                retrieved = getattr(envs, "retrieved_memories", None)
                active_step_skills = [
                    _json_safe((retrieved[i] if retrieved is not None else {}).get("step_skills", []))
                    for i in range(batch_size)
                ]
                batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
                non_tensor_keys = ["raw_prompt_ids"]
                for key in ("multi_modal_data", "raw_prompt", "tools_kwargs"):
                    if key in batch.non_tensor_batch:
                        non_tensor_keys.append(key)
                batch_input = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=non_tensor_keys,
                )
                batch_input.meta_info = gen_batch.meta_info
                padded_input, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
                batch_output = unpad_dataproto(actor_rollout_wg.generate_sequences(padded_input), pad_size=pad_size)
                batch.non_tensor_batch.update({
                    "uid": uid_batch,
                    # GiGPO discounts immediate rewards within a traj_uid.
                    # Keep attempts independent while sharing uid so every
                    # rollout x attempt remains in the same task-level group.
                    "traj_uid": reasoning_traj_uid,
                    "base_traj_uid": base_traj_uid,
                    "attempt_idx": np.full(batch_size, attempt_idx, dtype=np.int32),
                    "turn_idx": np.full(batch_size, turn_idx, dtype=np.int32),
                    "phase": np.full(batch_size, phase, dtype=object),
                    "skill_version": np.full(batch_size, skill_version, dtype=object),
                })
                batch = batch.union(batch_output)
                text_actions = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                next_obs, rewards, dones, infos = envs.step(text_actions)
                if len(rewards.shape) == 2:
                    rewards = rewards.squeeze(1)
                if len(dones.shape) == 2:
                    dones = dones.squeeze(1)

                if attempt_idx == 0:
                    skill_records, skill_batch = self._generate_skill_agent_trajectory_step(
                        gen_batch, actor_rollout_wg, obs, next_obs, active_step_skills,
                        reasoning_history, text_actions, rewards, dones, infos, active_masks,
                        uid_batch, base_traj_uid, turn_idx, skill_history_steps,
                    )
                    skill_batch.non_tensor_batch["attempt_idx"] = np.zeros(batch_size, dtype=np.int32)
                    skill_batch.non_tensor_batch["phase"] = np.full(batch_size, "edit", dtype=object)
                    skill_batch.non_tensor_batch["skill_version"] = np.full(batch_size, "original", dtype=object)
                    # The Skill-Agent prompt has no environment anchor of its
                    # own. Copy the paired pre-action Reasoning observation so
                    # post-rollout GiGPO-style state grouping is exact.
                    skill_anchor_obs = np.empty(batch_size, dtype=object)
                    reasoning_anchor_obs = batch.non_tensor_batch.get("anchor_obs", [None] * batch_size)
                    for i in range(batch_size):
                        skill_anchor_obs[i] = deepcopy(reasoning_anchor_obs[i])
                    skill_batch.non_tensor_batch["anchor_obs"] = skill_anchor_obs
                    skill_batch_list = to_list_of_dict(skill_batch)
                    for i in range(batch_size):
                        if active_masks[i]:
                            skill_records[i]["anchor_obs"] = _json_safe(skill_anchor_obs[i])
                            skill_records[i]["attempt_idx"] = 0
                            skill_records[i]["phase"] = "edit"
                            skill_records[i]["skill_version"] = "original"
                            skill_agent_trajectories[i].append(skill_records[i])
                            # Do not give discount credit to padding rows generated
                            # after this environment has already terminated.
                            total_skill_agent_batch_list[i].append(skill_batch_list[i])

                if getattr(envs, "retrieved_memories", None) is not None:
                    for i in range(batch_size):
                        per_step_retrieved[i].append({
                            "attempt_idx": attempt_idx,
                            "step": turn_idx + 1,
                            **_snapshot_retrieved_memories(envs.retrieved_memories[i], som_cfg.get("skill_text_for_retrieval", "full")),
                        })
                batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
                batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)
                batch.non_tensor_batch["is_action_valid"] = np.asarray(
                    [info.get("is_action_valid", True) for info in infos], dtype=bool
                )
                batch.non_tensor_batch["env_won"] = np.asarray(
                    [float(info.get("won", 0.0) or 0.0) for info in infos], dtype=np.float32
                )
                batch_list = to_list_of_dict(batch)
                for i in range(batch_size):
                    if active_masks[i]:
                        total_batch_list[i].append(batch_list[i])
                        total_infos[i].append(infos[i])
                        attempt_rewards[i, attempt_idx] += float(rewards[i])
                        attempt_successes[i, attempt_idx] = max(
                            attempt_successes[i, attempt_idx],
                            float(infos[i].get("won", 0.0) or 0.0),
                        )
                        episode_rewards[i] += float(rewards[i])
                        episode_lengths[i] += 1
                        if attempt_idx == 0:
                            raw_states = obs.get("anchor")
                            if raw_states is None:
                                raw_states = [""] * batch_size
                            raw_state = raw_states[i]
                            reasoning_history[i].append({
                                "step": turn_idx,
                                "state": str(raw_state),
                                "reasoning_action": text_actions[i],
                                "environment_feedback": _compact_env_feedback(infos[i], rewards[i], dones[i]),
                            })
                        tool_callings[i] += float(infos[i].get("tool_calling", 0) or 0)
                is_done = np.logical_or(is_done, dones)
                obs = next_obs
                if is_done.all():
                    break

        self._annotate_meta_attempt_returns(
            total_batch_list, total_skill_agent_batch_list, attempt_rewards,
        )
        for i, overlay in enumerate(overlays):
            if skill_agent_trajectories[i]:
                baseline_episode_reward = float(attempt_rewards[i, 0])
                edited_episode_reward_mean = float(np.mean(attempt_rewards[i, 1:]))
                skill_agent_trajectories[i][0]["meta_attempt_overlay"] = overlay.snapshot()
                skill_agent_trajectories[i][0]["attempt_rewards"] = attempt_rewards[i].tolist()
                skill_agent_trajectories[i][0]["attempt_successes"] = attempt_successes[i].tolist()
                skill_agent_trajectories[i][0][
                    "skill_baseline_episode_reward"
                ] = baseline_episode_reward
                skill_agent_trajectories[i][0][
                    "skill_edited_episode_reward_mean"
                ] = edited_episode_reward_mean
                skill_agent_trajectories[i][0][
                    "skill_episode_improvement_reward"
                ] = edited_episode_reward_mean - baseline_episode_reward
        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        # Retrieval records above hold all information needed for audit. Clear the
        # manager pointer before returning so the next rollout cannot accidentally
        # reuse this trajectory's private overlay.
        envs.clear_episode_skill_overlays()
        return (
            total_batch_list, episode_rewards, episode_lengths, success, base_traj_uid,
            tool_callings, per_step_retrieved, skill_agent_trajectories,
            total_skill_agent_batch_list, getattr(envs, "with_skills_mask", None),
        )

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        meta_cfg = self._meta_attempt_config() if is_train else {}
        if meta_cfg:
            return self._meta_attempt_multi_turn_loop(gen_batch, actor_rollout_wg, envs, meta_cfg)

        batch_size = len(gen_batch.batch)
        som_cfg = getattr(getattr(self.config, "env", None), "skills_only_memory", None) or {}
        # Collect per-step retrievals when using step-level skills (step_only or task_step); envs.retrieved_memories
        # is set in reset()/step() using that step's task+obs, so we snapshot after each step.
        mode = (som_cfg.get("skill_gen_mode") or "task_step").strip().lower()
        if mode not in ("task_only", "step_only", "task_step"):
            mode = "task_step"
        collect_per_step = (
            getattr(envs, "retrieval_memory", None) is not None
            and mode in ("step_only", "task_step")
        )
        per_step_retrieved: Optional[List[List[Dict]]] = [[] for _ in range(batch_size)] if collect_per_step else None
        env_cfg = getattr(self.config, "env", {}) or {}
        skill_agent_cfg = env_cfg.get("skill_agent") or {}
        enable_skill_agent = is_train and bool(skill_agent_cfg.get("enabled", False))
        skill_agent_history_steps = max(0, int(skill_agent_cfg.get("max_history_steps", 2)))
        skill_agent_trajectories: Optional[List[List[Dict]]] = (
            [[] for _ in range(batch_size)] if enable_skill_agent else None
        )

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        skill_text_mode = som_cfg.get("skill_text_for_retrieval", "full")
        if collect_per_step and envs.retrieved_memories is not None:
            for i in range(batch_size):
                per_step_retrieved[i].append({"step": 0, **_snapshot_retrieved_memories(envs.retrieved_memories[i], skill_text_mode)})
        
        # uid = one per "problem" group; traj_uid = one per trajectory. So group_size trajectories share one uid.
        rollout_n = int(getattr(self.config.env.rollout, 'n', 0) or 0) if is_train else 1
        if rollout_n > 0:
            # Same uid for consecutive rollout_n indices (align with repeat interleave: traj 0..n-1 = group 0)
            num_groups = (batch_size + rollout_n - 1) // rollout_n
            group_uuids = [str(uuid.uuid4()) for _ in range(num_groups)]
            uid_batch = np.array([group_uuids[i // rollout_n] for i in range(batch_size)], dtype=object)
            if batch_size <= 64:  # only log when batch is small enough
                n_unique = len(set(uid_batch.tolist()))
                print(f"[Rollout] uid grouping: batch_size={batch_size}, rollout.n={rollout_n}, unique uids={n_unique} (expected {num_groups})")
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        total_skill_agent_batch_list: Optional[List[List[Dict]]] = (
            [[] for _ in range(batch_size)] if enable_skill_agent else None
        )
        reasoning_history: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            # Snapshot before env.step(): these are the step skills that were visible
            # to the Reasoning Agent for its current action, not the next-step retrieval.
            retrieved_memories = getattr(envs, "retrieved_memories", None)
            active_step_skills = [
                _json_safe((retrieved_memories[i] if retrieved_memories is not None else {}).get("step_skills", []))
                for i in range(batch_size)
            ]

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            if collect_per_step and envs.retrieved_memories is not None:
                for i in range(batch_size):
                    per_step_retrieved[i].append({
                        "step": _step + 1,
                        **_snapshot_retrieved_memories(envs.retrieved_memories[i], skill_text_mode),
                    })
            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if enable_skill_agent:
                skill_records, skill_batch = self._generate_skill_agent_trajectory_step(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    obs=obs,
                    next_obs=next_obs,
                    active_step_skills=active_step_skills,
                    reasoning_history=reasoning_history,
                    text_actions=text_actions,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                    active_masks=active_masks,
                    uid_batch=uid_batch,
                    traj_uid=traj_uid,
                    step=_step,
                    max_history_steps=skill_agent_history_steps,
                )
                skill_batch_list = to_list_of_dict(skill_batch)
                for i in range(batch_size):
                    if active_masks[i]:
                        skill_agent_trajectories[i].append(skill_records[i])
                    total_skill_agent_batch_list[i].append(skill_batch_list[i])

            # The Skill Agent receives a compact history of completed reasoning
            # transitions; the current transition is available directly in its prompt.
            for i in range(batch_size):
                if active_masks[i]:
                    reasoning_history[i].append({
                        "step": int(_step),
                        "state": str((obs.get("text") or [""] * batch_size)[i]),
                        "reasoning_action": text_actions[i],
                        "environment_feedback": _compact_env_feedback(infos[i], rewards[i], dones[i]),
                    })

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
            batch.non_tensor_batch['env_won'] = np.asarray(
                [float(info.get('won', 0.0) or 0.0) for info in infos], dtype=np.float32
            )

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        wsm_traj = getattr(envs, "with_skills_mask", None)
        if wsm_traj is not None:
            wsm_traj = np.asarray(wsm_traj, dtype=bool).copy()

        return (
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
            per_step_retrieved,
            skill_agent_trajectories,
            total_skill_agent_batch_list,
            wsm_traj,
        )
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_wsm_chunks: List[np.ndarray] = []
        total_skill_agent_trajectories: List[List[Dict]] = []
        total_skill_agent_batch_list: List[List[Dict]] = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, _, skill_agent_trajectories, skill_agent_batch_list, wsm_traj = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, wsm_traj, keep_indices = filter_group_data(
                batch_list=batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                config=self.config,
                last_try=(try_count == max_try_count),
                with_skills_per_traj=wsm_traj,
                return_keep_indices=True,
            )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            if skill_agent_trajectories is not None:
                total_skill_agent_trajectories += [skill_agent_trajectories[i] for i in keep_indices]
            if skill_agent_batch_list is not None:
                total_skill_agent_batch_list += [skill_agent_batch_list[i] for i in keep_indices]
            if wsm_traj is not None:
                total_wsm_chunks.append(np.asarray(wsm_traj, dtype=bool))

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)
        total_wsm = np.concatenate(total_wsm_chunks, axis=0) if total_wsm_chunks else None

        return (
            total_batch_list,
            total_episode_rewards,
            total_episode_lengths,
            total_success,
            total_traj_uid,
            total_tool_callings,
            total_skill_agent_trajectories or None,
            total_skill_agent_batch_list or None,
            total_wsm,
        )

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        else:
            val_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1) or 1)
            if val_n != 1:
                raise ValueError(
                    "single-rollout validation requires actor_rollout_ref.rollout.val_kwargs.n=1"
                )
            if callable(getattr(getattr(envs, "envs", None), "set_active_num_processes", None)):
                envs.set_active_batch_size(len(gen_batch.batch))
            
        # Initial observations from the environment
        per_step_retrieved = None
        skill_agent_trajectories = None
        skill_agent_batch_list = None
        total_wsm: Optional[np.ndarray] = None
        if self.config.algorithm.filter_groups.enable and is_train:
            if self._meta_attempt_config():
                raise ValueError("skill_agent.meta_attempts is incompatible with algorithm.filter_groups.enable")
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, skill_agent_trajectories, skill_agent_batch_list, total_wsm = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            per_step_retrieved = None
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, per_step_retrieved, skill_agent_trajectories, skill_agent_batch_list, total_wsm = \
                self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                    is_train=is_train,
                )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        som_cfg = (self.config.env.get("skills_only_memory") or {}) if hasattr(self.config, "env") else {}
        enable_dynamic_management = bool(som_cfg.get("enable_dynamic_management", False))
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            per_step_retrieved=per_step_retrieved,
            skill_agent_trajectories=skill_agent_trajectories,
            envs=envs,
            enable_dynamic_management=enable_dynamic_management,
            with_skills_per_traj=total_wsm,
        )
        if skill_agent_batch_list is not None:
            # Keep a second, tensor-complete PPO batch rather than putting editor
            # prompts into the reasoning batch (their sequence lengths differ).
            skill_agent_rl_batch = self.gather_rollout_data(
                total_batch_list=skill_agent_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
                with_skills_per_traj=total_wsm,
            )
            skill_agent_rl_batch.meta_info["agent_role"] = "skill"
            if self._meta_attempt_config():
                skill_agent_rl_batch.meta_info["reward_source"] = "post_edit_episode_improvement"
                skill_agent_rl_batch.meta_info["num_attempts"] = int(self._meta_attempt_config().get("num_attempts", 3))
            else:
                skill_agent_rl_batch.meta_info["reward_source"] = "environment_transition"
            gen_batch_output.meta_info["skill_agent_rl_batch"] = skill_agent_rl_batch
        
        return gen_batch_output
