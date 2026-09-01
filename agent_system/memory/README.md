## Memory Manager

<p align="center">
    <img src="../../docs/gigpo/framework-comparison.png" alt="framework" width="100%">
</p>

`verl-agent` allows for flexibly choosing what history to include for each step, such as, recent steps, key events, summaries, or external knowledge.

We provide a simplest memory implementation as a starting point. Developers are encouraged to extend this module with custom memory strategies, such as dynamic summarization, selective memory retention, or external knowledge integration, to improve the handling of long-horizon interaction histories.

### Skill Agent meta-attempt evaluation

To evaluate online Skill-Agent edits without mutating the shared bank, enable the
three-attempt rollout mode below. Attempt 0 collects Reasoning-Agent actions and
Skill-Agent edits; Attempts 1 and 2 replay the same environment task with a
per-trajectory edited overlay. The overlay uses a local embedding model; it is
discarded unless its two evaluation attempts improve on Attempt 0, in which case
the final bundle is cloned into the global bank as an independent version.

```yaml
algorithm:
  adv_estimator: gigpo
  gamma: 0.95
  gigpo:
    step_advantage_w: 1.0
    mode: mean_std_norm
    enable_similarity: false

env:
  use_skills_only_memory: true
  skill_agent:
    enabled: true
    training:
      enabled: true
      adv_estimator: gigpo
      gamma: 0.95
      step_advantage_w: 1.0
    meta_attempts:
      enabled: true
      num_attempts: 3
  skills_only_memory:
    skill_gen_mode: task_step
    retrieval_mode: embedding
    embedding_model_path: /path/to/local/embedding-model
    # Do not set skill_retrieval_service_url in this mode.
    top_k_task: 1
    enable_dynamic_management: true
    management:
      baseline_ab_split: false
```

`algorithm.filter_groups.enable` and `management.baseline_ab_split` must also
remain disabled. The reasoning batch contains all three attempts with
`attempt_idx`, `phase`, and `skill_version`; the separate Skill-Agent batch only
contains Attempt 0. Its episode-improvement reward is the mean environment
return of Attempts 1 and 2 minus the Attempt-0 return. Every editor response in
the trajectory receives that same outcome; the paired Attempt-0 immediate
environment reward is retained as `skill_agent_env_reward` for audit.

When `skill_agent.training.enabled=true`, the trainer computes a separate GiGPO
advantage for these editor rows. As in verl-agent, intermediate editor rewards
are zero and the episode-improvement reward appears once on the terminal edit;
discounting that sparse reward supplies the exact-observation step-group
signal. Episode statistics keep verl-agent's default cross-step behavior.
Reasoning and Skill-Agent rows both use the verl-agent GiGPO implementation and
are interleaved for one shared actor update, so the optimizer and learning-rate
scheduler still advance exactly once per global step. Reasoning attempts share
the task-level `uid`, but each attempt has a distinct `traj_uid`; this keeps all
rollouts and attempts in the same relative-advantage group while preventing
discounted step returns from crossing attempt boundaries. Disabling Skill-Agent
training preserves Reasoning-only GiGPO updates while still recording the
Skill-Agent batch.

### Generate and load the initial hierarchical bank

Initial skill generation is an offline step and is not part of PPO training.
The standalone AlfWorld generator samples multiple attempts per task through an
OpenAI-compatible API, gives every rollout in each task group to the reflection
model, and asks it to create one task skill plus grounded child step skills.
Groups no longer need a balanced mixture of successful and failed attempts. It
writes both a directly loadable skill JSON and a reflection audit.

```bash
export OPENAI_API_KEY=...
python3 initial_skill_bank/generate_alfworld_initial_skill_bank.py \
  --api-base-url https://your-api.example/v1 \
  --model your-model \
  --target-task-bundles 300
```

Normal training only loads that file; it never collects initialization
trajectories or calls initialization reflection from `RayPPOTrainer.fit()`.

```yaml
env:
  use_skills_only_memory: true
  skills_only_memory:
    skills_json_path: initial_skill_bank/skill_bank/alfworld_gpt-5.5-2026-04-24_300.json
    load_initial_skills: true
    enable_dynamic_update: false
    management:
      eviction_enabled: true
      eviction_interval: 5
      eviction_max_task_bundles: 300
      eviction_protect_recent_steps: 10
```

After the generated bank is loaded, only Skill-Agent overlays whose two
evaluation attempts improve on Attempt 0 are promoted. Promoted bundles receive
independent task and child IDs. Capacity eviction removes complete task bundles
by retrieval frequency, last retrieval step, and creation step.
