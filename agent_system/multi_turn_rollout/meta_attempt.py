"""Pure accounting helpers for multi-attempt Skill-Agent rollouts."""


def build_reasoning_attempt_traj_uid(base_traj_uid, attempt_idx):
    """Return the GiGPO trajectory id for one Reasoning attempt.

    Every attempt is a separate environment episode for discounted step-return
    computation. The base id remains available for overlay, audit, and
    Skill-Agent bookkeeping.
    """
    return f"{base_traj_uid}:attempt:{int(attempt_idx)}"


def annotate_meta_attempt_returns(
    total_batch_list,
    total_skill_agent_batch_list,
    attempt_rewards,
):
    """Attach independent Reasoning outcomes and Skill-Agent episode improvement."""
    for traj_idx, steps in enumerate(total_batch_list):
        attempts = sorted({int(step.get("attempt_idx", 0)) for step in steps})
        for attempt_idx in attempts:
            attempt_steps = [step for step in steps if int(step.get("attempt_idx", 0)) == attempt_idx]
            attempt_return = float(attempt_rewards[traj_idx][attempt_idx])
            for step in attempt_steps:
                # Use an outcome-reward convention where every response
                # generated within one attempt receives that attempt's complete
                # environment return. Attempts remain independent; later
                # evaluation rewards are never propagated back into Attempt 0.
                step["attempt_return"] = attempt_return
                step["ppo_episode_reward"] = attempt_return
                step["ppo_episode_length"] = len(attempt_steps)

        skill_steps = total_skill_agent_batch_list[traj_idx]
        if not skill_steps:
            continue

        trajectory_attempt_rewards = [float(value) for value in attempt_rewards[traj_idx]]
        if len(trajectory_attempt_rewards) < 2:
            raise ValueError("Skill-Agent episode improvement requires at least two attempts")
        baseline_episode_reward = trajectory_attempt_rewards[0]
        edited_episode_reward_mean = (
            sum(trajectory_attempt_rewards[1:]) / len(trajectory_attempt_rewards[1:])
        )
        episode_improvement_reward = edited_episode_reward_mean - baseline_episode_reward

        for step_idx, step in enumerate(skill_steps):
            step["skill_agent_env_reward"] = step["rewards"]
            step["skill_baseline_episode_reward"] = baseline_episode_reward
            step["skill_edited_episode_reward_mean"] = edited_episode_reward_mean
            step["skill_episode_improvement_reward"] = episode_improvement_reward
            # Match verl-agent's sparse environment-reward convention: the
            # trajectory-level improvement appears once at the terminal edit,
            # while earlier edit steps have zero immediate reward. GiGPO then
            # discounts this terminal outcome backward for its step advantage.
            step["rewards"] = (
                episode_improvement_reward
                if step_idx == len(skill_steps) - 1
                else 0.0
            )
            step["ppo_episode_reward"] = episode_improvement_reward
            step["ppo_episode_length"] = len(skill_steps)
