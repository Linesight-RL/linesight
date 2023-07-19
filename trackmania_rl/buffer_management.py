import random

import numpy as np
from torchrl.data import ReplayBuffer

from . import misc
from .experience_replay.experience_replay_interface import Experience


def fill_buffer_from_rollout_with_n_steps_rule(
    buffer: ReplayBuffer,
    buffer_test: ReplayBuffer,
    rollout_results: dict,
    n_steps_max: int,
    gamma: float,
    discard_non_greedy_actions_in_nsteps: bool,
    n_zone_centers_in_inputs: int,
    zone_centers,
):
    Start_Time = time.perf_counter()
    assert n_zone_centers_in_inputs >= 2

    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])

    number_memories_added = 0
    buffer_to_fill = buffer_test if random.random() < misc.buffer_test_ratio else buffer

    gammas = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(
        np.float32
    )  # Discount factor that will be placed in front of next_step in Bellman equation, depending on n_steps chosen

    reward_into = np.zeros(n_frames)
    for i in range(1,n_frames):
        reward_into[i] += misc.constant_reward_per_ms * misc.ms_per_action
        reward_into[i] += (rollout_results["meters_advanced_along_centerline"][i] - rollout_results["meters_advanced_along_centerline"][i-1]) * misc.reward_per_m_advanced_along_centerline
        reward_into[i] += rollout_results["input_w"][i-1] * misc.reward_per_ms_press_forward_early_training * misc.ms_per_action

    for i in range(n_frames - 1):# Loop over all frames that were generated
        # Switch memory buffer sometimes
        if random.random() < 0.1:
            buffer_to_fill = buffer_test if random.random() < misc.buffer_test_ratio else buffer

        n_steps = min(n_steps_max,n_frames-1-i)
        if discard_non_greedy_actions_in_nsteps:
            try:
                first_non_greedy = rollout_results["action_was_greedy"][i + 1 : i + n_steps].index(False) + 1
                n_steps = min(n_steps,first_non_greedy)
            except ValueError:
                pass

        rewards = np.empty(n_steps_max).astype(np.float32)
        for j in range(n_steps):
            rewards[j] = reward_into[i+j+1] + (gamma*rewards[j-1] if j>=1 else 0)

        state_img = rollout_results["frames"][i]
        state_float = rollout_results["state_float"][i]

        # Get action that was played
        action = rollout_results["actions"][i]

        if i + n_steps < n_frames - 1:
            # next_step is not the final frame of the rollout, we do not need to force the transition to be final in the minirace
            minirace_min_time_actions = 0

            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = rollout_results["state_float"][i + n_steps]
        else:
            # next_step is the final frame of the rollout, we do need to force the transition to be final in the minirace
            minirace_min_time_actions = misc.temporal_mini_race_duration_actions - n_steps
            # It doesn't matter what next_state_img and next_state_float contain, as the transition will be forced to be final
            next_state_img = state_img
            next_state_float = state_float

        buffer_to_fill.add(
            Experience(
                state_img,
                state_float,
                action,
                n_steps,
                rewards,
                next_state_img,
                next_state_float,
                gammas,
                minirace_min_time_actions,
            ),
        )
        number_memories_added += 1

    return buffer, buffer_test, number_memories_added
