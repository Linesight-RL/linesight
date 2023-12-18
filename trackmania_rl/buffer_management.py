import math
import random

import numpy as np
from torchrl.data import ReplayBuffer

from . import misc
from .experience_replay.experience_replay_interface import Experience
from .reward_shaping import speedslide_quality_tarmac


def fill_buffer_from_rollout_with_n_steps_rule(
    buffer: ReplayBuffer,
    buffer_test: ReplayBuffer,
    rollout_results: dict,
    n_steps_max: int,
    gamma: float,
    discard_non_greedy_actions_in_nsteps: bool,
    speedslide_reward: float,
):
    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])

    number_memories_added_train = 0
    number_memories_added_test = 0
    buffer_to_fill = buffer_test if random.random() < misc.buffer_test_ratio else buffer

    gammas = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(
        np.float32
    )  # Discount factor that will be placed in front of next_step in Bellman equation, depending on n_steps chosen

    reward_into = np.zeros(n_frames)
    for i in range(1, n_frames):
        reward_into[i] += misc.constant_reward_per_ms * (
            misc.ms_per_action
            if (i < n_frames - 1 or ("race_time" not in rollout_results))
            else rollout_results["race_time"] - (n_frames - 2) * misc.ms_per_action
        )
        reward_into[i] += (
            rollout_results["meters_advanced_along_centerline"][i] - rollout_results["meters_advanced_along_centerline"][i - 1]
        ) * misc.reward_per_m_advanced_along_centerline
        if i < n_frames - 1 and np.all(rollout_results["state_float"][i][25:29]):
            reward_into[i] += speedslide_reward * max(
                0, 1 - abs(speedslide_quality_tarmac(rollout_results["state_float"][i][56], rollout_results["state_float"][i][58]) - 1)
            )  # TODO : indices 25:29, 56 and 58 are hardcoded, this is bad....

    for i in range(n_frames - 1):  # Loop over all frames that were generated
        # Switch memory buffer sometimes
        if random.random() < 0.1:
            buffer_to_fill = buffer_test if random.random() < misc.buffer_test_ratio else buffer

        n_steps = min(n_steps_max, n_frames - 1 - i)
        if discard_non_greedy_actions_in_nsteps:
            try:
                first_non_greedy = rollout_results["action_was_greedy"][i + 1 : i + n_steps].index(False) + 1
                n_steps = min(n_steps, first_non_greedy)
            except ValueError:
                pass

        rewards = np.empty(n_steps_max).astype(np.float32)
        for j in range(n_steps):
            rewards[j] = (gamma**j) * reward_into[i + j + 1] + (rewards[j - 1] if j >= 1 else 0)

        state_img = rollout_results["frames"][i]
        state_float = rollout_results["state_float"][i]

        # Get action that was played
        action = rollout_results["actions"][i]
        terminal_actions = float((n_frames - 1) - i) if "race_time" in rollout_results else math.inf
        next_state_has_passed_finish = ((i + n_steps) == (n_frames - 1)) and ("race_time" in rollout_results)

        if not next_state_has_passed_finish:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = rollout_results["state_float"][i + n_steps]
        else:
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
                terminal_actions,
            ),
        )
        if buffer_to_fill is buffer:
            number_memories_added_train += 1
        else:
            number_memories_added_test += 1

    return buffer, buffer_test, number_memories_added_train, number_memories_added_test
