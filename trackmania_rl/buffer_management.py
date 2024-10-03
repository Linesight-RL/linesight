"""
This file's main entry point is the function fill_buffer_from_rollout_with_n_steps_rule().
Its main inputs are a rollout_results object (obtained from a GameInstanceManager object), and a buffer to be filled.
It reassembles the rollout_results object into transitions, as defined in /trackmania_rl/experience_replay/experience_replay_interface.py
"""

import math
import random

import numpy as np
from numba import jit
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl.experience_replay.experience_replay_interface import Experience
from trackmania_rl.reward_shaping import speedslide_quality_tarmac


@jit(nopython=True)
def get_potential(state_float):
    # https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    vector_vcp_to_vcp_further_ahead = state_float[65:68] - state_float[62:65]
    vector_vcp_to_vcp_further_ahead_normalized = vector_vcp_to_vcp_further_ahead / np.linalg.norm(vector_vcp_to_vcp_further_ahead)

    return (
        config_copy.shaped_reward_dist_to_cur_vcp
        * max(
            config_copy.shaped_reward_min_dist_to_cur_vcp,
            min(config_copy.shaped_reward_max_dist_to_cur_vcp, np.linalg.norm(state_float[62:65])),
        )
    ) + (config_copy.shaped_reward_point_to_vcp_ahead * (vector_vcp_to_vcp_further_ahead_normalized[2] - 1))


def fill_buffer_from_rollout_with_n_steps_rule(
    buffer: ReplayBuffer,
    buffer_test: ReplayBuffer,
    rollout_results: dict,
    n_steps_max: int,
    gamma: float,
    discard_non_greedy_actions_in_nsteps: bool,
    engineered_speedslide_reward: float,
    engineered_neoslide_reward: float,
    engineered_kamikaze_reward: float,
    engineered_close_to_vcp_reward: float,
):
    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])

    number_memories_added_train = 0
    number_memories_added_test = 0
    Experiences_For_Buffer = []
    Experiences_For_Buffer_Test = []
    list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

    gammas = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(
        np.float32
    )  # Discount factor that will be placed in front of next_step in Bellman equation, depending on n_steps chosen

    reward_into = np.zeros(n_frames)
    for i in range(1, n_frames):
        reward_into[i] += config_copy.constant_reward_per_ms * (
            config_copy.ms_per_action
            if (i < n_frames - 1 or ("race_time" not in rollout_results))
            else rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action
        )
        reward_into[i] += (
            rollout_results["meters_advanced_along_centerline"][i] - rollout_results["meters_advanced_along_centerline"][i - 1]
        ) * config_copy.reward_per_m_advanced_along_centerline
        if i < n_frames - 1:
            if config_copy.final_speed_reward_per_m_per_s != 0 and rollout_results["state_float"][i][58] > 0:
                # car has velocity *forward*
                reward_into[i] += config_copy.final_speed_reward_per_m_per_s * (
                    np.linalg.norm(rollout_results["state_float"][i][56:59]) - np.linalg.norm(rollout_results["state_float"][i - 1][56:59])
                )
            if engineered_speedslide_reward != 0 and np.all(rollout_results["state_float"][i][25:29]):
                # all wheels touch the ground
                reward_into[i] += engineered_speedslide_reward * max(
                    0.0,
                    1 - abs(speedslide_quality_tarmac(rollout_results["state_float"][i][56], rollout_results["state_float"][i][58]) - 1),
                )  # TODO : indices 25:29, 56 and 58 are hardcoded, this is bad....

            # lateral speed is higher than 2 meters per second
            reward_into[i] += (
                engineered_neoslide_reward if abs(rollout_results["state_float"][i][56]) >= 2.0 else 0
            )  # TODO : 56 is hardcoded, this is bad....
            # kamikaze reward
            if (
                engineered_kamikaze_reward != 0
                and rollout_results["actions"][i] <= 2
                or np.sum(rollout_results["state_float"][i][25:29]) <= 1
            ):
                reward_into[i] += engineered_kamikaze_reward
            if engineered_close_to_vcp_reward != 0:
                reward_into[i] += engineered_close_to_vcp_reward * max(
                    config_copy.engineered_reward_min_dist_to_cur_vcp,
                    min(config_copy.engineered_reward_max_dist_to_cur_vcp, np.linalg.norm(rollout_results["state_float"][i][62:65])),
                )
    for i in range(n_frames - 1):  # Loop over all frames that were generated
        # Switch memory buffer sometimes
        if random.random() < 0.1:
            list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

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
        state_potential = get_potential(rollout_results["state_float"][i])

        # Get action that was played
        action = rollout_results["actions"][i]
        terminal_actions = float((n_frames - 1) - i) if "race_time" in rollout_results else math.inf
        next_state_has_passed_finish = ((i + n_steps) == (n_frames - 1)) and ("race_time" in rollout_results)

        if not next_state_has_passed_finish:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = rollout_results["state_float"][i + n_steps]
            next_state_potential = get_potential(rollout_results["state_float"][i + n_steps])
        else:
            # It doesn't matter what next_state_img and next_state_float contain, as the transition will be forced to be final
            next_state_img = state_img
            next_state_float = state_float
            next_state_potential = 0

        list_to_fill.append(
            Experience(
                state_img,
                state_float,
                state_potential,
                action,
                n_steps,
                rewards,
                next_state_img,
                next_state_float,
                next_state_potential,
                gammas,
                terminal_actions,
            )
        )
    number_memories_added_train += len(Experiences_For_Buffer)
    if len(Experiences_For_Buffer) > 1:
        buffer.extend(Experiences_For_Buffer)
    elif len(Experiences_For_Buffer) == 1:
        buffer.add(Experiences_For_Buffer[0])
    number_memories_added_test += len(Experiences_For_Buffer_Test)
    if len(Experiences_For_Buffer_Test) > 1:
        buffer_test.extend(Experiences_For_Buffer_Test)
    elif len(Experiences_For_Buffer_Test) == 1:
        buffer_test.add(Experiences_For_Buffer_Test[0])

    return buffer, buffer_test, number_memories_added_train, number_memories_added_test
