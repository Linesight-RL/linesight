import random

import numpy as np

from . import misc
from .experience_replay.basic_experience_replay import ReplayBuffer
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
    assert n_zone_centers_in_inputs >= 2

    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])

    number_memories_added = 0
    buffer_to_fill = buffer_test if random.random() < misc.buffer_test_ratio else buffer

    gammas = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(
        np.float32
    )  # Discount factor that will be placed in front of next_step in Bellman equation, depending on n_steps chosen

    for i in range(n_frames - 1):
        # Loop over all frames that were generated

        # Switch memory buffer sometimes
        if random.random() < 0.1:
            buffer_to_fill = buffer_test if random.random() < misc.buffer_test_ratio else buffer

        # Find the longest n-step transition we can use.
        for n_steps in range(n_steps_max, -1, -1):
            if i + n_steps < n_frames and (
                (not discard_non_greedy_actions_in_nsteps) or all(rollout_results["action_was_greedy"][i + 1 : i + n_steps])
            ):
                # We have enough frames
                # and there was no exploration action during the n_steps
                # ==> We can use that n_steps value
                break
        else:
            print("We were not able to find a suitable n-steps value")
            assert False

        assert n_steps >= 1

        rewards = np.zeros(n_steps_max).astype(np.float32)
        for j in range(1, 1 + n_steps):
            reward = 0
            # Not sure I'll use this, maybe to ""normalize"" q-values ?
            reward += np.sum(gammas[:j]) * misc.constant_reward_per_ms * misc.ms_per_action
            # Reward due to meters advanced
            reward += (
                np.sum(
                    gammas[:j]
                    * (
                        np.array(rollout_results["meters_advanced_along_centerline"][i + 1 : i + 1 + j])
                        - np.array(rollout_results["meters_advanced_along_centerline"][i : i + j])
                    )
                )
                * misc.reward_per_m_advanced_along_centerline
            )
            # # Reward due to speed
            # reward += (
            #     np.sum(gammas[:j] * np.array(rollout_results["display_speed"][i + 1 : i + 1 + j]))
            #     * misc.reward_per_ms_velocity
            #     * misc.ms_per_action
            # )
            # Reward due to press forward
            reward += (
                np.sum(gammas[:j] * np.array(rollout_results["input_w"][i : i + j])) * misc.reward_per_ms_press_forward * misc.ms_per_action
            )

            rewards[j - 1] = reward

        # Construct state description
        current_zone_idx = rollout_results["current_zone_idx"][i]
        car_position = rollout_results["car_position"][i]
        car_orientation = rollout_results["car_orientation"][i]
        car_velocity = rollout_results["car_velocity"][i]
        state_zone_center_coordinates_in_car_reference_system = car_orientation.T.dot(
            (zone_centers[current_zone_idx : current_zone_idx + n_zone_centers_in_inputs, :] - car_position).T
        ).T  # (n_zone_centers_in_inputs, 3)
        state_y_map_vector_in_car_reference_system = car_orientation.T.dot(np.array([0, 1, 0]))
        state_car_velocity_in_car_reference_system = car_orientation.T.dot(car_velocity)
        car_angular_speed = rollout_results["car_angular_speed"][i]
        previous_action = misc.inputs[misc.action_forward_idx if i == 0 else rollout_results["actions"][i - 1]]
        state_car_angular_velocity_in_car_reference_system = car_orientation.T.dot(car_angular_speed)
        assert state_zone_center_coordinates_in_car_reference_system.shape == (
            n_zone_centers_in_inputs,
            3,
        )
        assert len(state_y_map_vector_in_car_reference_system) == 3
        assert len(state_car_velocity_in_car_reference_system) == 3

        state_img = rollout_results["frames"][i]
        state_float = np.hstack(
            (
                0,  # Placeholder for mini_race_time_actions
                np.array(
                    [previous_action["accelerate"], previous_action["brake"], previous_action["left"], previous_action["right"]]
                ),  # NEW
                rollout_results["car_gear_and_wheels"][i].ravel(),  # NEW
                state_car_angular_velocity_in_car_reference_system.ravel(),  # NEW
                state_car_velocity_in_car_reference_system.ravel(),
                state_y_map_vector_in_car_reference_system.ravel(),
                state_zone_center_coordinates_in_car_reference_system.ravel(),
            )
        ).astype(np.float32)

        # Get action that was played
        action = rollout_results["actions"][i]

        if i + n_steps < n_frames - 1:
            # next_step is not the final frame of the rollout, we do not need to force the transition to be final in the minirace
            minirace_min_time_actions = 0

            # Construct next_state description
            next_current_zone_idx = rollout_results["current_zone_idx"][i + n_steps]
            next_car_position = rollout_results["car_position"][i + n_steps]
            next_car_orientation = rollout_results["car_orientation"][i + n_steps]
            next_car_velocity = rollout_results["car_velocity"][i + n_steps]
            next_state_zone_center_coordinates_in_car_reference_system = next_car_orientation.T.dot(
                (zone_centers[next_current_zone_idx : next_current_zone_idx + n_zone_centers_in_inputs, :] - next_car_position).T
            ).T  # (n_zone_centers_in_inputs, 3)
            next_state_y_map_vector_in_car_reference_system = next_car_orientation.T.dot(np.array([0, 1, 0]))
            next_state_car_velocity_in_car_reference_system = next_car_orientation.T.dot(next_car_velocity)
            # FIXME RUN13
            next_car_angular_speed = rollout_results["car_angular_speed"][i + n_steps]
            next_previous_action = misc.inputs[rollout_results["actions"][i + n_steps - 1]]
            next_state_car_angular_velocity_in_car_reference_system = next_car_orientation.T.dot(next_car_angular_speed)
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = np.hstack(
                (
                    0,  # Placeholder for mini_race_time_actions
                    np.array(
                        [
                            next_previous_action["accelerate"],
                            next_previous_action["brake"],
                            next_previous_action["left"],
                            next_previous_action["right"],
                        ]
                    ),  # NEW
                    rollout_results["car_gear_and_wheels"][i + n_steps].ravel(),  # NEW
                    next_state_car_angular_velocity_in_car_reference_system.ravel(),  # NEW
                    next_state_car_velocity_in_car_reference_system.ravel(),
                    next_state_y_map_vector_in_car_reference_system.ravel(),
                    next_state_zone_center_coordinates_in_car_reference_system.ravel(),
                )
            ).astype(np.float32)
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
