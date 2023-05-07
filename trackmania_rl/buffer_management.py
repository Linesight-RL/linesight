import numpy as np

from . import misc
from .experience_replay.experience_replay_interface import Experience, ExperienceReplayInterface
import random

def fill_buffer_from_rollout_with_n_steps_rule(
        buffer: ExperienceReplayInterface,
        rollout_results: dict,
        n_steps_max: int,
        gamma: float,
        discard_non_greedy_actions_in_nsteps: bool,
        n_checkpoints_in_inputs: int,
        zone_centers,
):
    assert n_checkpoints_in_inputs >= 2

    for k, v in rollout_results.items():
        print(k, len(v))

    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])

    n_frames = len(rollout_results["frames"])

    number_memories_added = 0
    for i in range(n_frames - 1):
        # Loop over all frames that were generated

        current_zone_idx = rollout_results["current_zone_idx"][i]
        assert current_zone_idx < len(zone_centers) - 1

        mini_race_range = range(
            max(0, current_zone_idx + 2 - n_checkpoints_in_inputs),
            1 + min(current_zone_idx, len(zone_centers) - n_checkpoints_in_inputs)
        )

        for first_zone_idx_in_input in random.sample(mini_race_range,
                                                     min(len(mini_race_range), misc.sample_n_mini_races)):
            # A mini-race is defined as a race between Zone and Zone + n_checkpoints_in_inputs - 1 (both included)
            # That mini-race terminates as soon as the car enters Zone + n_checkpoints_in_inputs - 1
            # Given a frame, we define multiple mini-races simultaneously: one where the car has just started, one where the car is close to finish, etc...

            last_zone_idx_in_input = first_zone_idx_in_input + n_checkpoints_in_inputs - 1  # included
            assert last_zone_idx_in_input < len(zone_centers)

            time_mini_race_start_ms = rollout_results["zone_entrance_time_ms"][first_zone_idx_in_input]
            current_overall_time_ms = i * misc.ms_per_tm_engine_step * misc.tm_engine_step_per_action
            mini_race_duration_ms = current_overall_time_ms - time_mini_race_start_ms

            # Find the longest n-step transition we can use.
            for n_steps in range(n_steps_max, 0, -1):
                # A n-step transition might be unusable if the mini-race was successfully finished before frame + n_steps
                # or if
                if (
                        i + n_steps < n_frames  # We have enough frames
                        and rollout_results["current_zone_idx"][i + n_steps - 1]
                        < last_zone_idx_in_input  # Previous frame was not "race finished"
                        and mini_race_duration_ms + (
                        n_steps - 1) * misc.ms_per_tm_engine_step * misc.tm_engine_step_per_action
                        < misc.max_minirace_duration_ms  # Previous frame was not "race timeout"
                ):
                    mini_race_finished = rollout_results["current_zone_idx"][i + n_steps] == last_zone_idx_in_input
                    mini_race_timeout = (not mini_race_finished) and (
                        (
                                mini_race_duration_ms + (
                                n_steps * misc.ms_per_tm_engine_step * misc.tm_engine_step_per_action)
                                >= misc.max_minirace_duration_ms
                        )
                    )
                    break
            else:
                print("We were not able to find a suitable n-steps value")
                assert False

            assert n_steps >= 1

            if discard_non_greedy_actions_in_nsteps and not all(
                    rollout_results["action_was_greedy"][i + 1: i + n_steps]):
                # There was an exploration action during the n_steps, can't use this to learn
                continue

            # At this point we know:
            #   - which frame we're looking at
            #   - what the next frame is
            #   - what checkpoints are included in the mini-race
            #   - whether the mini-race is finished, successfully or not
            # We still need to calculate the reward for this transition

            gammas = gamma ** np.linspace(0, n_steps - 1, n_steps)
            reward = 0
            if mini_race_finished:
                # If the race is finished, the last transition was spent partially in race, partially after the race ended.
                # We reduce the reward proportionnally such that the reward is only received during the time spent in race
                gammas[-1] *= rollout_results["fraction_time_in_previous_zone"][i + n_steps]
                reward += (gamma ** (n_steps - 1)) * misc.reward_on_finish
            elif mini_race_timeout:
                reward += (gamma ** (n_steps - 1)) * misc.reward_on_failed_to_finish
            # Reward due to time spent in race
            reward += np.sum(gammas) * misc.reward_per_ms_in_race * misc.ms_per_action
            # Reward due to speed
            reward += (
                    np.sum(gammas * np.array(rollout_results["display_speed"][i + 1: i + 1 + n_steps]))
                    * misc.reward_per_ms_velocity
                    * misc.ms_per_action
            )
            # Reward due to press forward
            reward += (
                    np.sum(gammas * np.array(rollout_results["input_w"][i: i + n_steps]))
                    * misc.reward_per_ms_press_forward
                    * misc.ms_per_action
            )

            car_position = rollout_results["car_position"][i]
            car_orientation = rollout_results["car_orientation"][i]
            car_velocity = rollout_results["car_velocity"][i]
            state_zone_center_coordinates_in_car_reference_system = car_orientation.T.dot(
                (zone_centers[first_zone_idx_in_input: last_zone_idx_in_input + 1, :] - car_position).T
            ).T  # (n_checkpoints_in_inputs, 3)
            state_y_map_vector_in_car_reference_system = car_orientation.T.dot(np.array([0, 1, 0]))
            state_car_velocity_in_car_reference_system = car_orientation.T.dot(car_velocity)
            car_angular_speed = rollout_results["car_angular_speed"][i]
            previous_action = misc.inputs[rollout_results["actions"][misc.action_forward_idx if i == 0 else i - 1]]
            state_car_angular_velocity_in_car_reference_system = car_orientation.T.dot(car_angular_speed)
            assert state_zone_center_coordinates_in_car_reference_system.shape == (n_checkpoints_in_inputs, 3)
            assert len(state_y_map_vector_in_car_reference_system) == 3
            assert len(state_car_velocity_in_car_reference_system) == 3

            state_img = rollout_results["frames"][i]
            state_float = np.hstack(
                (
                    mini_race_duration_ms,
                    np.array([previous_action['accelerate'], previous_action['brake'], previous_action['left'],
                              previous_action['right']]),  # NEW
                    rollout_results["car_gear_and_wheels"][i].ravel(),  # NEW
                    state_car_angular_velocity_in_car_reference_system.ravel(),  # NEW
                    state_car_velocity_in_car_reference_system.ravel(),
                    state_y_map_vector_in_car_reference_system.ravel(),
                    state_zone_center_coordinates_in_car_reference_system.ravel(),
                    current_zone_idx >= np.arange(first_zone_idx_in_input + 1, last_zone_idx_in_input),
                )
            ).astype(
                np.float32)  # TODO : cache this so that state_float and next_state_float can share memory for different experiences

            action = rollout_results["actions"][i]
            done = mini_race_finished or mini_race_timeout

            # if done:
            #     print(i, mini_race_duration_ms, n_steps, mini_race_finished, mini_race_timeout)

            if not done:
                next_car_position = rollout_results["car_position"][i + n_steps]
                next_car_orientation = rollout_results["car_orientation"][i + n_steps]
                next_car_velocity = rollout_results["car_velocity"][i + n_steps]
                next_state_zone_center_coordinates_in_car_reference_system = next_car_orientation.T.dot(
                    (zone_centers[first_zone_idx_in_input: last_zone_idx_in_input + 1, :] - next_car_position).T
                ).T  # (n_checkpoints_in_inputs, 3)
                next_state_y_map_vector_in_car_reference_system = next_car_orientation.T.dot(np.array([0, 1, 0]))
                next_state_car_velocity_in_car_reference_system = next_car_orientation.T.dot(next_car_velocity)
                next_car_angular_speed = rollout_results["car_angular_speed"][i + n_steps]
                next_previous_action = misc.inputs[rollout_results["actions"][i + n_steps - 1]]
                next_state_car_angular_velocity_in_car_reference_system = next_car_orientation.T.dot(
                    next_car_angular_speed)
                next_state_img = rollout_results["frames"][i + n_steps]
                next_state_float = np.hstack(
                    (
                        mini_race_duration_ms + n_steps * misc.ms_per_action,
                        np.array([next_previous_action['accelerate'], next_previous_action['brake'],
                                  next_previous_action['left'],
                                  next_previous_action['right']]),  # NEW
                        rollout_results["car_gear_and_wheels"][i + n_steps].ravel(),  # NEW
                        next_state_car_angular_velocity_in_car_reference_system.ravel(),  # NEW
                        next_state_car_velocity_in_car_reference_system.ravel(),
                        next_state_y_map_vector_in_car_reference_system.ravel(),
                        next_state_zone_center_coordinates_in_car_reference_system.ravel(),
                        rollout_results["current_zone_idx"][i + n_steps] >= np.arange(first_zone_idx_in_input + 1,
                                                                                      last_zone_idx_in_input),
                    )
                ).astype(np.float32)
            else:
                next_state_img = state_img
                next_state_float = state_float

            buffer.add(
                Experience(
                    state_img,
                    state_float,
                    action,
                    reward,
                    done,
                    next_state_img,
                    next_state_float,
                    gamma ** n_steps,
                ),
            )

            number_memories_added += 1

    return buffer, number_memories_added
