import numpy as np

from . import misc
from .experience_replay.experience_replay_interface import Experience, ExperienceReplayInterface

# def scale_float_inputs(array):
#     return (array - misc.float_inputs_mean) / misc.float_inputs_std


def fill_buffer_from_rollout_with_n_steps_rule(buffer: ExperienceReplayInterface, rollout_results: dict, n_steps: int):
    number_memories_added = 0
    for i in range(len(rollout_results["done"]) - n_steps):
        if not all(rollout_results["action_was_greedy"][i + 1 : i + n_steps]):
            # There was an exploration action during the n_steps, can't use this to learn
            continue

        state_img = rollout_results["frames"][i]
        state_float = rollout_results["floats"][i]
        action = rollout_results["actions"][i]
        reward = np.sum(
            np.array(rollout_results["rewards"][i + 1 : i + 1 + n_steps]) * (misc.gamma ** np.linspace(0, n_steps - 1, n_steps))
        )
        done = rollout_results["done"][i + n_steps]
        if (not done) and (i == len(rollout_results["done"]) - n_steps - 1):
            break
        elif done:
            # continue  # Temporary way to **not** add any final transitions in the memory #TODO #FIXME
            # Should be none, but we need to have a placeholder with correct data type and shape
            next_state_img = state_img
            next_state_float = state_float
        else:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = rollout_results["floats"][i + n_steps]

        buffer.add(
            Experience(
                state_img,
                state_float,
                action,
                reward,
                done,
                next_state_img,
                next_state_float,
            ),
        )

        number_memories_added += 1

    return buffer, number_memories_added
