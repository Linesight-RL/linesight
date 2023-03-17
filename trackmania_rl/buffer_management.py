from collections import deque
from typing import Tuple
from . import misc
import numpy as np
import collections


def scale_float_inputs(array):
    return (array - misc.float_inputs_meanm) / misc.float_inputs_std


def get_buffer():
    return collections.deque(maxlen=misc.memory_size)


def fill_buffer_from_rollout_with_n_steps_rule(buffer: deque[Tuple], rollout_results: dict, n_steps):
    for i in range(len(rollout_results["done"]) - n_steps):
        if not all([rollout_results["action_was_greedy"][i + 1 : i + n_steps]]):
            # There was an exploration action during the n_steps, can't use this to learn
            continue

        state_img = rollout_results["frames"][i]
        state_float = scale_float_inputs(rollout_results["floats"][i])
        action = rollout_results["actions"][i]
        reward = np.sum(
            np.array(rollout_results["rewards"][i + 1 : i + 1 + n_steps])
            * (misc.gamma ** np.linspace(0, n_steps - 1, n_steps))
        )
        done = rollout_results["done"][i + n_steps]
        if done:
            next_state_img = None
            next_state_float = None
        else:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = scale_float_inputs(rollout_results["floats"][i + n_steps])

        buffer.append(
            (
                state_img,
                state_float,
                action,
                reward,
                done,
                next_state_img,
                next_state_float,
            )
        )

    return buffer
