import collections
import random
from collections import deque
from typing import Tuple

import numpy as np

from . import misc


def scale_float_inputs(array):
    return (array - misc.float_inputs_mean) / misc.float_inputs_std


def get_buffer():
    return collections.deque(maxlen=misc.memory_size)


class Memory:
    __slots__ = (
        "state_img",
        "state_float",
        "action",
        "reward",
        "done",
        "next_state_img",
        "next_state_float",
    )

    def __init__(
        self,
        state_img,
        state_float,
        action,
        reward,
        done,
        next_state_img,
        next_state_float,
    ):
        self.state_img = state_img
        self.state_float = state_float
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state_img = next_state_img
        self.next_state_float = next_state_float


def fill_buffer_from_rollout_with_n_steps_rule(buffer: deque[Tuple], rollout_results: dict, n_steps: int):
    number_memories_added = 0
    for i in range(len(rollout_results["done"]) - n_steps):
        if not all(rollout_results["action_was_greedy"][i + 1 : i + n_steps]):
            # There was an exploration action during the n_steps, can't use this to learn
            print("not all")
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
            # Should be none, but we need to have a placeholder with correct data type and shape
            next_state_img = state_img
            next_state_float = state_float
        else:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = scale_float_inputs(rollout_results["floats"][i + n_steps])

        buffer.append(
            Memory(
                state_img,
                state_float,
                action,
                reward,
                done,
                next_state_img,
                next_state_float,
            )
        )

        number_memories_added += 1

    return buffer, number_memories_added


def sample(buffer, n):
    # Simple sample with replacement, this way we don't have to worry about the case where n > len(buffer)
    return random.choices(population=buffer, k=n)
