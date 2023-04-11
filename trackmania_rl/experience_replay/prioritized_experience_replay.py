import random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from .experience_replay_interface import Experience, ExperienceReplayInterface

"""
Prioritized Experience Replay behavior:

- alpha = 0 : uniform sampling
- alpha > 0 : gradually more non uniform

- beta = 0 : absolutely no correction
- beta = 1 : full IS compensation of weight

- alpha = 0, beta = 1 to have a "normal" experience
- alpha = 0.6-0.7, beta = 0.4-0.5 recommended by original paper
- alpha = 0.2, beta = 1 recommended in IQN paper
"""


class PrioritizedExperienceReplay(ExperienceReplayInterface):
    def __init__(self, capacity: int, sample_with_segments: bool, prio_alpha: float, prio_beta: float, prio_epsilon: float):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.sample_with_segments = sample_with_segments  # Currently unused
        self.prio_alpha = prio_alpha
        self.prio_beta = prio_beta
        self.prio_epsilon = prio_epsilon

    def add(self, experience: Experience) -> None:
        default_prio_float = 1.0  # Only used when adding the very first element
        if self.tree.n_entries != 0:
            default_prio_float = float(self.tree.total() / (1e9 * self.tree.n_entries))
        self.tree.add(default_prio_float, experience)

    def sample(self, n: int) -> Tuple[List[Experience], List[int], npt.NDArray[np.float32]]:
        batch = []
        idxs = []
        prios_int64 = []
        total_int64 = self.tree.total()
        for _ in range(n):
            s_int64 = np.random.randint(0, total_int64, dtype=np.int64)
            (idx, p_int64, experience) = self.tree.get(s_int64)  # type: ignore
            prios_int64.append(p_int64)
            batch.append(experience)
            idxs.append(idx)
            # self.tree.update(idx, p_int64 / 2e9)  # Modified vs Agade's code
        is_weight = np.power(self.tree.n_entries * np.array(prios_int64) / total_int64, -self.prio_beta).astype(np.float32)
        return batch, idxs, is_weight

    def update(self, idxs: List[int], errors) -> None:
        prios_float = self._calculate_priority(errors)
        for idx, prio_float in zip(idxs, prios_float):
            self.tree.update(idx, prio_float)

    def __len__(self) -> int:
        return self.tree.n_entries

    def max_len(self) -> int:
        return self.capacity

    def _calculate_priority(self, error):
        return (np.absolute(error) + self.prio_epsilon) ** self.prio_alpha


# Originally from https://github.com/rlcode/per, somewhat modified since then
class SumTree:
    __slots__ = (
        "capacity",
        "tree_int64",
        "data",
        "n_entries",
        "write",
    )

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_int64 = np.zeros(2 * capacity - 1, dtype=np.int64)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    # update to the root node
    def _propagate(self, idx: int, change_int64: np.int64) -> None:
        parent = (idx - 1) // 2
        self.tree_int64[parent] += change_int64
        if parent != 0:
            self._propagate(parent, change_int64)

    # find sample on leaf node
    def _retrieve(self, idx: int, s_int64: np.int64) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree_int64):
            return idx

        if s_int64 <= self.tree_int64[left]:
            return self._retrieve(left, s_int64)
        else:
            return self._retrieve(right, s_int64 - self.tree_int64[left])

    def total(self) -> np.int64:
        return self.tree_int64[0]

    # store priority and sample
    def add(self, prio_float: float, data: Experience) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, prio_float)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, prio_float) -> None:
        prio_int64 = np.int64(prio_float * 1e9)
        change_int64 = prio_int64 - self.tree_int64[idx]
        self.tree_int64[idx] = prio_int64
        self._propagate(idx, change_int64)

    # get sample based on priority
    def get(self, s_int64: np.int64) -> Tuple[int, np.int64, Experience]:
        idx = self._retrieve(0, s_int64)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree_int64[idx], self.data[dataIdx]
