from .experience_replay_interface import ExperienceReplayInterface, Experience
from typing import List, Tuple, Any
import random
import numpy as np


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
    def __init__(self, capacity, sample_with_segments: bool, prio_alpha: float, prio_beta: float, prio_epsilon: float):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.sample_with_segments = sample_with_segments
        self.prio_alpha = prio_alpha
        self.prio_beta = prio_beta
        self.prio_epsilon = prio_epsilon

    def add(self, experience:Experience)->None: 
        default_prio = (
            self.tree.total() / self.tree.n_entries if self.tree.n_entries != 0 else 1
        )  # Modified vs Agade's code
        self.tree.add(default_prio, experience)

    def sample(self, n:int)->Tuple[List[Experience], List[int], np.array]: 
        initial_total = self.tree.total()
        batch = []
        idxs = []
        priorities = []
        for i in range(n):
            if self.sample_with_segments:
                segment = self.tree.total() / n
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
            else:
                s = random.uniform(0, self.tree.total())
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            self.tree.update(idx, self._calculate_priority(0))  # Modified vs Agade's code

        sampling_probabilities = np.array(priorities) / initial_total
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.prio_beta)
        return batch, idxs, is_weight.astype(np.float32)

    def update(self, idxs:List[int], errors)->None:
        prios = self._calculate_priority(errors)
        for idx, prio in zip(idxs, prios):
            self.tree.update(idx, prio)

    def __len__(self)->int:
        return self.sum_tree.n_entries

    def max_len(self)->int:
        return self.capacity

    def _calculate_priority(self, error):
        return (np.absolute(error) + self.prio_epsilon) ** self.prio_alpha



# Originally from https://github.com/rlcode/per, somewhat modified since then
class SumTree:
    __slots__ = (
        "capacity",
        "tree",
        "data",
        "n_entries",
        "write",
    )

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, prio, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, prio)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, prio):
        change = prio - self.tree[idx]

        self.tree[idx] = prio
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])