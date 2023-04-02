import random
from collections import deque
from typing import List, Tuple

import numpy as np

from .experience_replay_interface import Experience, ExperienceReplayInterface


class BasicExperienceReplay(ExperienceReplayInterface):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, n: int) -> Tuple[List[Experience], None, None]:
        # Simple sample with replacement, this way we don't have to worry about the case where n > len(buffer)
        return random.choices(population=self.buffer, k=n), None, np.ones(n)

    def update(self, idxs: List[int], errors) -> None:
        pass

    def __len__(self) -> int:
        return len(self.buffer)

    def max_len(self) -> int:
        return self.buffer.maxlen
