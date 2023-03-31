from .experience_replay_interface import ExperienceReplayInterface, Experience
from typing import List, Tuple
from collections import deque
import random

class BasicExperienceReplay(ExperienceReplayInterface):
    def __init__(self, buffer_max_size):
        self.buffer = deque(maxlen=buffer_max_size)

    def add(self, experience:Experience)->None:
        self.buffer.append(experience)

    def sample(self, n:int)->Tuple[List[Experience], None, None]: 
        # Simple sample with replacement, this way we don't have to worry about the case where n > len(buffer)
        return random.choices(population=self.buffer, k=n)

    def update(self, idxs:List[int], errors)->None:
        pass

    def __len__(self)->int:
        return len(self.buffer)

    def max_len(self)->int:
        return self.buffer.maxlen