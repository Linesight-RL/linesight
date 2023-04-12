from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class Experience:
    __slots__ = (
        "state_img",
        "state_float",
        "action",
        "reward",
        "done",
        "next_state_img",
        "next_state_float",
        "gamma_pow_nsteps",
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
        gamma_pow_nsteps,
    ):
        self.state_img = state_img
        self.state_float = state_float
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state_img = next_state_img
        self.next_state_float = next_state_float
        self.gamma_pow_nsteps = gamma_pow_nsteps

    def __repr__(self):
        return f"{self.state_img=}\n{self.state_float=}\n{self.action=}\n{self.reward=}\n{self.done=}\n{self.next_state_img=}\n{self.next_state_float=}\n{self.gamma_pow_nsteps=}\n"


class ExperienceReplayInterface(ABC):
    @abstractmethod
    def add(self, experience: Experience) -> None:
        pass

    @abstractmethod
    def sample(self, n: int) -> Tuple[List[Experience], Any, Any]:
        pass

    @abstractmethod
    def update(self, idxs: List[int], errors) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def max_len(self) -> int:
        pass
