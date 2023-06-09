import numpy.typing as npt


class Experience:
    __slots__ = (
        "state_img",
        "state_float",
        "action",
        "n_steps",
        "rewards",
        "next_state_img",
        "next_state_float",
        "gammas",
        "minirace_min_time_actions",
    )

    def __init__(
        self,
        state_img: npt.NDArray,
        state_float: npt.NDArray,
        action: int,
        n_steps: int,
        rewards: npt.NDArray,
        next_state_img: npt.NDArray,
        next_state_float: npt.NDArray,
        gammas: npt.NDArray,
        minirace_min_time_actions: int,
    ):
        self.state_img = state_img
        self.state_float = state_float
        self.action = action
        self.n_steps = n_steps
        self.rewards = rewards
        self.next_state_img = next_state_img
        self.next_state_float = next_state_float
        self.gammas = gammas
        self.minirace_min_time_actions = minirace_min_time_actions

    def __repr__(self):
        return f"{self.state_img=}\n{self.state_float=}\n{self.action=}\n{self.n_steps=}\n{self.rewards=}\n{self.next_state_img=}\n{self.next_state_float=}\n{self.gammas=}\n{self.minirace_min_time_actions=}\n"
