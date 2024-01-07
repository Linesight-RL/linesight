import numpy.typing as npt


class Experience:
    __slots__ = (
        "state_img",
        "state_float",
        "state_potential",
        "action",
        "n_steps",
        "rewards",
        "next_state_img",
        "next_state_float",
        "next_state_potential",
        "gammas",
        "terminal_actions",
    )

    def __init__(
        self,
        state_img: npt.NDArray,
        state_float: npt.NDArray,
        state_potential: float,
        action: int,
        n_steps: int,
        rewards: npt.NDArray,
        next_state_img: npt.NDArray,
        next_state_float: npt.NDArray,
        next_state_potential: float,
        gammas: npt.NDArray,
        terminal_actions: int,
    ):
        self.state_img = state_img
        self.state_float = state_float
        self.state_potential = state_potential
        self.action = action
        self.n_steps = n_steps
        self.rewards = rewards
        self.next_state_img = next_state_img
        self.next_state_float = next_state_float
        self.next_state_potential = next_state_potential
        self.gammas = gammas
        self.terminal_actions = terminal_actions
