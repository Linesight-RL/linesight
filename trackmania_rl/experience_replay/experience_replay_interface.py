"""
In this file, we define the Experience type.
This is used to represent a transition sampled from a ReplayBuffer.
"""

import numpy.typing as npt


class Experience:
    """
    (state_img, state_float):                   represent "state", ubiquitous in reinforcement learning
                                                state_img is a np.array of shape (1, H, W) and dtype np.uint8
                                                state_float is a np.array of shape (config.float_input_dim, ) and dtype np.float32
    (next_state_img, next_state_float):         represent "next_state"
                                                next_state_img is a np.array of shape (1, H, W) and dtype np.uint8
                                                next_state_float is a np.array of shape (config.float_input_dim, ) and dtype np.float32
    (state_potential and next_state_potential)  are floats, used for reward shaping as per Andrew Ng's paper: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
    action                                      is an integer representing the action taken for this transition, mapped to config_files/inputs_list.py
    terminal_actions                            is an integer representing the number of steps between "state" and race finish in the rollout from which this transition was extracted. If the rollout did not finish (ie: early cutoff), then contains math.inf
    n_steps                                     How many steps were taken between "state" and "next state". Not all transitions contain the same value, as this may depend on exploration policy. Note that in buffer_collate_function, a transition may be reinterpreted as terminal with a lower n_steps, depending on the random horizon that was sampled.
    gammas                                      a numpy array of shape (config.n_steps, ) containing the gamma value if steps = 0, 1, 2, etc...
    rewards                                     a numpy array of shape (config.n_steps, ) containing the reward value if steps = 0, 1, 2, etc...

    The structure of these transitions is unusual. It comes from our "mini-race" logic which will be explained somewhere else. I don't know where yet.
    This is how we are able to define Q-values as "the sum of expected rewards obtained during the next 7 seconds", and how we can optimise with gamma = 1.
    """

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
