import numpy as np

inputs = [
    {
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": False,
    },
    {
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": False,
    },
    {
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": True,
    },
    {
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
]

default_action_idx = 7  # Accelerate forward, don't turn

W = 640
H = 480

wind32gui_margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

gamma = 0.999
reward_per_tm_engine_step = -1
reward_failed_to_finish = reward_per_tm_engine_step * 100 * 10  # As if we finished 10 seconds later
reward_per_cp_passed = 0.1
reward_per_velocity = 0.1

max_rollout_time_ms = 10_000
n_steps = 3

memory_size = 20_000

float_input_dim = 14
# float_hidden_dim =

float_inputs_mean = np.array(
    [
        200,
        max_rollout_time_ms / 3,
        2,
        -0.5,
        -0.5,
        0,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
    ]
)

float_inputs_std = np.array(
    [
        200,
        max_rollout_time_ms / 2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
)
