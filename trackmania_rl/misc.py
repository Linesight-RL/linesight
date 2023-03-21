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

action_forward_idx = 7  # Accelerate forward, don't turn
action_backward_idx = 8  # Go backward, don't turn

W = 640
H = 480

wind32gui_margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

gamma = 0.95
reward_scaling = 0 * 1e-3
reward_per_tm_engine_step = -0.0001 * reward_scaling
reward_failed_to_finish = reward_per_tm_engine_step * 100 * 2 * reward_scaling  # As if we finished 2 seconds later
reward_per_cp_passed = 0.1 * reward_scaling
reward_per_velocity = 0.1 * reward_scaling
reward_per_input_gas = 0.2 * reward_scaling
bogus_reward_per_speed = 1e-4
bogus_reward_per_input_gas = 1e-2

running_speed = 2
run_steps_per_action = 10
max_rollout_time_ms = 5_000
n_steps = 1


float_input_dim = 14
float_hidden_dim = 64
conv_head_output_dim = 1152
dense_hidden_dimension = 1024
memory_size = 20_000
batch_size = 1024
learning_rate = 2e-4
number_times_single_memory_is_used_before_discard = 32
number_memories_trained_on_between_target_network_updates = 4096
soft_update_tau = 0.1
epsilon = 0.2


float_inputs_mean = np.array(
    [
        200,
        max_rollout_time_ms / 3,
        2,
        0.5,
        0.5,
        0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
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
