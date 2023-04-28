import numpy as np

W = 640
H = 480

wind32gui_margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

running_speed = 20
tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action
n_checkpoints_in_inputs = 10
max_overall_duration_ms = 50_000
max_minirace_duration_ms = 24_000
n_steps = 3

epsilon = 0.01
discard_non_greedy_actions_in_nsteps = True
anneal_step = 1
gamma = [
    0.95,
    0.99,
    0.99,
    1,
    1,
    1,
][anneal_step]
reward_per_ms_in_race = [
    -0.15 / ms_per_action,
    -0.03 / ms_per_action,
    -0.03 / ms_per_action,
][anneal_step]

reward_on_finish = [
    1,
    1,
    1,
][anneal_step]
reward_on_failed_to_finish = 0
reward_per_ms_velocity = [
    0.15 / ms_per_action / 800,
    0.03 / ms_per_action / 800,
    0.03 / ms_per_action / 4000,
][
    anneal_step
]  # If we manage to have 400 speed, the agent will want to run forever
reward_per_ms_press_forward = [
    0.15 / ms_per_action / 4,
    0.03 / ms_per_action / 8,
    0,
][anneal_step]

statistics_save_period_seconds = 60 * 10


float_input_dim = 5 + 4 * n_checkpoints_in_inputs
float_hidden_dim = 64
conv_head_output_dim = 1152
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8
iqn_k = 32
iqn_kappa = 1
AL_alpha = [0, 0, 0, 0, 0, 0, 0.8][anneal_step]

memory_size = 30_000 * (n_checkpoints_in_inputs - 1)
memory_size_start_learn = 10_000
virtual_memory_size_start_learn = 10_000
number_memories_generated_high_exploration = 500_000
high_exploration_ratio = 10
batch_size = 1024
learning_rate = 5e-5

number_times_single_memory_is_used_before_discard = 8
number_memories_trained_on_between_target_network_updates = 10000

soft_update_tau = 0.2

float_inputs_mean = np.array(
    [
        max_minirace_duration_ms / 4,
        0,
        0,
        150,
        0,
        1,
        0,
        0,
        0,
        -45,  #
        0,
        0,
        -35,  #
        0,
        0,
        -25,  #
        0,
        0,
        -15,  #
        0,
        0,
        -5,  #
        0,
        0,
        5,  #
        0,
        0,
        15,  #
        0,
        0,
        25,  #
        0,
        0,
        35,  #
        0,
        0,
        45,  #
        0.5,
        0.5,
        0.5,
        0.5,  #
    ]
)

float_inputs_std = np.array(
    [
        max_minirace_duration_ms / 2,
        10,
        10,
        150,
        1,
        1,
        1,
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        25,
        25,
        25,  #
        1,
        1,
        1,
        1,  #
    ]
)

# float_inputs_mean = np.zeros(float_input_dim)
# float_inputs_std = np.ones(float_input_dim)

# 1, 4, 7 : accelerate
# 0, 3, 6 :
# 2, 5, 8 : backwards

inputs = [
    {  # 0 : Left, nothing else
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 1
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 2
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 3 : Right, nothing else
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": False,
    },
    {  # 4
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": False,
    },
    {  # 5
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": True,
    },
    {  # 6 : Do nothing
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 7 : Accelerate forward, don't turn
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 8 : Go backward, don't turn
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
]

action_forward_idx = 7  # Accelerate forward, don't turn
action_backward_idx = 8  # Go backward, don't turn
