import numpy as np

W = 640
H = 480

wind32gui_margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

gamma = 1
reward_per_tm_engine_step = -0.0025
reward_on_finish = 2
reward_on_failed_to_finish = 0

running_speed = 2
run_steps_per_action = 5
ms_per_run_step = 10
max_rollout_time_ms = 45_000
n_steps = 3

float_input_dim = 2
float_hidden_dim = 64
conv_head_output_dim = 1152
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8
iqn_k = 32
iqn_kappa = 1
AL_alpha = 0.8

memory_size = 30_000
memory_size_start_learn = 29_000
virtual_memory_size_start_learn = -10_000
batch_size = 1024
learning_rate = 1e-4
# clip_grad_value = 100

number_times_single_memory_is_used_before_discard = 32
number_memories_trained_on_between_target_network_updates = 10000

soft_update_tau = 0.1
epsilon = 0.01

prio_sample_with_segments = False
prio_alpha = 0.2
prio_beta = 0.8
prio_epsilon = 1e-6

float_inputs_mean = np.array(
    [
        200,
        max_rollout_time_ms / 3,
        # 2,
        # 0.5,
        # 0.5,
        # 0,
        # 0.5,
        # 0.5,
        # 0.5,
        # 0.5,
        # 0.5,
        # 0.5,
        # 0.5,
        # 0.5,
        # 0.5,
    ]
)

float_inputs_std = np.array(
    [
        200,
        max_rollout_time_ms / 2,
        # 2,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
        # 1,
    ]
)

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
