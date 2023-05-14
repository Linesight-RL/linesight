import numpy as np

W = 640
H = 480

wind32gui_margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

running_speed = 20
tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action
n_zone_centers_in_inputs = 16
max_overall_duration_ms = 300_000
max_minirace_duration_ms = 25_000

epsilon = 0.02
epsilon_boltzmann = 0.02
tau_epsilon_boltzmann = 0.01
tau_greedy_boltzmann = 0.0005
discard_non_greedy_actions_in_nsteps = True
buffer_test_ratio = 0.05

anneal_step = 1
n_steps = [
    1,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
][anneal_step]

gamma = [
    0.95,
    0.99,
    0.99,
    1,
    1,
    1,
    1,
][anneal_step]
reward_per_ms_in_race = [
    -0.15 / ms_per_action,
    -0.03 / ms_per_action,
    -0.03 / ms_per_action,
    -0.0003,
    # With previous step, V(first_frame = -1.8). We want to maintain that, knowing that it takes 9 seconds to do the first 400 meters in race, that's 2.8/9000
    -0.0003,
][anneal_step]

reward_on_finish = [
    1,
    1,
    1,
    1,
    1,
][anneal_step]
reward_on_failed_to_finish = [0, -1, -1, -1, -1][anneal_step]
reward_per_ms_velocity = [
    0.15 / ms_per_action / 800,
    0.03 / ms_per_action / 800,
    0.03 / ms_per_action / 800,
    0,
    0,
][anneal_step]
reward_per_ms_press_forward = [
    0.15 / ms_per_action / 4,
    0.03 / ms_per_action / 8,
    0,
    0,
    0,
][anneal_step]

statistics_save_period_seconds = 60 * 10

float_input_dim = 22 + 3 * n_zone_centers_in_inputs
float_hidden_dim = 256
conv_head_output_dim = 1152
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8
iqn_k = 32
iqn_kappa = 1
AL_alpha = [0, 0, 0, 0, 0.8][anneal_step]

memory_size = 300_000
memory_size_start_learn = 10000
virtual_memory_size_start_learn = 1000
number_memories_generated_high_exploration = 100000
high_exploration_ratio = 10
batch_size = 1024
learning_rate = 5e-5

number_times_single_memory_is_used_before_discard = 32
number_memories_trained_on_between_target_network_updates = 10000
subsample_n_mini_races = 4000000000

soft_update_tau = 0.2  # [1.0, 0.5, 0.2, 0.1][anneal_step]

float_inputs_mean = np.array(
    [
        2400,
        #######
        0.5,
        0.5,
        0.5,
        0.5,  # Previous action
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        2.5,
        7000,  # Car gear and wheels
        0,
        0,
        0,  # Angular velocity
        #######
        0,
        0,
        55,
        0,
        1,
        0,
        # ==================== BEGIN 40 CP =====================
        # ==================== END   40 CP =====================
        # ==================== BEGIN 16 CP =====================
        0,
        0,
        -8,
        0,
        0,
        7,
        0,
        0,
        20,
        0,
        0,
        32,
        0,
        0,
        44,
        0,
        0,
        54,
        0,
        0,
        64,
        0,
        0,
        74,
        0,
        0,
        84,
        0,
        0,
        94,
        0,
        0,
        104,
        0,
        0,
        114,
        0,
        0,
        124,
        0,
        0,
        134,
        0,
        0,
        144,
        0,
        0,
        154,
        # ==================== END 16 CP =====================
        n_zone_centers_in_inputs / 2,
    ]
)

float_inputs_std = np.array(
    [
        1500,
        #######
        1,
        1,
        1,
        1,  # Previous action
        1,
        1,
        1,
        1,
        1,
        3,
        10000,  # Car gear and wheels
        2,
        2,
        2,  # Angular velocity
        #######
        10,
        10,
        15,
        1,
        1,
        1,
        # ==================== BEGIN 40 CP =====================
        # ==================== END   40 CP =====================
        # ==================== BEGIN 16 CP =====================
        25,
        25,
        25,
        25,
        25,
        25,
        35,
        35,
        35,
        45,
        45,
        45,
        55,
        55,
        55,
        65,
        65,
        65,
        75,
        75,
        75,
        85,
        85,
        85,
        95,
        95,
        95,
        105,
        105,
        105,
        115,
        115,
        115,
        125,
        125,
        125,
        135,
        135,
        135,
        135,
        135,
        135,
        145,
        145,
        145,
        150,
        150,
        150,
        n_zone_centers_in_inputs / 2,
        # ==================== END   16 CP =====================
    ]
)


inputs = [
    {  # 0 Forward
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 1 Forward left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 2 Forward right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": False,
    },
    {  # 3 Nothing
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 4 Nothing left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 5 Nothing right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": False,
    },
    {  # 6 Brake
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 7 Brake left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 8 Brake right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": True,
    },
    {  # 9 Brake and accelerate
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 10 Brake and accelerate left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 11 Brake and accelerate right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": True,
    },
]

action_forward_idx = 0  # Accelerate forward, don't turn
action_backward_idx = 6  # Go backward, don't turn

distance_between_checkpoints = 25
road_width = 40  ## a little bit of margin, could be closer to 24 probably ? Don't take risk there are curvy roads
max_allowable_distance_to_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

zone_centers_jitter = 0.0  # TODO : eval with zero jitter on zone centers !!
