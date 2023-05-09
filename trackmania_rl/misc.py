import numpy as np

W = 640
H = 480

wind32gui_margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

running_speed = 20
tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action
n_zone_centers_in_inputs = 16
max_overall_duration_ms = 180_000
max_minirace_duration_ms = 25_000

epsilon = 0.05
discard_non_greedy_actions_in_nsteps = True

anneal_step = 1
n_steps = [
    1,
    1,
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

float_input_dim = [5, 19][0] + 4 * n_zone_centers_in_inputs
float_hidden_dim = 256
conv_head_output_dim = 1152
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8
iqn_k = 32
iqn_kappa = 1
AL_alpha = [0, 0, 0, 0, 0.8][anneal_step]

memory_size = 500_000
memory_size_start_learn = 1000
virtual_memory_size_start_learn = 1000
number_memories_generated_high_exploration = 400_000
high_exploration_ratio = 3
batch_size = 1024
learning_rate = 5e-5

number_times_single_memory_is_used_before_discard = 32
number_memories_trained_on_between_target_network_updates = 10000
sample_n_mini_races = 8

soft_update_tau = [1.0, 0.5, 0.2, 0.1][anneal_step]

float_inputs_mean = np.array(
    [
        2400,
        # 0.5, 0.5, 0.5, 0.5,  # Previous action
        # 0.5, 0.5, 0.5, 0.5, 0.5, 2.5, 7000,  # Car gear and wheels
        # 0, 0, 0,  # Angular velocity
        0,
        0,
        55,
        0,
        1,
        0,
        # 0, 0, -88,
        # 0, 0, -83,
        # 0, 0, -78,
        # 0, 0, -73,
        # 0, 0, -68,
        # 0, 0, -63,
        # 0, 0, -58,
        # 0, 0, -53,
        # 0, 0, -48,
        # 0, 0, -43,
        # 0, 0, -38,
        # 0, 0, -33,
        # 0, 0, -28,
        # 0, 0, -24,
        # 0, 0, -19,
        # 0, 0, -14,
        # 0, 0, -9,
        # 0, 0, -4,
        # 0, 0, 0,
        # 0, 0, 5,
        # 0, 0, 10,
        # 0, 0, 15,
        # 0, 0, 20,
        # 0, 0, 25,
        # 0, 0, 30,
        # 0, 0, 35,
        # 0, 0, 40,
        # 0, 0, 44,
        # 0, 0, 49,
        # 0, 0, 54,
        # 0, 0, 59,
        # 0, 0, 64,
        # 0, 0, 69,
        # 0, 0, 74,
        # 0, 0, 79,
        # 0, 0, 84,
        # 0, 0, 89,
        # 0, 0, 94,
        # 0, 0, 99,
        # 0, 0, 104,
        #
        # 0.97, 0.95, 0.92, 0.9, 0.87, 0.85, 0.82, 0.79, 0.77, 0.74, 0.72,
        # 0.69, 0.67, 0.64, 0.62, 0.59, 0.56, 0.54, 0.51, 0.49, 0.46, 0.44,
        # 0.41, 0.38, 0.36, 0.33, 0.31, 0.28, 0.26, 0.23, 0.21, 0.18, 0.15,
        # 0.13, 0.1, 0.08, 0.05, 0.03
        0,
        0,
        -88,
        0,
        0,
        -78,
        0,
        0,
        -68,
        0,
        0,
        -58,
        0,
        0,
        -46,
        0,
        0,
        -31,
        0,
        0,
        -14,
        0,
        0,
        3.5,
        0,
        0,
        20,
        0,
        0,
        37,
        0,
        0,
        51,
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
        0.93333333,
        0.86666667,
        0.8,
        0.73333333,
        0.66666667,
        0.6,
        0.53333333,
        0.46666667,
        0.4,
        0.33333333,
        0.26666667,
        0.2,
        0.13333333,
        0.06666667,
    ]
)

float_inputs_std = np.array(
    [
        1500,
        # 1, 1, 1, 1,  # Previous action
        # 1, 1, 1, 1, 1, 3, 10000,  # Car gear and wheels
        # 2, 2, 2,  # Angular velocity
        10,
        10,
        15,
        1,
        1,
        1,
        # ==================== 40 CP =====================
        # 88, 88, 44,
        # 85, 85, 44,
        # 81, 81, 44,
        # 78, 78, 44,
        # 75, 75, 44,
        # 72, 72, 44,
        # 68, 68, 44,
        # 65, 65, 44,
        # 62, 62, 44,
        # 59, 59, 44,
        # 55, 55, 44,
        # 52, 52, 44,
        # 49, 49, 44,
        # 46, 46, 44,
        # 42, 42, 44,
        # 39, 39, 44,
        # 36, 36, 44,
        # 33, 33, 44,
        # 29, 29, 44,
        # 26, 26, 44,
        # 26, 26, 44,
        # 30, 30, 44,
        # 34, 34, 44,
        # 38, 38, 44,
        # 42, 42, 44,
        # 47, 47, 44,
        # 51, 51, 44,
        # 55, 55, 44,
        # 59, 59, 44,
        # 63, 63, 44,
        # 67, 67, 44,
        # 71, 71, 44,
        # 75, 75, 44,
        # 79, 79, 44,
        # 83, 83, 44,
        # 88, 88, 44,
        # 92, 92, 44,
        # 96, 96, 44,
        # 100, 100, 44,
        # 104, 104, 44,
        #
        # 3.1,
        # 3.2,
        # 3.3,
        # 3.4,
        # 3.5,
        # 3.6,
        # 3.7,
        # 3.8,
        # 3.9,
        # 4.1,
        # 4.2,
        # 4.3,
        # 4.4,
        # 4.5,
        # 4.6,
        # 4.7,
        # 4.8,
        # 4.9,
        # 5.0,
        # 5.0,
        # 4.9,
        # 4.8,
        # 4.7,
        # 4.6,
        # 4.5,
        # 4.4,
        # 4.3,
        # 4.2,
        # 4.1,
        # 3.9,
        # 3.8,
        # 3.7,
        # 3.6,
        # 3.5,
        # 3.4,
        # 3.3,
        # 3.2,
        # 3.1
        # ==================== 40 CP =====================
        # ==================== 16 CP =====================
        88,
        88,
        44,
        77,
        77,
        44,
        66,
        66,
        44,
        55,
        55,
        44,
        44,
        44,
        48,
        35,
        35,
        50,
        28,
        28,
        54,
        26,
        26,
        54,
        31,
        31,
        53,
        41,
        41,
        50,
        53,
        53,
        46,
        64,
        64,
        44,
        74,
        74,
        44,
        84,
        84,
        44,
        94,
        94,
        44,
        104,
        104,
        44,
        3,
        3,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        4,
        4,
        4,
        3,
        3
        # ==================== 16 CP =====================
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
    # {  # 9 Brake and accelerate
    #     "left": False,
    #     "right": False,
    #     "accelerate": True,
    #     "brake": True,
    # },
    # {  # 10 Brake and accelerate left
    #     "left": True,
    #     "right": False,
    #     "accelerate": True,
    #     "brake": True,
    # },
    # {  # 11 Brake and accelerate right
    #     "left": False,
    #     "right": True,
    #     "accelerate": True,
    #     "brake": True,
    # },
]

action_forward_idx = 0  # Accelerate forward, don't turn
action_backward_idx = 6  # Go backward, don't turn

distance_between_checkpoints = 25
road_width = 40  ## a little bit of margin, could be closer to 24 probably ? Don't take risk there are curvy roads
max_allowable_distance_to_checkpoint = np.sqrt(
    (distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2
)

zone_centers_jitter = 0.0  # TODO : eval with zero jitter on zone centers !!
