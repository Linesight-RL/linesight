import numpy as np
import psutil

is_pb_desktop = psutil.virtual_memory().total > 5e10

W_screen = 640
H_screen = 480

W_downsized = 160
H_downsized = 120

wind32gui_margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

running_speed = 50
tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action
n_zone_centers_in_inputs = 40
cutoff_rollout_if_race_not_finished_within_duration_ms = 300_000
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 25_000

temporal_mini_race_duration_ms = 7000
temporal_mini_race_duration_actions = temporal_mini_race_duration_ms / ms_per_action  # 120
# If mini_race_time == mini_race_duration this is the end of the minirace

epsilon = 0.03
epsilon_boltzmann = 0.03
tau_epsilon_boltzmann = 0.01
tau_greedy_boltzmann = 0
discard_non_greedy_actions_in_nsteps = True
buffer_test_ratio = 0.05

n_steps = 3
constant_reward_per_ms = -3 / 5000
reward_per_m_advanced_along_centerline = 5 / 500

gamma = 1
reward_per_ms_press_forward = 0.5 / 5000
float_input_dim = 21 + 3 * n_zone_centers_in_inputs
float_hidden_dim = 256
conv_head_output_dim = 5632
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8
iqn_k = 32
iqn_kappa = 1
AL_alpha = 0

memory_size = 750_000 if is_pb_desktop else 750_000
memory_size_start_learn = 300_000
number_times_single_memory_is_used_before_discard = 32  # 32 // 4
offset_cumul_number_single_memories_used = memory_size_start_learn * number_times_single_memory_is_used_before_discard
# Sign and effet of offset_cumul_number_single_memories_used:
# Positive : We need to generate more memories before we start learning.
# Negative : The first memories we generate will be used for more batches.
number_memories_generated_high_exploration_early_training = 100_000
high_exploration_ratio = 3
batch_size = 2048
learning_rate = 0.2*5e-5
weight_decay = 0.2*1e-6


number_memories_trained_on_between_target_network_updates = 10000
subsample_n_mini_races = 100000000000  # disable TODO REMOVE

soft_update_tau = 0.1

float_inputs_mean = np.array(
    [
        temporal_mini_race_duration_actions / 2,
        #######
        0.8,
        0.2,
        0.3,
        0.3,  # Previous action
        0.1,
        0.1,
        0.1,
        0.1,
        0.3,
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
        -1.30e00,
        -1.30e00,
        9.00e-01,
        -0.975,
        -0.975,
        3.4125,
        -0.65,
        -0.65,
        5.925,
        -0.325,
        -0.325,
        8.4375,
        0,
        0,
        10.95,
        0.325,
        0.325,
        13.4625,
        0.65,
        0.65,
        15.975,
        0.975,
        0.975,
        18.4875,
        1.3,
        1.3,
        21,
        1.625,
        1.625,
        23.5125,
        1.95,
        1.95,
        26.025,
        2.275,
        2.275,
        28.5375,
        2.6,
        2.6,
        31.05,
        2.925,
        2.925,
        33.5625,
        3.25,
        3.25,
        36.075,
        3.575,
        3.575,
        38.5875,
        3.9,
        3.9,
        41.1,
        4.225,
        4.225,
        43.6125,
        4.55,
        4.55,
        46.125,
        4.875,
        4.875,
        48.6375,
        5.2,
        5.2,
        51.15,
        5.525,
        5.525,
        53.6625,
        5.85,
        5.85,
        56.175,
        6.175,
        6.175,
        58.6875,
        6.5,
        6.5,
        61.2,
        6.825,
        6.825,
        63.7125,
        7.15,
        7.15,
        66.225,
        7.475,
        7.475,
        68.7375,
        7.8,
        7.8,
        71.25,
        8.125,
        8.125,
        73.7625,
        8.45,
        8.45,
        76.275,
        8.775,
        8.775,
        78.7875,
        9.1,
        9.1,
        81.3,
        9.425,
        9.425,
        83.8125,
        9.75,
        9.75,
        86.3250000000001,
        10.075,
        10.075,
        88.8375000000001,
        10.4,
        10.4,
        91.3500000000001,
        10.725,
        10.725,
        93.8625000000001,
        11.05,
        11.05,
        96.3750000000001,
        1.17e01,
        1.17e01,
        1.01e02,
        # ==================== END   40 CP =====================
        # ==================== BEGIN 16 CP =====================
        # -1.30e00,
        # -1.30e00,
        # 9.00e-01,
        # 1.00e-01,
        # 1.00e-01,
        # 2.26e01,
        # 1.70e00,
        # 1.70e00,
        # 4.18e01,
        # 2.90e00,
        # 2.90e00,
        # 5.75e01,
        # 3.90e00,
        # 3.90e00,
        # 6.96e01,
        # 4.80e00,
        # 4.80e00,
        # 7.82e01,
        # 5.50e00,
        # 5.50e00,
        # 8.40e01,
        # 6.10e00,
        # 6.10e00,
        # 8.77e01,
        # 7.10e00,
        # 7.10e00,
        # 9.03e01,
        # 8.20e00,
        # 8.20e00,
        # 9.22e01,
        # 9.50e00,
        # 9.50e00,
        # 9.38e01,
        # 1.07e01,
        # 1.07e01,
        # 9.55e01,
        # 1.14e01,
        # 1.14e01,
        # 9.71e01,
        # 1.18e01,
        # 1.18e01,
        # 9.85e01,
        # 1.20e01,
        # 1.20e01,
        # 1.00e02,
        # 1.17e01,
        # 1.17e01,
        # 1.01e02,
        # ==================== END 16 CP =====================
    ]
)

float_inputs_std = np.array(
    [
        temporal_mini_race_duration_actions / 2,
        #######
        0.5,
        0.5,
        0.5,
        0.5,  # Previous action
        0.5,
        0.5,
        0.5,
        0.5,
        1,
        2,
        3000,  # Car gear and wheels
        0.5,
        1,
        0.5,  # Angular velocity
        #######
        5,
        5,
        20,
        0.5,
        0.5,
        0.5,
        # ==================== BEGIN 40 CP =====================
        7.20e00,
        7.20e00,
        1.06e01,
        11.74,
        11.74,
        14.2475,
        16.28,
        16.28,
        17.895,
        20.82,
        20.82,
        21.5425,
        25.36,
        25.36,
        25.19,
        29.9,
        29.9,
        28.8375,
        34.44,
        34.44,
        32.485,
        38.98,
        38.98,
        36.1325,
        43.52,
        43.52,
        39.78,
        48.06,
        48.06,
        43.4275,
        52.6,
        52.6,
        47.075,
        57.14,
        57.14,
        50.7225,
        61.68,
        61.68,
        54.37,
        66.22,
        66.22,
        58.0175,
        70.76,
        70.76,
        61.665,
        75.3,
        75.3,
        65.3125,
        79.84,
        79.84,
        68.96,
        84.38,
        84.38,
        72.6075,
        88.92,
        88.92,
        76.255,
        93.46,
        93.46,
        79.9025,
        98,
        98,
        83.55,
        102.54,
        102.54,
        87.1975,
        107.08,
        107.08,
        90.845,
        111.62,
        111.62,
        94.4925,
        116.16,
        116.16,
        98.1399999999999,
        120.7,
        120.7,
        101.7875,
        125.24,
        125.24,
        105.435,
        129.78,
        129.78,
        109.0825,
        134.32,
        134.32,
        112.73,
        138.86,
        138.86,
        116.3775,
        143.4,
        143.4,
        120.025,
        147.94,
        147.94,
        123.6725,
        152.48,
        152.48,
        127.32,
        157.02,
        157.02,
        130.9675,
        161.56,
        161.56,
        134.615,
        166.1,
        166.1,
        138.2625,
        170.64,
        170.64,
        141.91,
        175.18,
        175.18,
        145.5575,
        179.72,
        179.72,
        149.205,
        1.89e02,
        1.89e02,
        1.57e02,
        # ==================== END   40 CP =====================
        # ==================== BEGIN 16 CP =====================
        # 7.20e00,
        # 7.20e00,
        # 1.06e01,
        # 1.10e01,
        # 1.10e01,
        # 1.38e01,
        # 2.03e01,
        # 2.03e01,
        # 2.10e01,
        # 3.30e01,
        # 3.30e01,
        # 3.05e01,
        # 4.76e01,
        # 4.76e01,
        # 4.15e01,
        # 6.27e01,
        # 6.27e01,
        # 5.34e01,
        # 7.74e01,
        # 7.74e01,
        # 6.61e01,
        # 9.16e01,
        # 9.16e01,
        # 7.90e01,
        # 1.06e02,
        # 1.06e02,
        # 9.14e01,
        # 1.20e02,
        # 1.20e02,
        # 1.03e02,
        # 1.34e02,
        # 1.34e02,
        # 1.12e02,
        # 1.46e02,
        # 1.46e02,
        # 1.22e02,
        # 1.58e02,
        # 1.58e02,
        # 1.31e02,
        # 1.68e02,
        # 1.68e02,
        # 1.40e02,
        # 1.79e02,
        # 1.79e02,
        # 1.48e02,
        # 1.89e02,
        # 1.89e02,
        # 1.57e02,
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

distance_between_checkpoints = 10
road_width = 40  ## a little bit of margin, could be closer to 24 probably ? Don't take risk there are curvy roads
max_allowable_distance_to_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

zone_centers_jitter = 0.0  # TODO : eval with zero jitter on zone centers !!
good_time_save_all_ms = 0

timeout_during_run_ms = 2_100
timeout_between_runs_ms = 300_000

explo_races_per_eval_race = 5
