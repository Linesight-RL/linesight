from itertools import chain, repeat

import numpy as np
import psutil

is_pb_desktop = psutil.virtual_memory().total > 5e10

W_screen = 640
H_screen = 480

W_downsized = 160
H_downsized = 120

run_name = "110"
running_speed = 50

tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action
n_zone_centers_in_inputs = 40
n_prev_actions_in_inputs = 5
n_contact_material_physics_behavior_types = 4  # See contact_materials.py
cutoff_rollout_if_race_not_finished_within_duration_ms = 300_000
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 25_000

temporal_mini_race_duration_ms = 7000
temporal_mini_race_duration_actions = temporal_mini_race_duration_ms / ms_per_action
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
reward_per_ms_press_forward_early_training = 0.5 / 5000
float_input_dim = 26 + 3 * n_zone_centers_in_inputs + 4 * n_prev_actions_in_inputs + 4 * n_contact_material_physics_behavior_types
float_hidden_dim = 256
conv_head_output_dim = 5632
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8
iqn_k = 32
iqn_kappa = 1
AL_alpha = 0

memory_size = 800_000 if is_pb_desktop else 50_000
memory_size_start_learn = 1_000
number_times_single_memory_is_used_before_discard = 64  # 32 // 4
number_times_single_memory_is_used_before_discard_reset = 64 - number_times_single_memory_is_used_before_discard
offset_cumul_number_single_memories_used = memory_size_start_learn * number_times_single_memory_is_used_before_discard
# Sign and effet of offset_cumul_number_single_memories_used:
# Positive : We need to generate more memories before we start learning.
# Negative : The first memories we generate will be used for more batches.
number_memories_generated_high_exploration_early_training = 100_000
apply_horizontal_flip_augmentation = False
flip_augmentation_ratio = 0.5
flip_pair_indices_to_swap = [
    (3, 4),  # previous action left/right
    (7, 8),  # previous**2 action left/right
    (11, 12),  # previous**3 action left/right
    (15, 16),  # previous**4 action left/right
    (19, 20),  # previous**5 action left/right
    (21, 22),  # front wheels sliding
    (23, 24),  # back wheels sliding
    (25, 26),  # front wheels has_ground_contact
    (27, 28),  # back wheels has_ground_contact
    (29, 30),  # front wheels damper_absorb
    (31, 32),  # back wheels damper_absorb
    (37, 41),  # front wheels physics behavior 0
    (38, 42),  # front wheels physics behavior 1
    (39, 43),  # front wheels physics behavior 2
    (40, 44),  # front wheels physics behavior 3
    (45, 49),  # back wheels physics behavior 0
    (46, 50),  # back wheels physics behavior 1
    (47, 51),  # back wheels physics behavior 2
    (48, 52),  # back wheels physics behavior 3
]

flip_indices_floats_before_swap = list(chain(*flip_pair_indices_to_swap))
flip_indices_floats_after_swap = list(chain(*map(reversed, flip_pair_indices_to_swap)))

indices_floats_sign_inversion = [
    54,  # state_car_angular_velocity_in_car_reference_system.y
    55,  # state_car_angular_velocity_in_car_reference_system.z
    56,  # state_car_velocity_in_car_reference_system.x
    59,  # state_y_map_vector_in_car_reference_system.x
] + [62 + i * 3 for i in range(n_zone_centers_in_inputs)]

high_exploration_ratio = 3
batch_size = 512
lr_schedule = [
    (0, 2e-4),
    (1_000_000, 2e-4),
    (2_500_000, 5e-5),
    (3_600_000, 1e-4),  # start 800k memory
    (7_000_000, 5e-5),
    (12_800_000, 5e-5),
    (13_000_000, 1e-5),
    (14_800_000, 5e-5),
    (19_300_000, 5e-5),
    (19_500_000, 1e-5),
    (19_600_000, 5e-5),
]
weight_decay_lr_ratio = 1 / 50


reset_frequency = 100 #in loops
reset_mul_factor = 0.8

adam_epsilon = 1e-4
clip_grad_value = 100
clip_grad_norm = 30.0

number_memories_trained_on_between_target_network_updates = 20_000
soft_update_tau = 0.1

float_inputs_mean = np.array(
    [
        temporal_mini_race_duration_actions / 2,
        #######
        0.8,
        0.2,
        0.3,
        0.3,
        0.8,
        0.2,
        0.3,
        0.3,
        0.8,
        0.2,
        0.3,
        0.3,
        0.8,
        0.2,
        0.3,
        0.3,
        0.8,
        0.2,
        0.3,
        0.3,
        # Previous action
        0.1,
        0.1,
        0.1,
        0.1,
        0.9,
        0.9,
        0.9,
        0.9,
        0.02,
        0.02,
        0.02,
        0.02,
        0.3,
        2.5,
        7000,
        0.1,  # Car gear and wheels
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,  # Wheel contact material types
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
    ]
)

float_inputs_std = np.array(
    [
        temporal_mini_race_duration_actions / 2,
        #######
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,  # Previous action
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.06,
        0.06,
        0.06,
        0.06,
        1,
        2,
        3000,
        10,  # Car gear and wheels
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,  # Wheel contact material types
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
road_width = 40  ## a little bit of margin, could be closer to 24 probably ? Don't take risks there are curvy roads
max_allowable_distance_to_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

zone_centers_jitter = 0.0  # TODO : eval with zero jitter on zone centers !!

timeout_during_run_ms = 2_100
timeout_between_runs_ms = 600_000_000
tmi_protection_timeout_s = 500 if is_pb_desktop else 60


map_cycle = [
    repeat(("map5", '"My Challenges\Map5.Challenge.Gbx"', "map5_10m_cl.npy", True, True, False), 4),
    repeat(("map5", '"My Challenges\Map5.Challenge.Gbx"', "map5_10m_cl.npy", False, True, True), 1),
    # repeat(("A06", '"Official Maps\White\A06-Obstacle.Challenge.Gbx"', "A06-Obstacle_10m_cl.npy", True, True, False), 4),
    # repeat(("A06", '"Official Maps\White\A06-Obstacle.Challenge.Gbx"', "A06-Obstacle_10m_cl.npy", False, True, False), 1),
    # repeat(("A07", '"Official Maps\White\A07-Race.Challenge.Gbx"', "A07-Race_10m_cl.npy", True, True, False), 4),
    # repeat(("A07", '"Official Maps\White\A07-Race.Challenge.Gbx"', "A07-Race_10m_cl.npy", False, True, False), 1),
    # repeat(("B01", '"Official Maps\Green\B01-Race.Challenge.Gbx"', "B01-Race_10m_cl.npy", True, True, False), 4),
    # repeat(("B01", '"Official Maps\Green\B01-Race.Challenge.Gbx"', "B01-Race_10m_cl.npy", False, True, False), 1),
    # repeat(("B02", '"Official Maps\Green\B02-Race.Challenge.Gbx"', "B02-Race_10m_cl.npy", True, True, False), 4),
    # repeat(("B02", '"Official Maps\Green\B02-Race.Challenge.Gbx"', "B02-Race_10m_cl.npy", False, True, False), 1),
    # repeat(("B03", '"Official Maps\Green\B03-Race.Challenge.Gbx"', "B03-Race_10m_cl.npy", True, True, False), 4),
    # repeat(("B03", '"Official Maps\Green\B03-Race.Challenge.Gbx"', "B03-Race_10m_cl.npy", False, True, False), 1),
    # repeat(("B05", '"Official Maps\Green\B05-Race.Challenge.Gbx"', "B05-Race_10m_cl.npy", True, True, False), 4),
    # repeat(("B05", '"Official Maps\Green\B05-Race.Challenge.Gbx"', "B05-Race_10m_cl.npy", False, True, False), 1),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_10m_cl_2.npy", True, True, False), 4),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_10m_cl_2.npy", False, True, False), 1),
    # repeat(("A02", '"Official Maps\White\A02-Race.Challenge.Gbx"', "A02-Race_10m_cl.npy", False, False, False), 1),
    # repeat(("map3", '"My Challenges\Map3_nowalls.Challenge.Gbx"', "map3_10m_cl.npy", False, False, True), 1),
]

# repeat(("parrots", '"ESL - Parrots are cool.Challenge.Gbx"', "parrots_are_cool_10m_cl.npy", True, True, False), 4),
# repeat(("parrots", '"ESL - Parrots are cool.Challenge.Gbx"', "parrots_are_cool_10m_cl.npy", False, True, False), 1),
# repeat(("leaveit", '"Leave it behind.Challenge.Gbx"', "leave_it_behind_10m_cl.npy", True, True, False), 4),
# repeat(("leaveit", '"Leave it behind.Challenge.Gbx"', "leave_it_behind_10m_cl.npy", False, True, False), 1),
# repeat(("leavepast", '"Leave the past where it belongs..Challenge.Gbx"', "leave_past_belong_10m_cl.npy", True, True,
#         False), 4),
# repeat(("leavepast", '"Leave the past where it belongs..Challenge.Gbx"', "leave_past_belong_10m_cl.npy", False, True,
#         False), 1),
