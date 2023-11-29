import os
from itertools import chain, repeat
from pathlib import Path
from sys import platform

import numpy as np
import psutil

is_linux = platform in ["linux", "linux2"]
is_pb_desktop = psutil.virtual_memory().total > 5e10

W_screen = 640
H_screen = 480

W_downsized = 160
H_downsized = 120

run_name = "run_name_to_be_changed"
running_speed = 80 if is_pb_desktop else 100

tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action
n_zone_centers_in_inputs = 40
one_every_n_zone_centers_in_inputs = 20
n_zone_centers_extrapolate_after_end_of_map = 1000
n_zone_centers_extrapolate_before_start_of_map = 20
n_prev_actions_in_inputs = 5
n_contact_material_physics_behavior_types = 4  # See contact_materials.py
cutoff_rollout_if_race_not_finished_within_duration_ms = 300_000
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2_500

temporal_mini_race_duration_ms = 7000
temporal_mini_race_duration_actions = temporal_mini_race_duration_ms // ms_per_action
oversample_long_term_steps = 40
oversample_maximum_term_steps = 5
min_horizon_to_update_priority_actions = temporal_mini_race_duration_actions - 40
# If mini_race_time == mini_race_duration this is the end of the minirace
margin_to_announce_finish_meters = 700

epsilon_schedule = [
    (0, 0.1),
    (1_000_000, 0.03),
]
epsilon_boltzmann_schedule = [
    (0, 0.15),
    (1_000_000, 0.03),
]
tau_epsilon_boltzmann = 0.01
discard_non_greedy_actions_in_nsteps = True
buffer_test_ratio = 0.05

n_steps = 3
constant_reward_per_ms = -6 / 5000
reward_per_m_advanced_along_centerline = 5 / 500

gamma = 1
reward_per_ms_press_forward_schedule = [
    (0, 0),
]
float_input_dim = 26 + 3 * n_zone_centers_in_inputs + 4 * n_prev_actions_in_inputs + 4 * n_contact_material_physics_behavior_types + 1
float_hidden_dim = 256
conv_head_output_dim = 5632
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8  # must be an even number because we sample tau symmetrically around 0.5
iqn_k = 32  # must be an even number because we sample tau symmetrically around 0.5
iqn_kappa = 1
use_ddqn = False

prio_alpha = np.float32(0)  # Rainbow-IQN paper: 0.2, Rainbow paper: 0.5, PER paper 0.6
prio_epsilon = np.float32(2e-3)  # Defaults to 10^-6 in stable-baselines
prio_beta = np.float32(1)

number_times_single_memory_is_used_before_discard = 60  # 32 // 4

memory_size_schedule = [
    (0, (50_000, 20_000)),
    (3_000_000, (800_000, 200_000)),
]
lr_schedule = [
    (0, 1e-3),
    (2_000_000, 5e-5),
    (6_000_000, 5e-5),
    (8_000_000, 1e-5),
]
tensorboard_suffix_schedule = [
    (0, ""),
    # (3_000_000, "_1"),
    # (15_000_000, "_2"),
    # (30_000_000, "_3"),
]

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

batch_size = 512
weight_decay_lr_ratio = 1 / 50
adam_epsilon = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

single_reset_flag = 0
reset_every_n_frames_generated = 400_000_00000000
additional_transition_after_reset = 1_600_000
last_layer_reset_factor = 0.8  # 0 : full reset, 1 : nothing happens
overall_reset_mul_factor = 0.01  # 0 : nothing happens ; 1 : full reset

clip_grad_value = 1000
clip_grad_norm = 30

number_memories_trained_on_between_target_network_updates = 2048
soft_update_tau = 0.02

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
        0.00e00,
        -2.10e00,
        -2.00e-01,
        0.00e00,
        -2.20e00,
        9.50e00,
        0.00e00,
        -2.20e00,
        1.91e01,
        0.00e00,
        -2.20e00,
        2.85e01,
        0.00e00,
        -2.20e00,
        3.77e01,
        0.00e00,
        -2.10e00,
        4.66e01,
        0.00e00,
        -2.00e00,
        5.52e01,
        0.00e00,
        -1.80e00,
        6.35e01,
        0.00e00,
        -1.50e00,
        7.14e01,
        0.00e00,
        -1.30e00,
        7.89e01,
        0.00e00,
        -1.00e00,
        8.62e01,
        0.00e00,
        -7.00e-01,
        9.31e01,
        0.00e00,
        -4.00e-01,
        9.98e01,
        0.00e00,
        -0.00e00,
        1.06e02,
        0.00e00,
        3.00e-01,
        1.12e02,
        0.00e00,
        7.00e-01,
        1.18e02,
        0.00e00,
        1.10e00,
        1.24e02,
        0.00e00,
        1.60e00,
        1.29e02,
        0.00e00,
        2.10e00,
        1.34e02,
        0.00e00,
        2.60e00,
        1.39e02,
        0.00e00,
        3.10e00,
        1.43e02,
        0.00e00,
        3.70e00,
        1.48e02,
        0.00e00,
        4.20e00,
        1.52e02,
        0.00e00,
        4.80e00,
        1.56e02,
        0.00e00,
        5.30e00,
        1.60e02,
        0.00e00,
        5.80e00,
        1.64e02,
        0.00e00,
        6.40e00,
        1.67e02,
        0.00e00,
        6.90e00,
        1.71e02,
        0.00e00,
        7.40e00,
        1.74e02,
        0.00e00,
        7.80e00,
        1.77e02,
        0.00e00,
        8.30e00,
        1.80e02,
        0.00e00,
        8.70e00,
        1.82e02,
        0.00e00,
        9.10e00,
        1.85e02,
        0.00e00,
        9.50e00,
        1.87e02,
        0.00e00,
        9.90e00,
        1.89e02,
        0.00e00,
        1.02e01,
        1.91e02,
        0.00e00,
        1.06e01,
        1.93e02,
        0.00e00,
        1.09e01,
        1.94e02,
        0.00e00,
        1.12e01,
        1.96e02,
        0.00e00,
        1.15e01,
        1.97e02,
        # ==================== END   40 CP =====================
        margin_to_announce_finish_meters,
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
        7.70e00,
        7.70e00,
        2.90e00,
        7.80e00,
        7.80e00,
        3.30e00,
        8.30e00,
        8.30e00,
        4.10e00,
        9.50e00,
        9.50e00,
        5.10e00,
        1.15e01,
        1.15e01,
        6.40e00,
        1.43e01,
        1.43e01,
        7.90e00,
        1.78e01,
        1.78e01,
        9.70e00,
        2.19e01,
        2.19e01,
        1.17e01,
        2.63e01,
        2.63e01,
        1.42e01,
        3.11e01,
        3.11e01,
        1.70e01,
        3.61e01,
        3.61e01,
        2.01e01,
        4.13e01,
        4.13e01,
        2.35e01,
        4.66e01,
        4.66e01,
        2.72e01,
        5.20e01,
        5.20e01,
        3.11e01,
        5.74e01,
        5.74e01,
        3.52e01,
        6.30e01,
        6.30e01,
        3.94e01,
        6.85e01,
        6.85e01,
        4.37e01,
        7.42e01,
        7.42e01,
        4.81e01,
        7.99e01,
        7.99e01,
        5.25e01,
        8.56e01,
        8.56e01,
        5.70e01,
        9.14e01,
        9.14e01,
        6.14e01,
        9.72e01,
        9.72e01,
        6.58e01,
        1.03e02,
        1.03e02,
        7.02e01,
        1.09e02,
        1.09e02,
        7.46e01,
        1.15e02,
        1.15e02,
        7.91e01,
        1.21e02,
        1.21e02,
        8.35e01,
        1.27e02,
        1.27e02,
        8.80e01,
        1.32e02,
        1.32e02,
        9.26e01,
        1.38e02,
        1.38e02,
        9.72e01,
        1.44e02,
        1.44e02,
        1.02e02,
        1.49e02,
        1.49e02,
        1.07e02,
        1.55e02,
        1.55e02,
        1.12e02,
        1.60e02,
        1.60e02,
        1.17e02,
        1.66e02,
        1.66e02,
        1.21e02,
        1.71e02,
        1.71e02,
        1.26e02,
        1.76e02,
        1.76e02,
        1.31e02,
        1.82e02,
        1.82e02,
        1.36e02,
        1.87e02,
        1.87e02,
        1.41e02,
        1.92e02,
        1.92e02,
        1.46e02,
        1.98e02,
        1.98e02,
        1.51e02,
        # ==================== END   40 CP =====================
        margin_to_announce_finish_meters / 2,
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
road_width = 100  ## a little bit of margin, could be closer to 24 probably ? Don't take risks there are curvy roads
max_allowable_distance_to_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

zone_centers_jitter = 0.0  # TODO : eval with zero jitter on zone centers !!

timeout_during_run_ms = 10_100
timeout_between_runs_ms = 600_000_000
tmi_protection_timeout_s = 500 if is_pb_desktop else 60
game_reboot_interval = 3600 * 4  # In seconds

frames_before_save_best_runs = 1_500_000


nadeo_maps_to_train_and_test = [
    "A01-Race",
    # "A02-Race",
    "A03-Race",
    # "A04-Acrobatic",
    "A05-Race",
    # "A06-Obstacle",
    "A07-Race",
    # "A08-Endurance",
    # "A09-Race",
    # "A10-Acrobatic",
    "A11-Race",
    # "A12-Speed",
    # "A13-Race",
    "A14-Race",
    "A15-Speed",
    "B01-Race",
    "B02-Race",
    "B03-Race",
    # "B04-Acrobatic",
    "B05-Race",
    # "B06-Obstacle",
    # "B07-Race",
    # "B08-Endurance",
    # "B09-Acrobatic",
    "B10-Speed",
    # "B11-Race",
    # "B12-Race",
    # "B13-Obstacle",
    "B14-Speed",
    # "B15-Race",
]

map_cycle = []
for map_name in nadeo_maps_to_train_and_test:
    short_map_name = map_name[0:3]
    map_cycle.append(repeat((short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", True, True), 4))
    map_cycle.append(repeat((short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", False, True), 1))


map_cycle += [
    repeat(("map5", '"My Challenges\Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", True, True), 4),
    repeat(("map5", '"My Challenges\Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", False, True), 1),
    # repeat(("yosh1", '"My Challenges\Yosh1.Challenge.Gbx"', "yosh1_0.5m_clprog.npy", True, True), 4),
    # repeat(("yosh1", '"My Challenges\Yosh1.Challenge.Gbx"', "yosh1_0.5m_clprog.npy", False, True), 1),
    # repeat(("wallb1", "Wallbang_full.Challenge.Gbx", "Wallbang_full_0.5m_cl.npy", True, True), 4),
    # repeat(("wallb1", "Wallbang_full.Challenge.Gbx", "Wallbang_full_0.5m_cl.npy", False, True), 1),
    # repeat(("yosh3", '"My Challenges\Yosh3.Challenge.Gbx"', "yosh3_0.5m_clprog_cut1.npy", True, True), 4),
    # repeat(("yosh3", '"My Challenges\Yosh3.Challenge.Gbx"', "yosh3_0.5m_clprog_cut1.npy", False, True), 1),
    # repeat(("A06", '"Official Maps\White\A06-Obstacle.Challenge.Gbx"', "A06-Obstacle_10m_cl.npy", True, True), 4),
    # repeat(("A06", '"Official Maps\White\A06-Obstacle.Challenge.Gbx"', "A06-Obstacle_10m_cl.npy", False, True), 1),
    # repeat(("A07", '"Official Maps\White\A07-Race.Challenge.Gbx"', "A07-Race_10m_cl.npy", True, True), 4),
    # repeat(("A07", '"Official Maps\White\A07-Race.Challenge.Gbx"', "A07-Race_10m_cl.npy", False, True), 1),
    # repeat(("B01", '"Official Maps\Green\B01-Race.Challenge.Gbx"', "B01-Race_10m_cl.npy", True, True), 4),
    # repeat(("B01", '"Official Maps\Green\B01-Race.Challenge.Gbx"', "B01-Race_10m_cl.npy", False, True), 1),
    # repeat(("B02", '"Official Maps\Green\B02-Race.Challenge.Gbx"', "B02-Race_10m_cl.npy", True, True), 4),
    # repeat(("B02", '"Official Maps\Green\B02-Race.Challenge.Gbx"', "B02-Race_10m_cl.npy", False, True), 1),
    # repeat(("B03", '"Official Maps\Green\B03-Race.Challenge.Gbx"', "B03-Race_10m_cl.npy", True, True), 4),
    # repeat(("B03", '"Official Maps\Green\B03-Race.Challenge.Gbx"', "B03-Race_10m_cl.npy", False, True), 1),
    # repeat(("B05", '"Official Maps\Green\B05-Race.Challenge.Gbx"', "B05-Race_10m_cl.npy", True, True), 4),
    # repeat(("B05", '"Official Maps\Green\B05-Race.Challenge.Gbx"', "B05-Race_10m_cl.npy", False, True), 1),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", True, True), 4),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", False, True), 1),
    repeat(("A02", f'"Official Maps\A02-Race.Challenge.Gbx"', "A02-Race_0.5m_cl2.npy", False, False), 1),
    repeat(("yellowmile", f'"The Yellow Mile_.Challenge.Gbx"', "YellowMile_0.5m_cl.npy", False, False), 1),
    repeat(("te86", f'"te 86.Challenge.Gbx"', "te86_0.5m_cl.npy", False, False), 1),
    repeat(("minishort037", f'"Mini-Short.037.Challenge.Gbx"', "minishort037_0.5m_cl.npy", False, False), 1),
    repeat(("map3", '"My Challenges\Map3_nowalls.Challenge.Gbx"', "map3_0.5m_cl.npy", False, False), 1),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_cl2.npy", True, True), 4),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_cl2.npy", False, True), 1),
    # repeat(("A11", f'"Official Maps\A11-Race.Challenge.Gbx"', f"A11-Race_0.5m_cl2.npy", True, True), 4),
    # repeat(("A11", f'"Official Maps\A11-Race.Challenge.Gbx"', f"A11-Race_0.5m_cl2.npy", False, True), 1),
]

plot_race_time_left_curves = True if is_pb_desktop else True
make_highest_prio_figures = True
apply_randomcrop_augmentation = True if is_pb_desktop else False
n_pixels_to_crop_on_each_side = 2

max_rollout_queue_size = 1

use_jit = True
base_tmi_port = 8478
gpu_collectors_count = 1

target_python_link_path = (
    Path(os.path.expanduser("~")) / "windocs" / "TMInterface" / "Plugins" / "Python_Link.as"
    if is_pb_desktop and is_linux
    else Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"
)

linux_launch_game_path = ("/mnt/ext4_data/projects/trackmania_rl/scripts/launch_game_pb.sh") if is_pb_desktop else "./launch_game_agade.sh"

send_shared_network_every_n_batches = 10
update_inference_network_every_n_actions = 20
