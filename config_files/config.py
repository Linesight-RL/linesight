"""
This file contains a run's configuration.
It is expected that this file contains all relevant information about a run.

Two files named "config.py" and "config_copy.py" coexist in the same folder.

At the beginning of training, parameters are copied from config.py to config_copy.py
During training, config_copy.py will be reloaded at regular time intervals.
config_copy.py is NOT tracked in git, as it is essentially a temporary file.

Training parameters modifications made during training in config_copy.py will be applied on the fly
without losing the existing content of the replay buffer.

The content of config.py may be modified after starting a run: it will have no effect on the ongoing run.
This setup provides the possibility to:
  1) Modify training parameters on the fly
  2) Continue to code, use git, and modify config.py without impacting an ongoing run.

This file is preconfigured with sensible hyperparameters for the map ESL-Hockolicious, assuming the user
has a computer with 16GB RAM.
"""
from itertools import repeat

from config_files.inputs_list import *
from config_files.state_normalization import *
from config_files.user_config import *

W_downsized = 160
H_downsized = 120

run_name = "run_name_to_be_changed"
running_speed = 80

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
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2_000

temporal_mini_race_duration_ms = 7000
temporal_mini_race_duration_actions = temporal_mini_race_duration_ms // ms_per_action
oversample_long_term_steps = 40
oversample_maximum_term_steps = 5
min_horizon_to_update_priority_actions = temporal_mini_race_duration_actions - 40
# If mini_race_time == mini_race_duration this is the end of the minirace
margin_to_announce_finish_meters = 700

global_schedule_speed = 1

epsilon_schedule = [
    (0, 1),
    (50_000, 1),
    (300_000, 0.1),
    (3_000_000 * global_schedule_speed, 0.03),
]
epsilon_boltzmann_schedule = [
    (0, 0.15),
    (3_000_000 * global_schedule_speed, 0.03),
]
tau_epsilon_boltzmann = 0.01
discard_non_greedy_actions_in_nsteps = True
buffer_test_ratio = 0.05

engineered_speedslide_reward_schedule = [
    (0, 0),
]
engineered_neoslide_reward_schedule = [
    (0, 0),
]
engineered_kamikaze_reward_schedule = [
    (0, 0),
]
engineered_close_to_vcp_reward_schedule = [
    (0, 0),
]

n_steps = 3
constant_reward_per_ms = -6 / 5000
reward_per_m_advanced_along_centerline = 5 / 500

float_input_dim = 27 + 3 * n_zone_centers_in_inputs + 4 * n_prev_actions_in_inputs + 4 * n_contact_material_physics_behavior_types + 1
float_hidden_dim = 256
conv_head_output_dim = 5632
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8  # must be an even number because we sample tau symmetrically around 0.5
iqn_k = 32  # must be an even number because we sample tau symmetrically around 0.5
iqn_kappa = 5e-3
use_ddqn = False

prio_alpha = np.float32(0)  # Rainbow-IQN paper: 0.2, Rainbow paper: 0.5, PER paper 0.6
prio_epsilon = np.float32(2e-3)  # Defaults to 10^-6 in stable-baselines
prio_beta = np.float32(1)

number_times_single_memory_is_used_before_discard = 32  # 32 // 4

memory_size_schedule = [
    (0, (50_000, 20_000)),
    (5_000_000 * global_schedule_speed, (100_000, 75_000)),
    (7_000_000 * global_schedule_speed, (200_000, 150_000)),
]
lr_schedule = [
    (0, 1e-3),
    (3_000_000 * global_schedule_speed, 5e-5),
    (12_000_000 * global_schedule_speed, 5e-5),
    (15_000_000 * global_schedule_speed, 1e-5),
]
tensorboard_suffix_schedule = [
    (0, ""),
    (6_000_000 * global_schedule_speed, "_2"),
    (15_000_000 * global_schedule_speed, "_3"),
    (30_000_000 * global_schedule_speed, "_4"),
    (45_000_000 * global_schedule_speed, "_5"),
    (80_000_000 * global_schedule_speed, "_6"),
    (150_000_000 * global_schedule_speed, "_7"),
]
gamma_schedule = [
    (0, 0.999),
    (1_500_000, 0.999),
    (2_500_000, 1),
]

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
n_transitions_to_plot_in_distribution_curves = 0

distance_between_checkpoints = 0.5
road_width = 90  ## a little bit of margin, could be closer to 24 probably ? Don't take risks there are curvy roads
max_allowable_distance_to_virtual_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

timeout_during_run_ms = 10_100
timeout_between_runs_ms = 600_000_000
tmi_protection_timeout_s = 500
game_reboot_interval = 3600 * 12  # In seconds

frames_before_save_best_runs = 1_500_000

plot_race_time_left_curves = False
make_highest_prio_figures = False
apply_randomcrop_augmentation = False
n_pixels_to_crop_on_each_side = 2

max_rollout_queue_size = 1

use_jit = True
gpu_collectors_count = 4

send_shared_network_every_n_batches = 10
update_inference_network_every_n_actions = 20

target_self_loss_clamp_ratio = 4

final_speed_reward_as_if_duration_s = 0
final_speed_reward_per_m_per_s = reward_per_m_advanced_along_centerline * final_speed_reward_as_if_duration_s

shaped_reward_dist_to_cur_vcp = -0.1
shaped_reward_min_dist_to_cur_vcp = 2
shaped_reward_max_dist_to_cur_vcp = 25
engineered_reward_min_dist_to_cur_vcp = 5
engineered_reward_max_dist_to_cur_vcp = 25
shaped_reward_point_to_vcp_ahead = 0

threshold_to_save_all_runs_ms = 53700

deck_height = -np.inf
game_camera_number = 1

sync_virtual_and_real_checkpoints = True

""" 
============================================      MAP CYCLE     =======================================================

In this section we define the map cycle.

It is a list of iterators, each iterator must return tuples with the following information:
    - short map name        (string):     for logging purposes
    - map path              (string):     to automatically load the map in game. 
                                          This is the same map name as the "map" command in the TMInterface console.
    - reference line path   (string):     where to find the reference line for this map
    - is_explo              (boolean):    whether the policy when running on this map should be exploratory
    - fill_buffer           (boolean):    whether the memories generated during this run should be placed in the buffer 

The map cycle may seem complex at first glance, but it provides a large amount of flexibility:
    - can train on some maps, test blindly on others
    - can train more on some maps, less on others
    - can define multiple reference lines for a given map
    - etc...

The example below defines a simple cycle where the agent alternates between four exploratory runs on map5, and one 
evaluation run on the same map.

map_cycle = [
    repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", True, True), 4),
    repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", False, True), 1),
]
"""

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
# for map_name in nadeo_maps_to_train_and_test:
# short_map_name = map_name[0:3]
# map_cycle.append(repeat((short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", True, True), 4))
# map_cycle.append(repeat((short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", False, True), 1))


map_cycle += [
    # repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", True, True), 4),
    # repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", False, True), 1),
    # repeat(("map8", '"My Challenges/Map8.Challenge.Gbx"', "map8_0.5m_cl.npy", True, True), 4),
    # repeat(("map8", '"My Challenges/Map8.Challenge.Gbx"', "map8_0.5m_cl.npy", False, True), 1),
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
    repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", True, True), 4),
    repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", False, True), 1),
    # repeat(("A02", f'"Official Maps\A02-Race.Challenge.Gbx"', "A02-Race_0.5m_cl2.npy", False, False), 1),
    # repeat(("yellowmile", f'"The Yellow Mile_.Challenge.Gbx"', "YellowMile_0.5m_cl.npy", False, False), 1),
    # repeat(("te86", f'"te 86.Challenge.Gbx"', "te86_0.5m_cl.npy", False, False), 1),
    # repeat(("minishort037", f'"Mini-Short.037.Challenge.Gbx"', "minishort037_0.5m_cl.npy", False, False), 1),
    # repeat(("map3", '"My Challenges\Map3_nowalls.Challenge.Gbx"', "map3_0.5m_cl.npy", False, False), 1),
    # repeat(("wallb1", "Wallbang_full.Challenge.Gbx", "Wallbang_full_0.5m_cl.npy", False, False), 1),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", False, False), 1),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_cl2.npy", True, True), 4),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_cl2.npy", False, True), 1),
    # repeat(("A02", f'"Official Maps\A02-Race.Challenge.Gbx"', f"A02-Race_0.5m_alyen.npy", True, True), 4),
    # repeat(("A02", f'"Official Maps\A02-Race.Challenge.Gbx"', f"A02-Race_0.5m_alyen.npy", False, True), 1),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_rollin.npy", True, True), 4),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_rollin.npy", False, True), 1),
    # repeat(("A11", f'"Official Maps\A11-Race.Challenge.Gbx"', f"A11-Race_0.5m_cl2.npy", True, True), 4),
    # repeat(("A11", f'"Official Maps\A11-Race.Challenge.Gbx"', f"A11-Race_0.5m_cl2.npy", False, True), 1),
    # repeat(("A15", f'"Official Maps\A15-Speed.Challenge.Gbx"', f"A15-Speed_0.5m_hefest.npy", True, True), 4),
    # repeat(("A15", f'"Official Maps\A15-Speed.Challenge.Gbx"', f"A15-Speed_0.5m_hefest.npy", False, True), 1),
    # repeat(("E02", f'"Official Maps\E02-Endurance.Challenge.Gbx"', f"E02-Endurance_0.5m_karjen.npy", True, True), 4),
    # repeat(("E02", f'"Official Maps\E02-Endurance.Challenge.Gbx"', f"E02-Endurance_0.5m_karjen.npy", False, True), 1),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo-levon.npy", True, True), 4),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo-levon.npy", False, True), 1),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo.npy", True, True), 4),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo.npy", False, True), 1),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_darkbringer.npy", True, True), 4),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_darkbringer.npy", False, True), 1),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_linesight2rollin3.npy", True, True), 4),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_linesight2rollin3.npy", False, True), 1),
    # repeat(("D15", '"Official Maps\D15-Endurance.Challenge.Gbx"', f"D15-Endurance_0.5m_gwenlap3.npy", True, True), 4),
    # repeat(("D15", '"Official Maps\D15-Endurance.Challenge.Gbx"', f"D15-Endurance_0.5m_gwenlap3.npy", False, True), 1),
    # repeat(("C12", '"Official Maps\C12-Obstacle.Challenge.Gbx"', f"C12-Obstacle_0.5m_weapon.npy", True, True), 4),
    # repeat(("C12", '"Official Maps\C12-Obstacle.Challenge.Gbx"', f"C12-Obstacle_0.5m_weapon.npy", False, True), 1),
    # repeat(("D15olnc", '"D15-Endurance True One Lap No Cut.Challenge.Gbx"', f"D15-OnelapNocut_0.5m_wirtual.npy", True, True), 4),
    # repeat(("D15olnc", '"D15-Endurance True One Lap No Cut.Challenge.Gbx"', f"D15-OnelapNocut_0.5m_wirtual.npy", False, True), 1),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_racehansnocutlap3.npy", True, True), 4),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_racehansnocutlap3.npy", False, True), 1),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_linesight2racehans3.npy", True, True), 4),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_linesight2racehans3.npy", False, True), 1),
    # repeat(("A07", f'"Official Maps/A07-Race.Challenge.Gbx"', f"A07-Race_0.5m_raceta.npy", True, True), 4),
    # repeat(("A07", f'"Official Maps/A07-Race.Challenge.Gbx"', f"A07-Race_0.5m_raceta.npy", False, True), 1),
]
