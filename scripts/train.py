import importlib
import random
import shutil
import time
import typing
from collections import defaultdict
from datetime import datetime
from itertools import chain, count, cycle
from pathlib import Path

import joblib
import numpy as np
import torch
import torch_optimizer
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import ListStorage
from torchrl.data.replay_buffers.samplers import (PrioritizedSampler,
                                                  RandomSampler)

import trackmania_rl.agents.iqn as iqn
from trackmania_rl import (buffer_management, misc, nn_utilities,
                           tm_interface_manager)
from trackmania_rl.buffer_utilities import buffer_collate_function
from trackmania_rl.map_loader import load_next_map_zone_centers
from trackmania_rl.time_parsing import DigitsLibrary, parse_time

base_dir = Path(__file__).resolve().parents[1]

save_dir = base_dir / "save" / misc.run_name
save_dir.mkdir(parents=True, exist_ok=True)
tensorboard_writer = SummaryWriter(log_dir=str(base_dir / "tensorboard" / misc.run_name))

layout = {
    "84": {
        "eval_race_time_finished": [
            "Multiline",
            [
                "eval_race_time_finished",
            ],
        ],
        "explo_race_time_finished": [
            "Multiline",
            [
                "explo_race_time_finished",
            ],
        ],
        "loss_Q": ["Multiline", ["loss_Q$", "loss_Q_test$"]],
        "loss_policy": ["Multiline", ["loss_policy$", "loss_policy_test$"]],
        "loss_alpha": ["Multiline", ["loss_alpha$", "loss_alpha_test$"]],
        "values_starting_frame": [
            "Multiline",
            [f"q_value_starting_frame_{i}" for i in range(len(misc.inputs))],
        ],
        "policy_starting_frame": [
            "Multiline",
            [f"policy_{i}_starting_frame" for i in range(len(misc.inputs))],
        ],
        "single_zone_reached": [
            "Multiline",
            [
                "single_zone_reached",
            ],
        ],
        r"races_finished": ["Multiline", ["explo_race_finished", "eval_race_finished"]],
        "iqn_std": [
            "Multiline",
            [f"std_within_iqn_quantiles_for_action{i}" for i in range(len(misc.inputs))],
        ],
        "race_time_ratio": ["Multiline", ["race_time_ratio"]],
        "mean_action_gap": [
            "Multiline",
            [
                "mean_action_gap",
            ],
        ],
        "layer_L2": [
            "Multiline",
            [
                "layer_.*_L2",
            ],
        ],
        "lr_ratio_L2": [
            "Multiline",
            [
                "lr_ratio_.*_L2",
            ],
        ],
        "exp_avg_L2": [
            "Multiline",
            [
                "exp_avg_.*_L2",
            ],
        ],
        "exp_avg_sq_L2": [
            "Multiline",
            [
                "exp_avg_sq_.*_L2",
            ],
        ],
        "eval_race_time": [
            "Multiline",
            [
                "eval_race_time_[^_]*",
            ],
        ],
        "explo_race_time": [
            "Multiline",
            [
                "explo_race_time_[^_]*",
            ],
        ],
        "sac_alpha": [
            "Multiline",
            [
                "sac_alpha",
            ],
        ],
        "policy_entropy": [
            "Multiline",
            [
                "policy_entropy",
                "policy_entropy",
            ],
        ],
    },
}
tensorboard_writer.add_custom_scalars(layout)

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
random_seed = 444
torch.cuda.manual_seed_all(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


# ========================================================
# Create new stuff
# ========================================================


def make_untrained_SoftIQNQNetwork():
    return iqn.SoftIQNQNetwork(
        float_inputs_dim=misc.float_input_dim,
        float_hidden_dim=misc.float_hidden_dim,
        conv_head_output_dim=misc.conv_head_output_dim,
        dense_hidden_dimension=misc.dense_hidden_dimension,
        iqn_embedding_dimension=misc.iqn_embedding_dimension,
        n_actions=len(misc.inputs),
        float_inputs_mean=misc.float_inputs_mean,
        float_inputs_std=misc.float_inputs_std,
    )


def make_untrained_PolicyNetwork():
    return iqn.LogPolicyNetwork(
        float_inputs_dim=misc.float_input_dim,
        float_hidden_dim=misc.float_hidden_dim,
        conv_head_output_dim=misc.conv_head_output_dim,
        dense_hidden_dimension=misc.dense_hidden_dimension,
        n_actions=len(misc.inputs),
        float_inputs_mean=misc.float_inputs_mean,
        float_inputs_std=misc.float_inputs_std,
    )


soft_Q_model1 = torch.jit.script(make_untrained_SoftIQNQNetwork()).to("cuda", memory_format=torch.channels_last)
soft_Q_model2 = torch.jit.script(make_untrained_SoftIQNQNetwork()).to("cuda", memory_format=torch.channels_last)
policy_model = torch.jit.script(make_untrained_PolicyNetwork()).to("cuda", memory_format=torch.channels_last)
logalpha_model = torch.jit.script(iqn.LogAlphaSingletonNetwork()).to("cuda", memory_format=torch.channels_last)

print(soft_Q_model1)
print(policy_model)
print(logalpha_model)

accumulated_stats: defaultdict[str | typing.Any] = defaultdict(int)
accumulated_stats["alltime_min_ms"] = {}

# ========================================================
# Load existing stuff
# ========================================================
# noinspection PyBroadException
try:
    soft_Q_model1.load_state_dict(torch.load(save_dir / "soft_Q_weights1.torch"))
    soft_Q_model2.load_state_dict(torch.load(save_dir / "soft_Q_weights2.torch"))
    policy_model.load_state_dict(torch.load(save_dir / "policy_weights.torch"))
    logalpha_model.load_state_dict(torch.load(save_dir / "logalpha_weights.torch"))
    print(" =========================     Weights loaded !     ================================")
except:
    print(" Could not load weights")

# noinspection PyBroadException
try:
    accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
    print(" =========================      Stats loaded !      ================================")
except:
    print(" Could not load stats")

accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]
accumulated_stats["reset_counter"] = 0
accumulated_stats["single_reset_counter"] = misc.single_reset_counter

soft_Q_optimizer = torch_optimizer.Lookahead(
    torch.optim.RAdam(
        soft_Q_model1.parameters(),
        lr=nn_utilities.lr_from_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
        eps=misc.adam_epsilon,
        betas=(0.9, 0.95),
    ),
    k=5,
    alpha=0.5,
)
policy_optimizer = torch_optimizer.Lookahead(
    torch.optim.RAdam(
        policy_model.parameters(),
        lr=nn_utilities.lr_from_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
        eps=misc.adam_epsilon,
        betas=(0.9, 0.95),
    ),
    k=5,
    alpha=0.5,
)

logalpha_optimizer = torch.optim.RAdam(
    logalpha_model.parameters(),
    lr=nn_utilities.lr_from_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
    eps=misc.adam_epsilon,
    betas=(0.9, 0.95),
)


soft_Q_scaler = torch.cuda.amp.GradScaler()
policy_scaler = torch.cuda.amp.GradScaler()
logalpha_scaler = torch.cuda.amp.GradScaler()

buffer = ReplayBuffer(
    storage=ListStorage(misc.memory_size),
    batch_size=misc.batch_size,
    collate_fn=buffer_collate_function,
    prefetch=1,
    sampler=PrioritizedSampler(misc.memory_size, misc.prio_alpha, misc.prio_beta, misc.prio_epsilon, torch.float)
    if misc.prio_alpha > 0
    else RandomSampler(),
)
buffer_test = ReplayBuffer(
    storage=ListStorage(int(misc.memory_size * misc.buffer_test_ratio)),
    batch_size=misc.batch_size,
    collate_fn=buffer_collate_function,
    sampler=PrioritizedSampler(misc.memory_size, misc.prio_alpha, misc.prio_beta, misc.prio_epsilon, torch.float)
    if misc.prio_alpha > 0
    else RandomSampler(),
)

# noinspection PyBroadException
try:
    soft_Q_optimizer.load_state_dict(torch.load(save_dir / "soft_Q_optimizer.torch"))
    soft_Q_scaler.load_state_dict(torch.load(save_dir / "soft_Q_scaler.torch"))
    policy_optimizer.load_state_dict(torch.load(save_dir / "policy_optimizer.torch"))
    policy_scaler.load_state_dict(torch.load(save_dir / "policy_scaler.torch"))
    # logalpha_optimizer.load_state_dict(torch.load(save_dir / "logalpha_optimizer.torch"))
    # logalpha_scaler.load_state_dict(torch.load(save_dir / "logalpha_scaler.torch"))
    print(" =========================     Optimizer loaded !     ================================")
except:
    print(" Could not load optimizer")

loss_Q_history = []
loss_Q_test_history = []
loss_policy_history = []
loss_policy_test_history = []
loss_alpha_history = []
loss_alpha_test_history = []
policy_entropy_history = []
policy_entropy_test_history = []
train_on_batch_duration_history = []
grad_norm_history = []
layer_grad_norm_history = defaultdict(list)

# ========================================================
# Make the trainer
# ========================================================
trainer = iqn.Trainer(
    soft_Q_model=soft_Q_model1,
    soft_Q_model2=soft_Q_model2,
    soft_Q_optimizer=soft_Q_optimizer,
    soft_Q_scaler=soft_Q_scaler,
    policy_model=policy_model,
    policy_optimizer=policy_optimizer,
    policy_scaler=policy_scaler,
    logalpha_model=logalpha_model,
    logalpha_optimizer=logalpha_optimizer,
    logalpha_scaler=logalpha_scaler,
    batch_size=misc.batch_size,
    iqn_k=misc.iqn_k,
    iqn_n=misc.iqn_n,
    iqn_kappa=misc.iqn_kappa,
    gamma=misc.gamma,
    truncation_amplitude=misc.truncation_amplitude,
    target_entropy=misc.target_entropy,  # This parameter is typically set to dim(action_space)
    epsilon=nn_utilities.lr_from_schedule(misc.epsilon_schedule, accumulated_stats["cumul_number_memories_generated"]),
)

# ========================================================
# Training loop
# ========================================================
soft_Q_model1.train()
policy_model.train()
logalpha_model.train()
time_last_save = time.time()
tmi = tm_interface_manager.TMInterfaceManager(
    base_dir=base_dir,
    running_speed=misc.running_speed,
    run_steps_per_action=misc.tm_engine_step_per_action,
    max_overall_duration_ms=misc.cutoff_rollout_if_race_not_finished_within_duration_ms,
    max_minirace_duration_ms=misc.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
    interface_name="TMInterface0",
)

map_cycle_str = str(misc.map_cycle)
map_cycle_iter = cycle(chain(*misc.map_cycle))

next_map_tuple = next(map_cycle_iter)
zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
map_name, map_path, zone_centers_filename, is_explo, fill_buffer, save_aggregated_stats = next_map_tuple

# ========================================================
# Warmup numba
# ========================================================
parse_time(
    np.random.randint(low=0, high=256, size=(misc.H_screen, misc.W_screen, 4), dtype=np.uint8),
    DigitsLibrary(base_dir / "data" / "digits_file.npy"),
)

for loop_number in count(1):
    importlib.reload(misc)

    # ===============================================
    #   DID THE CYCLE CHANGE ?
    # ===============================================
    if str(misc.map_cycle) != map_cycle_str:
        map_cycle_str = str(misc.map_cycle)
        map_cycle_iter = cycle(chain(*misc.map_cycle))

    # ===============================================
    #   GET NEXT MAP FROM CYCLE
    # ===============================================

    next_map_tuple = next(map_cycle_iter)
    if next_map_tuple[2] != zone_centers_filename:
        zone_centers = load_next_map_zone_centers(next_map_tuple[2], base_dir)
    map_name, map_path, zone_centers_filename, is_explo, fill_buffer, save_aggregated_stats = next_map_tuple

    # ===============================================
    #   VERY BASIC TRAINING ANNEALING
    # ===============================================

    if accumulated_stats["cumul_number_memories_generated"] > 300_000:
        misc.reward_per_ms_press_forward_early_training = 0

    # LR and weight_decay calculation
    learning_rate = nn_utilities.lr_from_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"])
    epsilon = nn_utilities.lr_from_schedule(misc.epsilon_schedule, accumulated_stats["cumul_number_memories_generated"])
    weight_decay = misc.weight_decay_lr_ratio * learning_rate

    # ===============================================
    #   RELOAD
    # ===============================================

    for param_group in soft_Q_optimizer.param_groups:
        param_group["lr"] = learning_rate
    for param_group in policy_optimizer.param_groups:
        param_group["lr"] = learning_rate * misc.lr_policy_ratio
    for param_group in logalpha_optimizer.param_groups:
        param_group["lr"] = learning_rate * misc.lr_alpha_ratio
    trainer.gamma = misc.gamma
    trainer.target_entropy = misc.target_entropy
    trainer.truncation_amplitude = misc.truncation_amplitude

    if isinstance(buffer._sampler, PrioritizedSampler):
        buffer._sampler._alpha = misc.prio_alpha
        buffer._sampler._beta = misc.prio_beta
        buffer._sampler._eps = misc.prio_epsilon

    if is_explo:
        trainer.epsilon = epsilon
    else:
        trainer.epsilon = -1
        print("EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL")

    # ===============================================
    #   PLAY ONE ROUND
    # ===============================================

    rollout_start_time = time.time()
    rollout_results, end_race_stats = tmi.rollout(
        exploration_policy=trainer.get_exploration_action,
        map_path=map_path,
        zone_centers=zone_centers,
    )

    if len(rollout_results["log_policy"]) > 0:
        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # ===============================================
        #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
        # ===============================================
        race_stats_to_write = {
            f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / ((time.time() - rollout_start_time) * 1000),
            f"explo_race_time_{map_name}" if is_explo else f"eval_race_time_{map_name}": end_race_stats["race_time"] / 1000,
            f"explo_race_finished_{map_name}" if is_explo else f"eval_race_finished_{map_name}": end_race_stats["race_finished"],
            # f"mean_action_gap_{map_name}": -(
            #     np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
            # ).mean(),
            f"single_zone_reached_{map_name}": len(rollout_results["zone_entrance_time_ms"]) - 1,
            "time_to_answer_normal_step": end_race_stats["time_to_answer_normal_step"],
            "time_to_answer_action_step": end_race_stats["time_to_answer_action_step"],
            "time_between_normal_on_run_steps": end_race_stats["time_between_normal_on_run_steps"],
            "time_between_action_on_run_steps": end_race_stats["time_between_action_on_run_steps"],
            "time_to_grab_frame": end_race_stats["time_to_grab_frame"],
            "time_between_grab_frame": end_race_stats["time_between_grab_frame"],
            "time_A_rgb2gray": end_race_stats["time_A_rgb2gray"],
            "time_A_geometry": end_race_stats["time_A_geometry"],
            "time_A_stack": end_race_stats["time_A_stack"],
            "time_exploration_policy": end_race_stats["time_exploration_policy"],
            "time_to_iface_set_set": end_race_stats["time_to_iface_set_set"],
            "time_after_iface_set_set": end_race_stats["time_after_iface_set_set"],
            "tmi_protection_cutoff": end_race_stats["tmi_protection_cutoff"],
        }
        print("Race time ratio  ", race_stats_to_write[f"race_time_ratio_{map_name}"])

        if end_race_stats["race_finished"]:
            race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_name}"] = end_race_stats["race_time"] / 1000
        for i in range(len(misc.inputs)):
            race_stats_to_write[f"policy_{i}_starting_frame_{map_name}"] = end_race_stats[f"policy_{i}_starting_frame"]

        walltime_tb = float(accumulated_stats["cumul_training_hours"] * 3600) + time.time() - time_last_save
        for tag, value in race_stats_to_write.items():
            tensorboard_writer.add_scalar(
                tag=tag,
                scalar_value=value,
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

    # ===============================================
    #   SAVE STUFF IF THIS WAS A GOOD RACE
    # ===============================================

    if end_race_stats["race_time"] < accumulated_stats["alltime_min_ms"].get(map_name, 99999999999):
        # This is a new alltime_minimum
        accumulated_stats["alltime_min_ms"][map_name] = end_race_stats["race_time"]
        print("\a")

        sub_folder_name = f"{map_name}_{end_race_stats['race_time']}"
        (save_dir / "best_runs" / sub_folder_name).mkdir(parents=True, exist_ok=True)
        joblib.dump(
            rollout_results["actions"],
            save_dir / "best_runs" / sub_folder_name / f"actions.joblib",
        )
        joblib.dump(
            rollout_results["log_policy"],
            save_dir / "best_runs" / sub_folder_name / f"policy.joblib",
        )
        torch.save(
            soft_Q_model1.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "soft_Q_weights1.torch",
        )
        torch.save(
            soft_Q_model2.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "soft_Q_weights2.torch",
        )
        torch.save(
            soft_Q_optimizer.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "soft_Q_optimizer.torch",
        )
        torch.save(
            soft_Q_scaler.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "soft_Q_scaler.torch",
        )

        torch.save(
            policy_model.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "policy_weights.torch",
        )
        torch.save(
            policy_optimizer.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "policy_optimizer.torch",
        )
        torch.save(
            policy_scaler.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "policy_scaler.torch",
        )

        torch.save(
            logalpha_model.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "logalpha_weights.torch",
        )
        torch.save(
            logalpha_optimizer.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "logalpha_optimizer.torch",
        )
        torch.save(
            logalpha_scaler.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "logalpha_scaler.torch",
        )

        shutil.copy(base_dir / "trackmania_rl" / "misc.py", save_dir / "best_runs" / sub_folder_name / "misc.py.save")

    # ===============================================
    #   FILL BUFFER WITH (S, A, R, S') transitions
    # ===============================================

    if fill_buffer:
        (
            buffer,
            buffer_test,
            number_memories_added,
        ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
            buffer,
            buffer_test,
            rollout_results,
            misc.n_steps,
            misc.gamma,
            misc.discard_non_greedy_actions_in_nsteps,
            misc.n_zone_centers_in_inputs,
            zone_centers,
        )

        accumulated_stats["cumul_number_memories_generated"] += number_memories_added
        accumulated_stats["reset_counter"] += number_memories_added
        accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
            misc.number_times_single_memory_is_used_before_discard * number_memories_added
        )
        print(f" NMG={accumulated_stats['cumul_number_memories_generated']:<8}")

        # ===============================================
        #   PERIODIC RESET ?
        # ===============================================

        # if (
        #     accumulated_stats["reset_counter"] >= misc.reset_every_n_frames_generated
        #     or accumulated_stats["single_reset_counter"] != misc.single_reset_counter
        # ):
        #     accumulated_stats["reset_counter"] = 0
        #     accumulated_stats["single_reset_counter"] = misc.single_reset_counter
        #     accumulated_stats["cumul_number_single_memories_should_have_been_used"] += misc.additional_transition_after_reset
        #
        #     model3 = make_untrained_agent().to("cuda", memory_format=torch.channels_last)
        #     nn_utilities.soft_copy_param(soft_Q_model1, model3, misc.overall_reset_mul_factor)
        #
        #     with torch.no_grad():
        #         # for name, param in model1.named_parameters():
        #         #     param *= misc.overall_reset_mul_factor
        #         soft_Q_model1.A_head[0].weight *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.A_head[0].weight += (1 - misc.a_v_reset_mul_factor) * model3.A_head[0].weight
        #
        #         soft_Q_model1.A_head[0].bias *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.A_head[0].bias += (1 - misc.a_v_reset_mul_factor) * model3.A_head[0].bias
        #
        #         soft_Q_model1.A_head[2].weight *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.A_head[2].weight += (1 - misc.a_v_reset_mul_factor) * model3.A_head[2].weight
        #
        #         soft_Q_model1.A_head[2].bias *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.A_head[2].bias += (1 - misc.a_v_reset_mul_factor) * model3.A_head[2].bias
        #
        #         soft_Q_model1.V_head[0].weight *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.V_head[0].weight += (1 - misc.a_v_reset_mul_factor) * model3.V_head[0].weight
        #
        #         soft_Q_model1.V_head[0].bias *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.V_head[0].bias += (1 - misc.a_v_reset_mul_factor) * model3.V_head[0].bias
        #
        #         soft_Q_model1.V_head[2].weight *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.V_head[2].weight += (1 - misc.a_v_reset_mul_factor) * model3.V_head[2].weight
        #
        #         soft_Q_model1.V_head[2].bias *= misc.a_v_reset_mul_factor
        #         soft_Q_model1.V_head[2].bias += (1 - misc.a_v_reset_mul_factor) * model3.V_head[2].bias

        # ===============================================
        #   LEARN ON BATCH
        # ===============================================

        while (
            len(buffer) >= misc.memory_size_start_learn
            and accumulated_stats["cumul_number_single_memories_used"] + misc.offset_cumul_number_single_memories_used
            <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
        ):
            if (random.random() < misc.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                loss_Q, loss_policy, loss_alpha, policy_entropy = trainer.train_on_batch(buffer_test, do_learn=False)
                loss_Q_test_history.append(loss_Q)
                loss_policy_test_history.append(loss_policy)
                loss_alpha_test_history.append(loss_alpha)
                policy_entropy_test_history.append(policy_entropy)
                print(f"BT   {loss_Q=:<8.2e} {loss_policy=:<8.2e} {loss_alpha=:<8.2e}")
            else:
                train_start_time = time.time()
                loss_Q, loss_policy, loss_alpha, policy_entropy = trainer.train_on_batch(buffer, do_learn=True)
                accumulated_stats["cumul_number_single_memories_used"] += misc.batch_size
                train_on_batch_duration_history.append(time.time() - train_start_time)
                loss_Q_history.append(loss_Q)
                loss_policy_history.append(loss_policy)
                loss_alpha_history.append(loss_alpha)
                policy_entropy_history.append(policy_entropy)
                # if not math.isinf(grad_norm):
                #     grad_norm_history.append(grad_norm)
                #     for name, param in soft_Q_model1.named_parameters():
                #         layer_grad_norm_history[f"L2_grad_norm_{name}"].append(torch.norm(param.grad.detach(), 2.0).item())
                #         layer_grad_norm_history[f"Linf_grad_norm_{name}"].append(torch.norm(param.grad.detach(), float("inf")).item())

                accumulated_stats["cumul_number_batches_done"] += 1
                print(f"B    {loss_Q=:<8.2e} {loss_policy=:<8.2e} {loss_alpha=:<8.2e}")

                nn_utilities.custom_weight_decay(soft_Q_model1, 1 - weight_decay)
                nn_utilities.custom_weight_decay(policy_model, 1 - weight_decay)

                # ===============================================
                #   UPDATE TARGET NETWORK
                # ===============================================
                if (
                    accumulated_stats["cumul_number_single_memories_used"]
                    >= accumulated_stats["cumul_number_single_memories_used_next_target_network_update"]
                ):
                    accumulated_stats["cumul_number_target_network_updates"] += 1
                    accumulated_stats[
                        "cumul_number_single_memories_used_next_target_network_update"
                    ] += misc.number_memories_trained_on_between_target_network_updates
                    print("UPDATE")
                    nn_utilities.soft_copy_param(soft_Q_model2, soft_Q_model1, misc.soft_update_tau)
        print("")

    # ===============================================
    #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY NOW AND THEN
    # ===============================================
    if save_aggregated_stats:
        accumulated_stats["cumul_training_hours"] += (time.time() - time_last_save) / 3600
        time_last_save = time.time()

        # ===============================================
        #   COLLECT VARIOUS STATISTICS
        # ===============================================
        step_stats = {
            "gamma": misc.gamma,
            "n_steps": misc.n_steps,
            "epsilon": epsilon,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "discard_non_greedy_actions_in_nsteps": misc.discard_non_greedy_actions_in_nsteps,
            "reward_per_ms_press_forward": misc.reward_per_ms_press_forward_early_training,
            "memory_size": len(buffer),
            "number_times_single_memory_is_used_before_discard": misc.number_times_single_memory_is_used_before_discard,
            "sac_alpha": logalpha_model().exp().item(),
        }
        if len(loss_Q_history) > 0 and len(loss_Q_test_history) > 0:
            step_stats.update(
                {
                    "loss_Q": np.mean(loss_Q_history),
                    "loss_Q_test": np.mean(loss_Q_test_history),
                    "loss_policy": np.mean(loss_policy_history),
                    "loss_policy_test": np.mean(loss_policy_test_history),
                    "loss_alpha": np.mean(loss_alpha_history),
                    "loss_alpha_test": np.mean(loss_alpha_test_history),
                    "policy_entropy": np.mean(policy_entropy_history),
                    "policy_entropy_test": np.mean(policy_entropy_test_history),
                    "train_on_batch_duration": np.median(train_on_batch_duration_history),
                    # "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25),
                    # "grad_norm_history_median": np.quantile(grad_norm_history, 0.5),
                    # "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75),
                    # "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9),
                    # "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98),
                    # "grad_norm_history_max": np.max(grad_norm_history),
                }
            )
            # for key, val in layer_grad_norm_history.items():
            #     step_stats.update(
            #         {
            #             f"{key}_median": np.quantile(val, 0.5),
            #             f"{key}_q3": np.quantile(val, 0.75),
            #             f"{key}_d9": np.quantile(val, 0.9),
            #             f"{key}_d98": np.quantile(val, 0.98),
            #             f"{key}_max": np.max(val),
            #         }
            #     )

        for key, value in accumulated_stats.items():
            if key != "alltime_min_ms":
                step_stats[key] = value
        for key, value in accumulated_stats["alltime_min_ms"].items():
            step_stats[f"alltime_min_ms_{map_name}"] = value

        loss_Q_history = []
        loss_Q_test_history = []
        loss_policy_history = []
        loss_policy_test_history = []
        loss_alpha_history = []
        loss_alpha_test_history = []
        policy_entropy_history = []
        policy_entropy_test_history = []
        train_on_batch_duration_history = []
        grad_norm_history = []
        layer_grad_norm_history = defaultdict(list)

        # ===============================================
        #   COLLECT IQN SPREAD
        # ===============================================

        tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")
        state_img_tensor = torch.as_tensor(
            np.expand_dims(rollout_results["frames"][0], axis=0)
        ).to(  # TODO : remove as_tensor and expand dims, because this is already pinned memory
            "cuda", memory_format=torch.channels_last, non_blocking=True
        )
        state_float_tensor = torch.as_tensor(
            np.expand_dims(
                np.hstack(
                    (
                        0,
                        np.hstack([np.array([True, False, False, False]) for _ in range(misc.n_prev_actions_in_inputs)]),
                        # NEW
                        rollout_results["car_gear_and_wheels"][0].ravel(),  # NEW
                        rollout_results["car_orientation"][0].T.dot(rollout_results["car_angular_speed"][0]),  # NEW
                        rollout_results["car_orientation"][0].T.dot(rollout_results["car_velocity"][0]),
                        rollout_results["car_orientation"][0].T.dot(np.array([0, 1, 0])),
                        rollout_results["car_orientation"][0]
                        .T.dot((zone_centers[0 : misc.n_zone_centers_in_inputs, :] - rollout_results["car_position"][0]).T)
                        .T.ravel(),
                    )
                ).astype(np.float32),
                axis=0,
            )
        ).to("cuda", non_blocking=True)

        # Désactiver noisy, tirer des tau équitablement répartis
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                per_quantile_output = soft_Q_model1(state_img_tensor, state_float_tensor, misc.iqn_k, tau=tau)[0]

        for i, mean in enumerate(list(per_quantile_output.cpu().numpy().astype(np.float32).mean(axis=0))):
            step_stats[f"q_value_starting_frame_{i}"] = mean

        for i, std in enumerate(list(per_quantile_output.cpu().numpy().astype(np.float32).std(axis=0))):
            step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std
        soft_Q_model1.train()

        # ===============================================
        #   WRITE TO TENSORBOARD
        # ===============================================

        walltime_tb = float(accumulated_stats["cumul_training_hours"] * 3600) + time.time() - time_last_save
        for name, param in soft_Q_model1.named_parameters():
            tensorboard_writer.add_scalar(
                tag=f"layer_{name}_L2",
                scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )
        assert len(soft_Q_optimizer.param_groups) == 1
        try:
            for p, (name, _) in zip(soft_Q_optimizer.param_groups[0]["params"], soft_Q_model1.named_parameters()):
                state = soft_Q_optimizer.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                mod_lr = 1 / (exp_avg_sq.sqrt() + 1e-4)
                tensorboard_writer.add_scalar(
                    tag=f"lr_ratio_{name}_L2",
                    scalar_value=np.sqrt((mod_lr**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
                tensorboard_writer.add_scalar(
                    tag=f"exp_avg_{name}_L2",
                    scalar_value=np.sqrt((exp_avg**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
                tensorboard_writer.add_scalar(
                    tag=f"exp_avg_sq_{name}_L2",
                    scalar_value=np.sqrt((exp_avg_sq**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
        except:
            pass

        for k, v in step_stats.items():
            tensorboard_writer.add_scalar(
                tag=k,
                scalar_value=v,
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

        tensorboard_writer.add_text(
            "times_summary",
            f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
            + " ".join([f"{k}: {v / 1000:.2f}" for k, v in accumulated_stats["alltime_min_ms"].items()]),
            global_step=accumulated_stats["cumul_number_frames_played"],
            walltime=walltime_tb,
        )

        # ===============================================
        #   BUFFER STATS
        # ===============================================

        mean_in_buffer = np.array([experience.state_float for experience in buffer._storage]).mean(axis=0)
        std_in_buffer = np.array([experience.state_float for experience in buffer._storage]).std(axis=0)

        print("Raw mean in buffer  :", mean_in_buffer.round(1))
        print("Raw std in buffer   :", std_in_buffer.round(1))
        print("")
        print(
            "Corr mean in buffer :",
            ((mean_in_buffer - misc.float_inputs_mean) / misc.float_inputs_std).round(1),
        )
        print("Corr std in buffer  :", (std_in_buffer / misc.float_inputs_std).round(1))
        print("")

        # ===============================================
        #   SAVE
        # ===============================================

        torch.save(soft_Q_model1.state_dict(), save_dir / "soft_Q_weights1.torch")
        torch.save(soft_Q_model2.state_dict(), save_dir / "soft_Q_weights2.torch")
        torch.save(soft_Q_optimizer.state_dict(), save_dir / "soft_Q_optimizer.torch")
        torch.save(soft_Q_scaler.state_dict(), save_dir / "soft_Q_scaler.torch")
        torch.save(policy_model.state_dict(), save_dir / "policy_weights.torch")
        torch.save(policy_optimizer.state_dict(), save_dir / "policy_optimizer.torch")
        torch.save(policy_scaler.state_dict(), save_dir / "policy_scaler.torch")
        torch.save(logalpha_model.state_dict(), save_dir / "logalpha_weights.torch")
        torch.save(logalpha_optimizer.state_dict(), save_dir / "logalpha_optimizer.torch")
        torch.save(logalpha_scaler.state_dict(), save_dir / "logalpha_scaler.torch")

        joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
