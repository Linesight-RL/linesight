import importlib
import math
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

import trackmania_rl.agents.iqn as iqn
from trackmania_rl import buffer_management, misc, nn_utilities, tm_interface_manager
from trackmania_rl.buffer_utilities import buffer_collate_function
from trackmania_rl.experience_replay.basic_experience_replay import ReplayBuffer
from trackmania_rl.map_loader import load_next_map_zone_centers

base_dir = Path(__file__).resolve().parents[1]

save_dir = base_dir / "save" / misc.run_name
save_dir.mkdir(parents=True, exist_ok=True)
tensorboard_writer = SummaryWriter(log_dir=str(base_dir / "tensorboard" / misc.run_name))

layout = {
    "89": {
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
        "loss": ["Multiline", ["loss$", "loss_test$"]],
        "values_starting_frame": [
            "Multiline",
            [f"q_value_{i}_starting_frame" for i in range(len(misc.inputs))],
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


def make_untrained_agent():
    return iqn.Agent(
        float_inputs_dim=misc.float_input_dim,
        float_hidden_dim=misc.float_hidden_dim,
        conv_head_output_dim=misc.conv_head_output_dim,
        dense_hidden_dimension=misc.dense_hidden_dimension,
        iqn_embedding_dimension=misc.iqn_embedding_dimension,
        n_actions=len(misc.inputs),
        float_inputs_mean=misc.float_inputs_mean,
        float_inputs_std=misc.float_inputs_std,
    )


model1 = torch.jit.script(make_untrained_agent()).to("cuda", memory_format=torch.channels_last)
model2 = torch.jit.script(make_untrained_agent()).to("cuda", memory_format=torch.channels_last)

print(model1)

accumulated_stats: defaultdict[str | typing.Any] = defaultdict(int)
accumulated_stats["alltime_min_ms"] = {}

# ========================================================
# Load existing stuff
# ========================================================
# noinspection PyBroadException
try:
    model1.load_state_dict(torch.load(save_dir / "weights1.torch"))
    model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
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

optimizer1 = torch.optim.RAdam(
    model1.parameters(),
    lr=nn_utilities.lr_from_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
    eps=misc.adam_epsilon,
    betas=(0.9, 0.95),
)
# optimizer1 = torch.optim.AdamW(
#     model1.parameters(),
#     lr=nn_utilities.lr_from_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
#     eps=misc.adam_epsilon,
#     betas=(0.9, 0.95),
#     weight_decay=0.1,
# )
# optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate, eps=0.01)
# optimizer1 = torch.optim.SGD(model1.parameters(), lr=nn_utilities.lr_from_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"]), momentum=0.8)
# optimizer1 = torch_optimizer.Lamb(
#     model1.parameters(),
#     lr= 5e-5,
#     betas=(0.9, 0.999),
#     eps=1e-4,
#     weight_decay=0,
# )
optimizer1 = torch_optimizer.Lookahead(optimizer1, k=5, alpha=0.5)

scaler = torch.cuda.amp.GradScaler()
buffer = ReplayBuffer(capacity=misc.memory_size, batch_size=misc.batch_size, collate_fn=buffer_collate_function, prefetch=1)
buffer_test = ReplayBuffer(
    capacity=int(misc.memory_size * misc.buffer_test_ratio), batch_size=misc.batch_size, collate_fn=buffer_collate_function
)

# noinspection PyBroadException
try:
    optimizer1.load_state_dict(torch.load(save_dir / "optimizer1.torch"))
    print(" =========================     Optimizer loaded !     ================================")
except:
    print(" Could not load optimizer")

loss_history = []
loss_test_history = []
train_on_batch_duration_history = []
grad_norm_history = []
layer_grad_norm_history = defaultdict(list)

# ========================================================
# Make the trainer
# ========================================================
trainer = iqn.Trainer(
    model=model1,
    model2=model2,
    optimizer=optimizer1,
    scaler=scaler,
    batch_size=misc.batch_size,
    iqn_k=misc.iqn_k,
    iqn_n=misc.iqn_n,
    iqn_kappa=misc.iqn_kappa,
    epsilon=misc.epsilon,
    epsilon_boltzmann=misc.epsilon_boltzmann,
    gamma=misc.gamma,
    AL_alpha=misc.AL_alpha,
    tau_epsilon_boltzmann=misc.tau_epsilon_boltzmann,
    tau_greedy_boltzmann=misc.tau_greedy_boltzmann,
)

# ========================================================
# Training loop
# ========================================================
model1.train()
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
    weight_decay = misc.weight_decay_lr_ratio * learning_rate

    # ===============================================
    #   RELOAD
    # ===============================================

    for param_group in optimizer1.param_groups:
        param_group["lr"] = learning_rate
    trainer.gamma = misc.gamma
    trainer.AL_alpha = misc.AL_alpha
    trainer.tau_epsilon_boltzmann = misc.tau_epsilon_boltzmann
    trainer.tau_greedy_boltzmann = misc.tau_greedy_boltzmann

    if is_explo:
        trainer.epsilon = (
            misc.high_exploration_ratio * misc.epsilon
            if accumulated_stats["cumul_number_memories_generated"] < misc.number_memories_generated_high_exploration_early_training
            else misc.epsilon
        )
        trainer.epsilon_boltzmann = (
            misc.high_exploration_ratio * misc.epsilon_boltzmann
            if accumulated_stats["cumul_number_memories_generated"] < misc.number_memories_generated_high_exploration_early_training
            else misc.epsilon_boltzmann
        )
    else:
        trainer.epsilon = 0
        trainer.epsilon_boltzmann = 0
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

    accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

    # ===============================================
    #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
    # ===============================================
    race_stats_to_write = {
        f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / ((time.time() - rollout_start_time) * 1000),
        f"explo_race_time_{map_name}" if is_explo else f"eval_race_time_{map_name}": end_race_stats["race_time"] / 1000,
        f"explo_race_finished_{map_name}" if is_explo else f"eval_race_finished_{map_name}": end_race_stats["race_finished"],
        f"mean_action_gap_{map_name}": -(
            np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
        ).mean(),
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
        race_stats_to_write[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]

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
            rollout_results["q_values"],
            save_dir / "best_runs" / sub_folder_name / f"q_values.joblib",
        )
        torch.save(
            model1.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "weights1.torch",
        )
        torch.save(
            model2.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "weights2.torch",
        )
        torch.save(
            optimizer1.state_dict(),
            save_dir / "best_runs" / sub_folder_name / "optimizer1.torch",
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

        if accumulated_stats["reset_counter"] >= misc.reset_every_n_frames_generated:
            accumulated_stats["reset_counter"] = 0
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += misc.additional_transition_after_reset

            model3 = make_untrained_agent().to("cuda", memory_format=torch.channels_last)
            nn_utilities.soft_copy_param(model1, model3, misc.overall_reset_mul_factor)

            with torch.no_grad():
                # for name, param in model1.named_parameters():
                #     param *= misc.overall_reset_mul_factor
                model1.A_head[0].weight *= misc.a_v_reset_mul_factor
                model1.A_head[0].weight += (1 - misc.a_v_reset_mul_factor) * model3.A_head[0].weight

                model1.A_head[0].bias *= misc.a_v_reset_mul_factor
                model1.A_head[0].bias += (1 - misc.a_v_reset_mul_factor) * model3.A_head[0].bias

                model1.A_head[2].weight *= misc.a_v_reset_mul_factor
                model1.A_head[2].weight += (1 - misc.a_v_reset_mul_factor) * model3.A_head[2].weight

                model1.A_head[2].bias *= misc.a_v_reset_mul_factor
                model1.A_head[2].bias += (1 - misc.a_v_reset_mul_factor) * model3.A_head[2].bias

                model1.V_head[0].weight *= misc.a_v_reset_mul_factor
                model1.V_head[0].weight += (1 - misc.a_v_reset_mul_factor) * model3.V_head[0].weight

                model1.V_head[0].bias *= misc.a_v_reset_mul_factor
                model1.V_head[0].bias += (1 - misc.a_v_reset_mul_factor) * model3.V_head[0].bias

                model1.V_head[2].weight *= misc.a_v_reset_mul_factor
                model1.V_head[2].weight += (1 - misc.a_v_reset_mul_factor) * model3.V_head[2].weight

                model1.V_head[2].bias *= misc.a_v_reset_mul_factor
                model1.V_head[2].bias += (1 - misc.a_v_reset_mul_factor) * model3.V_head[2].bias

        # ===============================================
        #   LEARN ON BATCH
        # ===============================================

        while (
            len(buffer) >= misc.memory_size_start_learn
            and accumulated_stats["cumul_number_single_memories_used"] + misc.offset_cumul_number_single_memories_used
            <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
        ):
            if (random.random() < misc.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                loss, _ = trainer.train_on_batch(buffer_test, do_learn=False)
                loss_test_history.append(loss)
                print(f"BT   {loss=:<8.2e}")
            else:
                train_start_time = time.time()
                loss, grad_norm = trainer.train_on_batch(buffer, do_learn=True)
                accumulated_stats["cumul_number_single_memories_used"] += misc.batch_size
                train_on_batch_duration_history.append(time.time() - train_start_time)
                loss_history.append(loss)
                if not math.isinf(grad_norm):
                    grad_norm_history.append(grad_norm)
                    for name, param in model1.named_parameters():
                        layer_grad_norm_history[f"L2_grad_norm_{name}"].append(torch.norm(param.grad.detach(), 2.0).item())
                        layer_grad_norm_history[f"Linf_grad_norm_{name}"].append(torch.norm(param.grad.detach(), float("inf")).item())

                accumulated_stats["cumul_number_batches_done"] += 1
                print(f"B    {loss=:<8.2e} {grad_norm=:<8.2e}")

                nn_utilities.custom_weight_decay(model1, 1 - weight_decay)

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
                    nn_utilities.soft_copy_param(model2, model1, misc.soft_update_tau)
                    # model2.load_state_dict(model.state_dict())
        buffer.sync_prefetching()  # Finish all prefetching to avoid invalid prefetches during rollouts where the pinned image buffer will be overwritten
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
            "epsilon": misc.epsilon,
            "epsilon_boltzmann": misc.epsilon_boltzmann,
            "tau_epsilon_boltzmann": misc.tau_epsilon_boltzmann,
            "tau_greedy_boltzmann": misc.tau_greedy_boltzmann,
            "AL_alpha": misc.AL_alpha,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "discard_non_greedy_actions_in_nsteps": misc.discard_non_greedy_actions_in_nsteps,
            "reward_per_ms_press_forward": misc.reward_per_ms_press_forward_early_training,
            "memory_size": len(buffer),
            "number_times_single_memory_is_used_before_discard": misc.number_times_single_memory_is_used_before_discard,
        }
        if len(loss_history) > 0 and len(loss_test_history) > 0:
            step_stats.update(
                {
                    "loss": np.mean(loss_history),
                    "loss_test": np.mean(loss_test_history),
                    "train_on_batch_duration": np.median(train_on_batch_duration_history),
                    "grad_norm_history_q1": np.quantile(grad_norm_history, 0.25),
                    "grad_norm_history_median": np.quantile(grad_norm_history, 0.5),
                    "grad_norm_history_q3": np.quantile(grad_norm_history, 0.75),
                    "grad_norm_history_d9": np.quantile(grad_norm_history, 0.9),
                    "grad_norm_history_d98": np.quantile(grad_norm_history, 0.98),
                    "grad_norm_history_max": np.max(grad_norm_history),
                }
            )
            for key, val in layer_grad_norm_history.items():
                step_stats.update(
                    {
                        f"{key}_median": np.quantile(val, 0.5),
                        f"{key}_q3": np.quantile(val, 0.75),
                        f"{key}_d9": np.quantile(val, 0.9),
                        f"{key}_d98": np.quantile(val, 0.98),
                        f"{key}_max": np.max(val),
                    }
                )

        for key, value in accumulated_stats.items():
            if key != "alltime_min_ms":
                step_stats[key] = value
        for key, value in accumulated_stats["alltime_min_ms"].items():
            step_stats[f"alltime_min_ms_{map_name}"] = value

        loss_history = []
        loss_test_history = []
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
                        np.hstack([np.array([True, False, False, False]) for _ in range(misc.n_prev_actions_in_inputs)]),  # NEW
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
                per_quantile_output = model1(state_img_tensor, state_float_tensor, misc.iqn_k, tau=tau)[0]

        for i, std in enumerate(list(per_quantile_output.cpu().numpy().astype(np.float32).std(axis=0))):
            step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std
        model1.train()

        # ===============================================
        #   WRITE TO TENSORBOARD
        # ===============================================

        walltime_tb = float(accumulated_stats["cumul_training_hours"] * 3600) + time.time() - time_last_save
        for name, param in model1.named_parameters():
            tensorboard_writer.add_scalar(
                tag=f"layer_{name}_L2",
                scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )
        assert len(optimizer1.param_groups) == 1
        try:
            for p, (name, _) in zip(optimizer1.param_groups[0]["params"], model1.named_parameters()):
                state = optimizer1.state[p]
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

        torch.save(model1.state_dict(), save_dir / "weights1.torch")
        torch.save(model2.state_dict(), save_dir / "weights2.torch")
        torch.save(optimizer1.state_dict(), save_dir / "optimizer1.torch")
        joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
