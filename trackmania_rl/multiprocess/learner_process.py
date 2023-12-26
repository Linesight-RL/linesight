import copy
import importlib
import math
import random
import shutil
import time
import typing
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import PrioritizedSampler

from trackmania_rl import buffer_management, misc, run_to_video, utilities
from trackmania_rl.agents import iqn as iqn
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl.buffer_utilities import make_buffers, resize_buffers
from trackmania_rl.map_reference_times import reference_times
from trackmania_rl.analysis_metrics import (
    distribution_curves,
    highest_prio_transitions,
    race_time_left_curves,
    tau_curves,
    loss_distribution,
)


def learner_process_fn(
    rollout_queues,
    uncompiled_shared_network,
    shared_network_lock,
    shared_steps: mp.Value,
    base_dir: Path,
    save_dir: Path,
    tensorboard_base_dir: Path,
):
    layout_version = "lay_mono"
    SummaryWriter(log_dir=str(tensorboard_base_dir / layout_version)).add_custom_scalars(
        {
            layout_version: {
                # "eval_agg_ratio": [
                #     "Multiline",
                #     [
                #         "eval_agg_ratio_trained_author",
                #         "eval_agg_ratio_blind_author",
                #     ],
                # ],
                # "eval_ratio_trained_author": [
                #     "Multiline",
                #     [
                #         "eval_ratio_trained_author",
                #     ],
                # ],
                # "eval_ratio_blind_author": [
                #     "Multiline",
                #     [
                #         "eval_ratio_blind_author",
                #     ],
                # ],
                "eval_race_time_robust": [
                    "Multiline",
                    [
                        "eval_race_time_robust",
                    ],
                ],
                "explo_race_time_finished": [
                    "Multiline",
                    [
                        "explo_race_time_finished",
                    ],
                ],
                "loss": ["Multiline", ["loss$", "loss_test$"]],
                "avg_Q": [
                    "Multiline",
                    ["avg_Q"],
                ],
                "single_zone_reached": [
                    "Multiline",
                    [
                        "single_zone_reached",
                    ],
                ],
                "grad_norm_history": [
                    "Multiline",
                    [
                        "grad_norm_history_d9",
                        "grad_norm_history_d98",
                    ],
                ],
                "priorities": [
                    "Multiline",
                    [
                        "priorities",
                    ],
                ],
            },
        }
    )

    # ========================================================
    # Create new stuff
    # ========================================================

    online_network, uncompiled_online_network = make_untrained_iqn_network(misc.use_jit)
    target_network, _ = make_untrained_iqn_network(misc.use_jit)

    print(online_network)
    utilities.count_parameters(online_network)

    accumulated_stats: defaultdict[str | typing.Any] = defaultdict(int)
    accumulated_stats["alltime_min_ms"] = {}
    accumulated_stats["rolling_mean_ms"] = {}
    previous_alltime_min = None
    time_last_save = time.time()

    # ========================================================
    # Load existing stuff
    # ========================================================
    # noinspection PyBroadException
    try:
        online_network.load_state_dict(torch.load(save_dir / "weights1.torch"))
        target_network.load_state_dict(torch.load(save_dir / "weights2.torch"))
        print(" =====================     Learner weights loaded !     ============================")
    except:
        print(" Learner could not load weights")

    with shared_network_lock:
        uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

    # noinspection PyBroadException
    try:
        accumulated_stats = joblib.load(save_dir / "accumulated_stats.joblib")
        shared_steps.value = accumulated_stats["cumul_number_memories_generated"]
        print(" =====================      Learner stats loaded !      ============================")
    except:
        print(" Learner could not load stats")

    if "rolling_mean_ms" not in accumulated_stats.keys():
        # Temporary to preserve compatibility with old runs that doesn't have this feature. To be removed later.
        accumulated_stats["rolling_mean_ms"] = {}

    accumulated_stats["cumul_number_single_memories_should_have_been_used"] = accumulated_stats["cumul_number_single_memories_used"]
    neural_net_reset_counter = 0
    single_reset_flag = misc.single_reset_flag

    optimizer1 = torch.optim.RAdam(
        online_network.parameters(),
        lr=utilities.from_exponential_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"]),
        eps=misc.adam_epsilon,
        betas=(misc.adam_beta1, misc.adam_beta2),
    )
    # optimizer1 = torch_optimizer.Lookahead(optimizer1, k=5, alpha=0.5)

    scaler = torch.cuda.amp.GradScaler()
    memory_size, memory_size_start_learn = utilities.from_staircase_schedule(
        misc.memory_size_schedule, accumulated_stats["cumul_number_memories_generated"]
    )
    buffer, buffer_test = make_buffers(memory_size)
    offset_cumul_number_single_memories_used = memory_size_start_learn * misc.number_times_single_memory_is_used_before_discard

    # noinspection PyBroadException
    try:
        optimizer1.load_state_dict(torch.load(save_dir / "optimizer1.torch"))
        scaler.load_state_dict(torch.load(save_dir / "scaler.torch"))
        print(" =========================     Optimizer loaded !     ================================")
    except:
        print(" Could not load optimizer")

    tensorboard_suffix = utilities.from_staircase_schedule(
        misc.tensorboard_suffix_schedule, accumulated_stats["cumul_number_memories_generated"]
    )
    tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (misc.run_name + tensorboard_suffix)))

    loss_history = []
    loss_test_history = []
    train_on_batch_duration_history = []
    grad_norm_history = []
    layer_grad_norm_history = defaultdict(list)

    # ========================================================
    # Make the trainer
    # ========================================================
    trainer = iqn.Trainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer1,
        scaler=scaler,
        batch_size=misc.batch_size,
        iqn_n=misc.iqn_n,
        gamma=misc.gamma,
    )

    inferer = iqn.Inferer(inference_network=online_network, iqn_k=misc.iqn_k, tau_epsilon_boltzmann=misc.tau_epsilon_boltzmann)

    while True:  # Trainer loop
        i_start = random.randrange(len(rollout_queues))
        for i in range(i_start, i_start + len(rollout_queues)):
            if not rollout_queues[i % len(rollout_queues)].empty():
                (
                    rollout_results,
                    end_race_stats,
                    fill_buffer,
                    is_explo,
                    map_name,
                    map_status,
                    rollout_duration,
                    loop_number,
                ) = rollout_queues[i % len(rollout_queues)].get()
                break
        else:
            print("All rollout queues were empty. Learner sleeps 1 second.")
            time.sleep(1)
            continue

        importlib.reload(misc)

        new_tensorboard_suffix = utilities.from_staircase_schedule(
            misc.tensorboard_suffix_schedule, accumulated_stats["cumul_number_memories_generated"]
        )
        if new_tensorboard_suffix != tensorboard_suffix:
            tensorboard_suffix = new_tensorboard_suffix
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_base_dir / (misc.run_name + tensorboard_suffix)))

        new_memory_size, new_memory_size_start_learn = utilities.from_staircase_schedule(
            misc.memory_size_schedule, accumulated_stats["cumul_number_memories_generated"]
        )
        if new_memory_size != memory_size:
            buffer, buffer_test = resize_buffers(buffer, buffer_test, new_memory_size)
            offset_cumul_number_single_memories_used += (
                new_memory_size_start_learn - memory_size_start_learn
            ) * misc.number_times_single_memory_is_used_before_discard
            memory_size_start_learn = new_memory_size_start_learn
            memory_size = new_memory_size
        # ===============================================
        #   VERY BASIC TRAINING ANNEALING
        # ===============================================

        # LR and weight_decay calculation
        learning_rate = utilities.from_exponential_schedule(misc.lr_schedule, accumulated_stats["cumul_number_memories_generated"])
        weight_decay = misc.weight_decay_lr_ratio * learning_rate
        speedslide_reward = utilities.from_linear_schedule(
            misc.speedslide_reward_schedule, accumulated_stats["cumul_number_memories_generated"]
        )

        # ===============================================
        #   RELOAD
        # ===============================================

        for param_group in optimizer1.param_groups:
            param_group["lr"] = learning_rate
            param_group["epsilon"] = misc.adam_epsilon
            param_group["betas"] = (misc.adam_beta1, misc.adam_beta2)
        trainer.gamma = misc.gamma

        if isinstance(buffer._sampler, PrioritizedSampler):
            buffer._sampler._alpha = misc.prio_alpha
            buffer._sampler._beta = misc.prio_beta
            buffer._sampler._eps = misc.prio_epsilon

        if misc.plot_race_time_left_curves and not is_explo and (loop_number // 5) % 17 == 0:
            race_time_left_curves(rollout_results, inferer, save_dir, map_name)
            tau_curves(rollout_results, inferer, save_dir, map_name)
            distribution_curves(buffer, save_dir, online_network, target_network)
            loss_distribution(buffer, save_dir, online_network, target_network)
            # patrick_curves(rollout_results, trainer, save_dir, map_name)

        accumulated_stats["cumul_number_frames_played"] += len(rollout_results["frames"])

        # ===============================================
        #   WRITE SINGLE RACE RESULTS TO TENSORBOARD
        # ===============================================
        race_stats_to_write = {
            f"race_time_ratio_{map_name}": end_race_stats["race_time_for_ratio"] / (rollout_duration * 1000),
            f"explo_race_time_{map_status}_{map_name}"
            if is_explo
            else f"eval_race_time_{map_status}_{map_name}": end_race_stats["race_time"] / 1000,
            f"explo_race_finished_{map_status}_{map_name}"
            if is_explo
            else f"eval_race_finished_{map_status}_{map_name}": end_race_stats["race_finished"],
            f"mean_action_gap_{map_name}": -(
                np.array(rollout_results["q_values"]) - np.array(rollout_results["q_values"]).max(axis=1, initial=None).reshape(-1, 1)
            ).mean(),
            f"single_zone_reached_{map_status}_{map_name}": rollout_results["furthest_zone_idx"],
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

        if not is_explo:
            race_stats_to_write[f"avg_Q_{map_status}_{map_name}"] = np.mean(rollout_results["q_values"])

        if end_race_stats["race_finished"]:
            race_stats_to_write[f"{'explo' if is_explo else 'eval'}_race_time_finished_{map_status}_{map_name}"] = (
                end_race_stats["race_time"] / 1000
            )
            if not is_explo:
                accumulated_stats["rolling_mean_ms"][map_name] = (
                    accumulated_stats["rolling_mean_ms"].get(map_name, misc.cutoff_rollout_if_race_not_finished_within_duration_ms) * 0.9
                    + end_race_stats["race_time"] * 0.1
                )
        if (
            (not is_explo)
            and end_race_stats["race_finished"]
            and end_race_stats["race_time"] < 1.02 * accumulated_stats["rolling_mean_ms"][map_name]
        ):
            race_stats_to_write[f"eval_race_time_robust_{map_status}_{map_name}"] = end_race_stats["race_time"] / 1000
            if map_name in reference_times:
                for reference_time_name in ["author", "gold"]:
                    if reference_time_name in reference_times[map_name]:
                        reference_time = reference_times[map_name][reference_time_name]
                        race_stats_to_write[f"eval_ratio_{map_status}_{reference_time_name}_{map_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )
                        race_stats_to_write[f"eval_agg_ratio_{map_status}_{reference_time_name}"] = (
                            100 * (end_race_stats["race_time"] / 1000) / reference_time
                        )

        for i in [0]:
            race_stats_to_write[f"q_value_{i}_starting_frame_{map_name}"] = end_race_stats[f"q_value_{i}_starting_frame"]
        if not is_explo:
            for i, split_time in enumerate(
                [(e - s) / 1000 for s, e in zip(end_race_stats["cp_time_ms"][:-1], end_race_stats["cp_time_ms"][1:])]
            ):
                race_stats_to_write[f"split_{map_name}_{i}"] = split_time

        walltime_tb = time.time()
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
            if accumulated_stats["cumul_number_frames_played"] > misc.frames_before_save_best_runs:
                print("\a")
                sub_folder_name = f"{map_name}_{end_race_stats['race_time']}"
                (save_dir / "best_runs" / sub_folder_name).mkdir(parents=True, exist_ok=True)
                run_to_video.write_actions_in_tmi_format(
                    rollout_results["actions"],
                    save_dir / "best_runs" / sub_folder_name / f"{map_name}_{end_race_stats['race_time']}.inputs",
                )
                joblib.dump(
                    rollout_results["q_values"],
                    save_dir / "best_runs" / sub_folder_name / f"q_values.joblib",
                )
                torch.save(
                    online_network.state_dict(),
                    save_dir / "best_runs" / "weights1.torch",
                )
                torch.save(
                    target_network.state_dict(),
                    save_dir / "best_runs" / "weights2.torch",
                )
                torch.save(
                    optimizer1.state_dict(),
                    save_dir / "best_runs" / "optimizer1.torch",
                )
                torch.save(
                    scaler.state_dict(),
                    save_dir / "best_runs" / "scaler.torch",
                )
                shutil.copy(base_dir / "trackmania_rl" / "misc.py", save_dir / "best_runs" / sub_folder_name / "misc.py.save")

        # ===============================================
        #   FILL BUFFER WITH (S, A, R, S') transitions
        # ===============================================
        if fill_buffer:
            (
                buffer,
                buffer_test,
                number_memories_added_train,
                number_memories_added_test,
            ) = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
                buffer,
                buffer_test,
                rollout_results,
                misc.n_steps,
                misc.gamma,
                misc.discard_non_greedy_actions_in_nsteps,
                speedslide_reward,
            )

            accumulated_stats["cumul_number_memories_generated"] += number_memories_added_train + number_memories_added_test
            shared_steps.value = accumulated_stats["cumul_number_memories_generated"]
            neural_net_reset_counter += number_memories_added_train
            accumulated_stats["cumul_number_single_memories_should_have_been_used"] += (
                misc.number_times_single_memory_is_used_before_discard * number_memories_added_train
            )
            print(f" NMG={accumulated_stats['cumul_number_memories_generated']:<8}")

            # ===============================================
            #   PERIODIC RESET ?
            # ===============================================

            if neural_net_reset_counter >= misc.reset_every_n_frames_generated or single_reset_flag != misc.single_reset_flag:
                neural_net_reset_counter = 0
                single_reset_flag = misc.single_reset_flag
                accumulated_stats["cumul_number_single_memories_should_have_been_used"] += misc.additional_transition_after_reset

                untrained_iqn_network = make_untrained_iqn_network(misc.use_jit)
                utilities.soft_copy_param(online_network, untrained_iqn_network, misc.overall_reset_mul_factor)

                with torch.no_grad():
                    online_network.A_head[2].weight = utilities.linear_combination(
                        online_network.A_head[2].weight, untrained_iqn_network.A_head[2].weight, misc.last_layer_reset_factor
                    )
                    online_network.A_head[2].bias = utilities.linear_combination(
                        online_network.A_head[2].bias, untrained_iqn_network.A_head[2].bias, misc.last_layer_reset_factor
                    )
                    online_network.V_head[2].weight = utilities.linear_combination(
                        online_network.V_head[2].weight, untrained_iqn_network.V_head[2].weight, misc.last_layer_reset_factor
                    )
                    online_network.V_head[2].bias = utilities.linear_combination(
                        online_network.V_head[2].bias, untrained_iqn_network.V_head[2].bias, misc.last_layer_reset_factor
                    )

            # ===============================================
            #   LEARN ON BATCH
            # ===============================================

            if not online_network.training:
                online_network.train()

            while (
                len(buffer) >= memory_size_start_learn
                and accumulated_stats["cumul_number_single_memories_used"] + offset_cumul_number_single_memories_used
                <= accumulated_stats["cumul_number_single_memories_should_have_been_used"]
            ):
                if (random.random() < misc.buffer_test_ratio and len(buffer_test) > 0) or len(buffer) == 0:
                    loss, _ = trainer.train_on_batch(buffer_test, do_learn=False)
                    loss_test_history.append(loss)
                    print(f"BT   {loss=:<8.2e}")
                else:
                    train_start_time = time.perf_counter()
                    loss, grad_norm = trainer.train_on_batch(buffer, do_learn=True)
                    accumulated_stats["cumul_number_single_memories_used"] += (
                        10 * misc.batch_size
                        if (len(buffer) < buffer._storage.max_size and buffer._storage.max_size > 200_000)
                        else misc.batch_size
                    )  # do fewer batches while memory is not full
                    train_on_batch_duration_history.append(time.perf_counter() - train_start_time)
                    loss_history.append(loss)
                    if not math.isinf(grad_norm):
                        grad_norm_history.append(grad_norm)
                        for name, param in online_network.named_parameters():
                            layer_grad_norm_history[f"L2_grad_norm_{name}"].append(torch.norm(param.grad.detach(), 2.0).item())
                            layer_grad_norm_history[f"Linf_grad_norm_{name}"].append(torch.norm(param.grad.detach(), float("inf")).item())

                    accumulated_stats["cumul_number_batches_done"] += 1
                    print(f"B    {loss=:<8.2e} {grad_norm=:<8.2e} {train_on_batch_duration_history[-1]*1000:<8.1f}")

                    utilities.custom_weight_decay(online_network, 1 - weight_decay)
                    if accumulated_stats["cumul_number_batches_done"] % misc.send_shared_network_every_n_batches == 0:
                        with shared_network_lock:
                            uncompiled_shared_network.load_state_dict(uncompiled_online_network.state_dict())

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
                        # print("UPDATE")
                        utilities.soft_copy_param(target_network, online_network, misc.soft_update_tau)
            print("", flush=True)

        # ===============================================
        #   WRITE AGGREGATED STATISTICS TO TENSORBOARD EVERY 5 MINUTES
        # ===============================================
        if time.time() - time_last_save > 5 * 60:
            accumulated_stats["cumul_training_hours"] += (time.time() - time_last_save) / 3600
            time_last_save = time.time()

            # ===============================================
            #   COLLECT VARIOUS STATISTICS
            # ===============================================
            step_stats = {
                "gamma": misc.gamma,
                "n_steps": misc.n_steps,
                "epsilon": utilities.from_exponential_schedule(misc.epsilon_schedule, shared_steps.value),
                "epsilon_boltzmann": utilities.from_exponential_schedule(misc.epsilon_boltzmann_schedule, shared_steps.value),
                "tau_epsilon_boltzmann": misc.tau_epsilon_boltzmann,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "discard_non_greedy_actions_in_nsteps": misc.discard_non_greedy_actions_in_nsteps,
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
                            f"{key}_c98": np.quantile(val, 0.98),
                            f"{key}_max": np.max(val),
                        }
                    )
            if isinstance(buffer._sampler, PrioritizedSampler):
                all_priorities = np.array([buffer._sampler._sum_tree.at(i) for i in range(len(buffer))])
                step_stats.update(
                    {
                        "priorities_min": np.min(all_priorities),
                        "priorities_q1": np.quantile(all_priorities, 0.1),
                        "priorities_mean": np.mean(all_priorities),
                        "priorities_median": np.quantile(all_priorities, 0.5),
                        "priorities_q3": np.quantile(all_priorities, 0.75),
                        "priorities_d9": np.quantile(all_priorities, 0.9),
                        "priorities_c98": np.quantile(all_priorities, 0.98),
                        "priorities_max": np.max(all_priorities),
                    }
                )
            for key, value in accumulated_stats.items():
                if key not in ["alltime_min_ms", "rolling_mean_ms"]:
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

            if online_network.training:
                online_network.eval()
            tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")
            per_quantile_output = inferer.infer_network(rollout_results["frames"][0], rollout_results["state_float"][0], tau)
            for i, std in enumerate(list(per_quantile_output.std(axis=0))):
                step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std

            # ===============================================
            #   WRITE TO TENSORBOARD
            # ===============================================

            walltime_tb = time.time()
            for name, param in online_network.named_parameters():
                tensorboard_writer.add_scalar(
                    tag=f"layer_{name}_L2",
                    scalar_value=np.sqrt((param**2).mean().detach().cpu().item()),
                    global_step=accumulated_stats["cumul_number_frames_played"],
                    walltime=walltime_tb,
                )
            assert len(optimizer1.param_groups) == 1
            try:
                for p, (name, _) in zip(optimizer1.param_groups[0]["params"], online_network.named_parameters()):
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

            previous_alltime_min = previous_alltime_min or copy.deepcopy(accumulated_stats["alltime_min_ms"])

            tensorboard_writer.add_text(
                "times_summary",
                f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')} "
                + " ".join(
                    [
                        f"{'**' if v < previous_alltime_min[k] else ''}{k}: {v / 1000:.2f}{'**' if v < previous_alltime_min[k] else ''}"
                        for k, v in accumulated_stats["alltime_min_ms"].items()
                    ]
                ),
                global_step=accumulated_stats["cumul_number_frames_played"],
                walltime=walltime_tb,
            )

            previous_alltime_min = copy.deepcopy(accumulated_stats["alltime_min_ms"])

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
            #   HIGH PRIORITY TRANSITIONS
            # ===============================================
            if misc.make_highest_prio_figures and isinstance(buffer._sampler, PrioritizedSampler):
                highest_prio_transitions(buffer, save_dir)

            # ===============================================
            #   SAVE
            # ===============================================

            torch.save(online_network.state_dict(), save_dir / "weights1.torch")
            torch.save(target_network.state_dict(), save_dir / "weights2.torch")
            torch.save(optimizer1.state_dict(), save_dir / "optimizer1.torch")
            torch.save(scaler.state_dict(), save_dir / "scaler.torch")
            joblib.dump(accumulated_stats, save_dir / "accumulated_stats.joblib")
