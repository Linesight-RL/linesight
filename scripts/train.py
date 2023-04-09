import random
import time
import weakref
from collections import defaultdict
from pathlib import Path

import dxcam
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

import trackmania_rl
from trackmania_rl import buffer_management, misc, nn_utilities, rollout
import trackmania_rl.agents.noisy_iqn_pal2 as noisy_iqn_pal2
from trackmania_rl.experience_replay.basic_experience_replay import BasicExperienceReplay

# from trackmania_rl.experience_replay.prioritized_experience_replay import PrioritizedExperienceReplay

base_dir = Path(__file__).resolve().parents[1]
save_dir = base_dir / "save"

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(43)
torch.manual_seed(43)
random.seed(43)
np.random.seed(43)

plt.style.use("seaborn")

# ========================================================
# Create new stuff
# ========================================================
model1 = noisy_iqn_pal2.Agent(
    float_inputs_dim=misc.float_input_dim,
    float_hidden_dim=misc.float_hidden_dim,
    conv_head_output_dim=misc.conv_head_output_dim,
    dense_hidden_dimension=misc.dense_hidden_dimension,
    iqn_embedding_dimension=misc.iqn_embedding_dimension,
    n_actions=len(misc.inputs),
    float_inputs_mean=misc.float_inputs_mean,
    float_inputs_std=misc.float_inputs_std,
).to("cuda")
model2 = noisy_iqn_pal2.Agent(
    float_inputs_dim=misc.float_input_dim,
    float_hidden_dim=misc.float_hidden_dim,
    conv_head_output_dim=misc.conv_head_output_dim,
    dense_hidden_dimension=misc.dense_hidden_dimension,
    iqn_embedding_dimension=misc.iqn_embedding_dimension,
    n_actions=len(misc.inputs),
    float_inputs_mean=misc.float_inputs_mean,
    float_inputs_std=misc.float_inputs_std,
).to("cuda")

print(model1)
optimizer1 = torch.optim.RAdam(model1.parameters(), lr=misc.learning_rate)
# optimizer2 = torch.optim.RAdam(model2.parameters(), lr=misc.learning_rate)
scaler = torch.cuda.amp.GradScaler()
# buffer = PrioritizedExperienceReplay(
#     capacity=misc.memory_size,
#     sample_with_segments=misc.prio_sample_with_segments,
#     prio_alpha=misc.prio_alpha,
#     prio_beta=misc.prio_beta,
#     prio_epsilon=misc.prio_epsilon,
# )
buffer = BasicExperienceReplay(capacity=misc.memory_size)
# fast_stats_tracker = defaultdict(list)
# slow_stats_tracker = defaultdict(list)
# ========================================================
# Load existing stuff
# ========================================================
# noinspection PyBroadException
# try:
model1.load_state_dict(torch.load(save_dir / "weights1.torch"))
model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
optimizer1.load_state_dict(torch.load(save_dir / "optimizer1.torch"))
# optimizer2.load_state_dict(torch.load(save_dir / "optimizer2.torch"))
print(" =========================     Weights loaded !     ================================")
slow_stats_tracker = joblib.load(save_dir / "slow_stats_tracker.joblib")
fast_stats_tracker = joblib.load(save_dir / "fast_stats_tracker.joblib")
print(" =========================      Stats loaded !      ================================")

# ========================================================
# Make the trainer
# ========================================================

trainer = noisy_iqn_pal2.Trainer(
    model=model1,
    model2=model2,
    optimizer=optimizer1,
    scaler=scaler,
    batch_size=misc.batch_size,
    iqn_k=misc.iqn_k,
    iqn_n=misc.iqn_n,
    iqn_kappa=misc.iqn_kappa,
    epsilon=misc.epsilon,
    gamma=misc.gamma,
    n_steps=misc.n_steps,
    AL_alpha=misc.AL_alpha,
)

# ========================================================
# Training loop
# ========================================================
number_memories_generated = 0
number_batches_done = 0
number_target_network_updates = 0

model1.train()

time_last_save = time.time()
time_last_buffer_save = time.time()

dxcam.__factory._camera_instances = weakref.WeakValueDictionary()
tmi = rollout.TMInterfaceManager(
    running_speed=misc.running_speed,
    run_steps_per_action=misc.run_steps_per_action,
    max_time=misc.max_rollout_time_ms,
    interface_name="TMInterface0",
)

while True:
    # ===============================================
    #   PLAY ONE ROUND
    # ===============================================
    rollout_start_time = time.time()
    trainer.epsilon = 5 * misc.epsilon if number_memories_generated < misc.number_memories_generated_high_exploration else misc.epsilon
    rollout_results = tmi.rollout(
        exploration_policy=trainer.get_exploration_action,
        stats_tracker=fast_stats_tracker,
    )

    fast_stats_tracker["race_time_ratio"].append(
        fast_stats_tracker["race_time"][-1] / ((time.time() - rollout_start_time) * 1000))

    buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(buffer,
                                                                                                 rollout_results,
                                                                                                 misc.n_steps)
    number_memories_generated += number_memories_added
    print(f" NMG={number_memories_generated:<8}")

    # ===============================================
    #   LEARN ON BATCH
    # ===============================================
    while number_memories_generated >= misc.memory_size_start_learn and number_batches_done * misc.batch_size <= misc.number_times_single_memory_is_used_before_discard * (
            number_memories_generated - misc.virtual_memory_size_start_learn
    ):
        train_start_time = time.time()
        mean_q_values, loss = trainer.train_on_batch(buffer)
        fast_stats_tracker["train_on_batch_duration"].append(time.time() - train_start_time)
        fast_stats_tracker["loss"].append(loss)
        number_batches_done += 1
        # print("B ", end="")
        print(f"B {loss=:<12} {mean_q_values=}")

        # ===============================================
        #   UPDATE TARGET NETWORK
        # ===============================================
        if (
                misc.number_memories_trained_on_between_target_network_updates * number_target_network_updates
                <= number_batches_done * misc.batch_size
        ):
            number_target_network_updates += 1
            # print("UPDATE ", end="")
            print("UPDATE")
            # model1, model2 = model2, model1
            # optimizer1, optimizer2 = optimizer2, optimizer1
            # trainer.optimizer = optimizer1

            # B UPDATE Traceback (most recent call last):
            #   File "C:\Users\chopi\projects\trackmania_rl\scripts\train.py", line 145, in <module>
            #     mean_q_values, loss = trainer.train_on_batch(buffer)
            #   File "c:\users\chopi\projects\trackmania_rl\trackmania_rl\agents\noisy_iqn.py", line 246, in train_on_batch
            #     self.scaler.step(self.optimizer)
            #   File "C:\Users\chopi\tools\mambaforge\envs\tm309\lib\site-packages\torch\cuda\amp\grad_scaler.py", line 368, in step
            #     assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."
            # AssertionError: No inf checks were recorded for this optimizer.

            nn_utilities.soft_copy_param(model2, model1, misc.soft_update_tau)
            # model2.load_state_dict(model.state_dict())
    print("")

    # ===============================================
    #   STATISTICS EVERY NOW AND THEN
    # ===============================================
    if time.time() > time_last_save + 60 * 15:  # every 15 minutes

        time_last_save = time.time()

        # ===============================================
        #   EVAL RACE
        # ===============================================
        model1.reset_noise()
        model1.eval()
        trainer.epsilon = 0
        eval_stats_tracker = defaultdict(list)
        rollout_results = tmi.rollout(
            exploration_policy=trainer.get_exploration_action,
            stats_tracker=eval_stats_tracker,
        )
        trainer.epsilon = misc.epsilon
        model1.train()
        buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(buffer,
                                                                                                     rollout_results,
                                                                                                     misc.n_steps)
        number_memories_generated += number_memories_added

        print("EVAL EVAL EVAL EVAL EVAL")

        # ===============================================
        #   SPREAD 
        # ===============================================

        # Faire 100 tirages avec noisy, et les tau de IQN fixés
        tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")
        state_img_tensor = torch.as_tensor(np.expand_dims(rollout_results["frames"][0], axis=0)).to(
            "cuda", memory_format=torch.channels_last, non_blocking=True
        )
        state_float_tensor = torch.as_tensor(np.expand_dims(rollout_results["floats"][0], axis=0)).to("cuda",
                                                                                                  non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                list_of_q_values = []
                for i in range(400):
                    model1.reset_noise()
                    list_of_q_values.append(model1(state_img_tensor, state_float_tensor, misc.iqn_k, tau=tau)[
                        0].cpu().numpy().mean(axis=0))

        for i, std in enumerate(list(np.array(list_of_q_values).std(axis=0))):
            slow_stats_tracker[f"std_due_to_noisy_for_action{i}"].append(std)

        # Désactiver noisy, tirer des tau équitablement répartis 
        model1.eval()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                per_quantile_output, _ = model1(state_img_tensor, state_float_tensor, misc.iqn_k, tau=tau)

        for i, std in enumerate(list(per_quantile_output.cpu().numpy().std(axis=0))):
            slow_stats_tracker[f"std_within_iqn_quantiles_for_action{i}"].append(std)
        model1.train()

        # ===============================================
        #   FILL SLOW_STATS_TRACKER
        # ===============================================
        for i in range(len(misc.inputs)):
            # slow_stats_tracker[f"d1_q_values_starting_frame_{i}"] = [0] * (
            #         len(slow_stats_tracker["min_race_time"]) - len(slow_stats_tracker[f"d1_q_values_starting_frame_{i}"])) + \
            #                                                   slow_stats_tracker[
            #                                                       f"d1_q_values_starting_frame_{i}"]  # Temporary fix to stats tracker

            slow_stats_tracker[f"median_q_values_starting_frame_{i}"] = list(-np.abs(slow_stats_tracker[f"median_q_values_starting_frame_{i}"]))

            slow_stats_tracker[f"gap_median_q_values_starting_frame_{i}"] = list(-np.abs(slow_stats_tracker[f"gap_median_q_values_starting_frame_{i}"]))

        slow_stats_tracker[r"%race finished"].append(np.array(fast_stats_tracker["race_finished"]).mean())
        slow_stats_tracker["min_race_time"].append(np.array(fast_stats_tracker["race_time"]).min(initial=None))
        slow_stats_tracker["d1_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.1))
        slow_stats_tracker["q1_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.25))
        slow_stats_tracker["median_race_time"].append(np.median(np.array(fast_stats_tracker["race_time"])))
        slow_stats_tracker["q3_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.75))
        slow_stats_tracker["d9_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.9))
        slow_stats_tracker["eval_race_time"].append(eval_stats_tracker["race_time"][-1])

        slow_stats_tracker["mean_q_value_starting_frame"].append(
            np.array(fast_stats_tracker["q_value_starting_frame"]).mean())
        slow_stats_tracker["d1_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.1))
        slow_stats_tracker["q1_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.25))
        slow_stats_tracker["median_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.5))
        slow_stats_tracker["q3_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.75))
        slow_stats_tracker["d9_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.9))

        for i in range(len(misc.inputs)):
            slow_stats_tracker[f"d1_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"q_values_starting_frame_{i}"]), 0.1))
            slow_stats_tracker[f"q1_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"q_values_starting_frame_{i}"]), 0.25))
            slow_stats_tracker[f"median_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"q_values_starting_frame_{i}"]), 0.5))
            slow_stats_tracker[f"q3_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"q_values_starting_frame_{i}"]), 0.75))
            slow_stats_tracker[f"d9_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"q_values_starting_frame_{i}"]), 0.9))

            slow_stats_tracker[f"gap_d1_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"gap_q_values_starting_frame_{i}"]), 0.1))
            slow_stats_tracker[f"gap_q1_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"gap_q_values_starting_frame_{i}"]), 0.25))
            slow_stats_tracker[f"gap_median_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"gap_q_values_starting_frame_{i}"]), 0.5))
            slow_stats_tracker[f"gap_q3_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"gap_q_values_starting_frame_{i}"]), 0.75))
            slow_stats_tracker[f"gap_d9_q_values_starting_frame_{i}"].append(
                np.quantile(np.array(fast_stats_tracker[f"gap_q_values_starting_frame_{i}"]), 0.9))

        slow_stats_tracker["d1_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.1))
        slow_stats_tracker["q1_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.25))
        slow_stats_tracker["median_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.5))
        slow_stats_tracker["q3_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.75))
        slow_stats_tracker["d9_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.9))

        slow_stats_tracker["mean_loss"].append(np.array(fast_stats_tracker["loss"]).mean())
        slow_stats_tracker["d1_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.1))
        slow_stats_tracker["q1_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.25))
        slow_stats_tracker["median_loss"].append(np.median(np.array(fast_stats_tracker["loss"])))
        slow_stats_tracker["q3_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.75))
        slow_stats_tracker["d9_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.9))

        slow_stats_tracker[r"%light_desynchro"].append(
            np.array(fast_stats_tracker["n_ors_light_desynchro"]).sum()
            / (np.array(fast_stats_tracker["race_time"]).sum() / (misc.ms_per_run_step * misc.run_steps_per_action))
        )
        slow_stats_tracker[r"%consecutive_frames_equal"].append(
            np.array(fast_stats_tracker["n_two_consecutive_frames_equal"]).sum()
            / (np.array(fast_stats_tracker["race_time"]).sum() / (misc.ms_per_run_step * misc.run_steps_per_action))
        )
        slow_stats_tracker[r"n_tmi_protection"].append(
            np.array(fast_stats_tracker["n_frames_tmi_protection_triggered"]).sum())

        slow_stats_tracker[r"race_time_ratio"].append(np.median(np.array(fast_stats_tracker["race_time_ratio"])))
        slow_stats_tracker[r"train_on_batch_duration"].append(
            np.median(np.array(fast_stats_tracker["train_on_batch_duration"])))

        slow_stats_tracker["number_memories_generated"].append(number_memories_generated)

        # ===============================================
        #   MAKE THE PLOTS
        # ===============================================

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker[r"%light_desynchro"], label=r"%light_desynchro")
        ax.legend()
        fig.savefig(base_dir / "figures" / "light_desynchro.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker[r"%consecutive_frames_equal"], label=r"%consecutive_frames_equal")
        ax.legend()
        fig.savefig(base_dir / "figures" / "consecutive_frames_equal.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker[r"n_tmi_protection"], label=r"n_tmi_protection")
        ax.legend()
        fig.savefig(base_dir / "figures" / "tmi_protection.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker[r"%race finished"], label=r"%race finished")
        ax.legend()
        fig.savefig(base_dir / "figures" / "race_finished.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.vlines(np.where(np.array(
            slow_stats_tracker["number_memories_generated"][-misc.figures_max_steps_displayed:][:-1]) > np.array(
            slow_stats_tracker["number_memories_generated"][-misc.figures_max_steps_displayed:][1:]))[0], -1000000,
                  1000000, color="r", linewidth=1.5, linestyles='dashed')
        ax.plot(slow_stats_tracker["d1_q_value_starting_frame"][-misc.figures_max_steps_displayed:], "b",
                label="d1_q_value_starting_frame")
        ax.plot(slow_stats_tracker["q1_q_value_starting_frame"][-misc.figures_max_steps_displayed:], "b",
                label="q1_q_value_starting_frame")
        ax.plot(slow_stats_tracker["median_q_value_starting_frame"][-misc.figures_max_steps_displayed:], "b",
                label="median_q_value_starting_frame")
        ax.plot(slow_stats_tracker["q3_q_value_starting_frame"][-misc.figures_max_steps_displayed:], "b",
                label="q3_q_value_starting_frame")
        ax.plot(slow_stats_tracker["d9_q_value_starting_frame"][-misc.figures_max_steps_displayed:], "b",
                label="d9_q_value_starting_frame")
        ax.plot(slow_stats_tracker["d1_rollout_sum_rewards"][-misc.figures_max_steps_displayed:], "r",
                label="d1_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["q1_rollout_sum_rewards"][-misc.figures_max_steps_displayed:], "r",
                label="q1_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["median_rollout_sum_rewards"][-misc.figures_max_steps_displayed:], "r",
                label="median_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["q3_rollout_sum_rewards"][-misc.figures_max_steps_displayed:], "r",
                label="q3_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["d9_rollout_sum_rewards"][-misc.figures_max_steps_displayed:], "r",
                label="d9_rollout_sum_rewards")
        ax.legend()
        ax.set_ylim([-1.4, -0.9])
        fig.savefig(base_dir / "figures" / "start_q.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["mean_loss"][-misc.figures_max_steps_displayed:], label="mean_loss")
        ax.plot(slow_stats_tracker["d1_loss"][-misc.figures_max_steps_displayed:], label="d1_loss")
        ax.plot(slow_stats_tracker["q1_loss"][-misc.figures_max_steps_displayed:], label="q1_loss")
        ax.plot(slow_stats_tracker["median_loss"][-misc.figures_max_steps_displayed:], label="median_loss")
        ax.plot(slow_stats_tracker["q3_loss"][-misc.figures_max_steps_displayed:], label="q3_loss")
        ax.plot(slow_stats_tracker["d9_loss"][-misc.figures_max_steps_displayed:], label="d9_loss")
        ax.legend()
        ax.set_yscale("log")
        fig.savefig(base_dir / "figures" / "loss.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.vlines(np.where(np.array(
            slow_stats_tracker["number_memories_generated"][-misc.figures_max_steps_displayed:][:-1]) > np.array(
            slow_stats_tracker["number_memories_generated"][-misc.figures_max_steps_displayed:][1:]))[0], -1000000,
                  1000000, color="r", linewidth=1.5, linestyles='dashed')
        ax.plot(slow_stats_tracker["eval_race_time"][-misc.figures_max_steps_displayed:], label="eval_race_time",
                linewidth=0.75)
        ax.plot(slow_stats_tracker["d9_race_time"][-misc.figures_max_steps_displayed:], label="d9_race_time")
        ax.plot(slow_stats_tracker["q3_race_time"][-misc.figures_max_steps_displayed:], label="q3_race_time")
        ax.plot(slow_stats_tracker["median_race_time"][-misc.figures_max_steps_displayed:], label="median_race_time")
        ax.plot(slow_stats_tracker["q1_race_time"][-misc.figures_max_steps_displayed:], label="q1_race_time")
        ax.plot(slow_stats_tracker["d1_race_time"][-misc.figures_max_steps_displayed:], label="d1_race_time")
        ax.plot(slow_stats_tracker["min_race_time"][-misc.figures_max_steps_displayed:], label="min_race_time")
        ax.legend()
        ax.set_ylim([11700, 14200])
        fig.suptitle(
            f"min: {slow_stats_tracker['min_race_time'][-1] / 1000:.2f}, eval: {slow_stats_tracker['eval_race_time'][-1] / 1000:.2f}, d1: {slow_stats_tracker['d1_race_time'][-1] / 1000:.2f}, q1: {slow_stats_tracker['q1_race_time'][-1] / 1000:.2f}, med: {slow_stats_tracker['median_race_time'][-1] / 1000:.2f}, q3: {slow_stats_tracker['q3_race_time'][-1] / 1000:.2f}, d9: {slow_stats_tracker['d9_race_time'][-1] / 1000:.2f}"
        )
        fig.savefig(base_dir / "figures" / "race_time.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.vlines(np.where(np.array(
            slow_stats_tracker["number_memories_generated"][-misc.figures_max_steps_displayed:][:-1]) > np.array(
            slow_stats_tracker["number_memories_generated"][-misc.figures_max_steps_displayed:][1:]))[0], -1000000,
                  1000000, color="r", linewidth=1.5, linestyles='dashed')
        for i in range(len(misc.inputs)):
            ax.plot(slow_stats_tracker[f"gap_median_q_values_starting_frame_{i}"][-misc.figures_max_steps_displayed:], label=f"gap_median_q_values_starting_frame_{i}")
        ax.legend()
        ax.set_ylim([-0.30, 0.005])
        fig.savefig(base_dir / "figures" / "actions_gap_starting_frame.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 9))
        for i in range(len(misc.inputs)):
            ax.plot(slow_stats_tracker[f"median_q_values_starting_frame_{i}"][-misc.figures_max_steps_displayed:], label=f"median_q_values_starting_frame_{i}")
        ax.legend()
        fig.savefig(base_dir / "figures" / "actions_values_starting_frame.png")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(14, 9))
        for i in range(len(misc.inputs)):
            ax.plot(slow_stats_tracker[f"std_within_iqn_quantiles_for_action{i}"][-misc.figures_max_steps_displayed:], label=f"std_within_iqn_quantiles_for_action{i}")
        ax.legend()
        fig.savefig(base_dir / "figures" / "std_within_iqn_quantiles.png")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(14, 9))
        for i in range(len(misc.inputs)):
            ax.plot(slow_stats_tracker[f"std_due_to_noisy_for_action{i}"][-misc.figures_max_steps_displayed:], label=f"std_due_to_noisy_for_action{i}")
        ax.legend()
        fig.savefig(base_dir / "figures" / "std_due_to_noisy.png")
        plt.close()
 

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["train_on_batch_duration"], label="batch_duration")
        ax.legend()
        fig.savefig(base_dir / "figures" / "train_on_batch_duration.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["race_time_ratio"], label="race_time_ratio")
        ax.legend()
        fig.savefig(base_dir / "figures" / "race_time_ratio.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["number_memories_generated"], label="number_memories_generated")
        ax.legend()
        fig.savefig(base_dir / "figures" / "number_memories_generated.png")
        plt.close()

        # ===============================================
        #   Buffer stats
        # ===============================================

        # TODO : fix it for PER
        # print("Mean in buffer", np.array([experience.state_float for experience in buffer.buffer]).mean(axis=0))
        # print("Std in buffer ", np.array([experience.state_float for experience in buffer.buffer]).std(axis=0))

        # ===============================================
        #   CLEANUP
        # ===============================================

        fast_stats_tracker["race_finished"] = fast_stats_tracker["race_finished"][-400:]
        fast_stats_tracker["race_time"] = fast_stats_tracker["race_time"][-400:]
        fast_stats_tracker["q_value_starting_frame"] = fast_stats_tracker["q_value_starting_frame"][-400:]
        fast_stats_tracker["rollout_sum_rewards"] = fast_stats_tracker["rollout_sum_rewards"][-400:]
        fast_stats_tracker["loss"] = fast_stats_tracker["loss"][-400:]
        fast_stats_tracker["n_ors_light_desynchro"] = fast_stats_tracker["n_ors_light_desynchro"][-400:]
        fast_stats_tracker["n_two_consecutive_frames_equal"] = fast_stats_tracker["n_two_consecutive_frames_equal"][-400:]
        for i in range(len(misc.inputs)):
            fast_stats_tracker[f"q_values_starting_frame_{i}"] = fast_stats_tracker[f"q_values_starting_frame_{i}"][-400:]
            fast_stats_tracker[f"gap_q_values_starting_frame_{i}"] = fast_stats_tracker[f"gap_q_values_starting_frame_{i}"][-400:]
        fast_stats_tracker["n_frames_tmi_protection_triggered"].clear()
        fast_stats_tracker["train_on_batch_duration"].clear()
        fast_stats_tracker["race_time_ratio"].clear()

        # ===============================================
        #   SAVE
        # ===============================================
        torch.save(model1.state_dict(), save_dir / "weights1.torch")
        torch.save(model2.state_dict(), save_dir / "weights2.torch")
        torch.save(optimizer1.state_dict(), save_dir / "optimizer1.torch")
        # torch.save(optimizer2.state_dict(), save_dir / "optimizer2.torch")
        joblib.dump(slow_stats_tracker, save_dir / "slow_stats_tracker.joblib")
        joblib.dump(fast_stats_tracker, save_dir / "fast_stats_tracker.joblib")

    # if time.time() > time_last_buffer_save + 60 * 60 * 6:  # every 2 hours
    #     print("SAVING MODEL AND OPTIMIZER")
    #     time_last_buffer_save = time.time()
    #     joblib.dump(buffer, save_dir / "buffer.joblib")

# %%
#
# model.reset_noise()
#
# dxcam.__factory._camera_instances = weakref.WeakValueDictionary()
# tmi = rollout.TMInterfaceManager(
#     running_speed=misc.running_speed, run_steps_per_action=misc.run_steps_per_action, max_time=misc.max_rollout_time_ms
# )
# model.eval()
# rollout_results = tmi.rollout(
#     exploration_policy=partial(learning_algorithm.get_exploration_action, model, 0),
# )
# model.train()
#
# # %%
for i in range(40):
    plt.imshow(rollout_results["frames"][i][0, :, :], cmap='gray')
    plt.gcf().suptitle(f"{(i * 5) // 100} {(i * 5) % 100}")
    plt.show()
