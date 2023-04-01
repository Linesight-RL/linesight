import datetime
import random
import time
import weakref
from pathlib import Path

import dxcam
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

import trackmania_rl
import trackmania_rl.agents.noisy_iqn as learning_algorithm
from trackmania_rl import buffer_management, misc, nn_utilities, rollout
from trackmania_rl.experience_replay.basic_experience_replay import BasicExperienceReplay

base_dir = Path(__file__).resolve().parents[1]
save_dir = base_dir / "save"

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(43)
torch.manual_seed(43)
random.seed(43)

plt.style.use("seaborn")

# ========================================================
# Create new stuff
# ========================================================
model = trackmania_rl.agents.noisy_iqn.Agent(
    float_inputs_dim=misc.float_input_dim,
    float_hidden_dim=misc.float_hidden_dim,
    conv_head_output_dim=misc.conv_head_output_dim,
    dense_hidden_dimension=misc.dense_hidden_dimension,
    iqn_embedding_dimension=misc.iqn_embedding_dimension,
    n_actions=len(misc.inputs),
    float_inputs_mean=misc.float_inputs_mean,
    float_inputs_std=misc.float_inputs_std,
).to("cuda")
model2 = trackmania_rl.agents.noisy_iqn.Agent(
    float_inputs_dim=misc.float_input_dim,
    float_hidden_dim=misc.float_hidden_dim,
    conv_head_output_dim=misc.conv_head_output_dim,
    dense_hidden_dimension=misc.dense_hidden_dimension,
    iqn_embedding_dimension=misc.iqn_embedding_dimension,
    n_actions=len(misc.inputs),
    float_inputs_mean=misc.float_inputs_mean,
    float_inputs_std=misc.float_inputs_std,
).to("cuda")

print(model)
optimizer = torch.optim.RAdam(model.parameters(), lr=misc.learning_rate)
# buffer = PrioritizedExperienceReplay(
#     capacity=misc.memory_size,
#     sample_with_segments=misc.prio_sample_with_segments,
#     prio_alpha=misc.prio_alpha,
#     prio_beta=misc.prio_beta,
#     prio_epsilon=misc.prio_epsilon,
# )
buffer = BasicExperienceReplay(capacity=misc.memory_size)
scaler = torch.cuda.amp.GradScaler()

# ========================================================
# Load existing stuff
# ========================================================
# noinspection PyBroadException
# try:
#     model.load_state_dict(torch.load(save_dir / "weights.torch"))
#     model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
#     print(" =========================     Weights loaded !      ================================")
# except:
#     print(" ========     Could not find weights file, left default initialization    ===========")
#     model2.load_state_dict(model.state_dict()) Why would we do that ???

# optimizer = torch.optim.SGD(model.parameters(), lr=misc.learning_rate, momentum=0.2)

# try:
#     # optimizer.load_state_dict(torch.load(save_dir / "optimizer.torch"))
# except:
#     print("Could not find optimizer file, left default initialization")

# buffer = joblib.load(save_dir / "buffer.joblib")

trainer = trackmania_rl.agents.noisy_iqn.Trainer(
    model=model,
    model2=model2,
    optimizer=optimizer,
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

fast_stats_tracker = defaultdict(list)
slow_stats_tracker = defaultdict(list)

# ========================================================
# Training loop
# ========================================================
number_memories_generated = 0
number_batches_done = 0
number_target_network_updates = 0

model.train()

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
    rollout_results = tmi.rollout(
        exploration_policy=trainer.get_exploration_action,
        stats_tracker=fast_stats_tracker,
    )
    fast_stats_tracker["race_time_ratio"].append(
        (time.time() - rollout_start_time) * 1000 / fast_stats_tracker["race_time"][-1]
    )

    buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
        buffer, rollout_results, misc.n_steps
    )
    number_memories_generated += number_memories_added
    print(f"{number_memories_generated=}")

    # ===============================================
    #   LEARN ON BATCH
    # ===============================================
    while number_batches_done * misc.batch_size <= misc.number_times_single_memory_is_used_before_discard * (
            number_memories_generated - misc.memory_size_start_learn
    ):
        train_start_time = time.time()
        mean_q_values, loss = trainer.train_on_batch(buffer)
        fast_stats_tracker["train_on_batch_duration"].append(time.time() - train_start_time)
        fast_stats_tracker["loss"].append(loss)
        number_batches_done += 1

    # ===============================================
    #   UPDATE TARGET NETWORK
    # ===============================================
    if (
            misc.number_memories_trained_on_between_target_network_updates * number_target_network_updates
            <= number_batches_done * misc.batch_size
    ):
        number_target_network_updates += 1
        # print("------- ------- SOFT UPDATE TARGET NETWORK")

        nn_utilities.soft_copy_param(model2, model, misc.soft_update_tau)
        # model2.load_state_dict(model.state_dict())

    # ===============================================
    #   STATISTICS EVERY NOW AND THEN
    # ===============================================
    if time.time() > time_last_save + 60*15:  # every 15 minutes
        # ===============================================
        #   EVAL RACE
        # ===============================================
        model.reset_noise()
        model.eval()
        trainer.epsilon = 0
        eval_stats_tracker = defaultdict(list)
        rollout_results = tmi.rollout(
            exploration_policy=trainer.get_exploration_action,
            stats_tracker=eval_stats_tracker,
        )
        trainer.epsilon = misc.epsilon
        model.train()
        buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
            buffer, rollout_results, misc.n_steps
        )
        number_memories_generated += number_memories_added

        # ===============================================
        #   FILL SLOW_STATS_TRACKER
        # ===============================================
        slow_stats_tracker[r"%race finished"].append(np.array(fast_stats_tracker["race_finished"]).mean())
        slow_stats_tracker["min_race_time"].append(np.array(fast_stats_tracker["race_time"]).min(initial=None))
        slow_stats_tracker["d1_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.1))
        slow_stats_tracker["q1_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.25))
        slow_stats_tracker["median_race_time"].append(np.median(np.array(fast_stats_tracker["race_time"])))
        slow_stats_tracker["q3_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.75))
        slow_stats_tracker["d9_race_time"].append(np.quantile(np.array(fast_stats_tracker["race_time"]), 0.9))
        slow_stats_tracker["eval_race_time"].append(eval_stats_tracker["race_time"][-1])

        slow_stats_tracker["mean_q_value_starting_frame"].append(
            np.array(fast_stats_tracker["q_value_starting_frame"]).mean()
        )
        slow_stats_tracker["d1_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.1)
        )
        slow_stats_tracker["q1_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.25)
        )
        slow_stats_tracker["median_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.5)
        )
        slow_stats_tracker["q3_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.75)
        )
        slow_stats_tracker["d9_q_value_starting_frame"].append(
            np.quantile(np.array(fast_stats_tracker["q_value_starting_frame"]), 0.9)
        )

        slow_stats_tracker["d1_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.1)
        )
        slow_stats_tracker["q1_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.25)
        )
        slow_stats_tracker["median_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.5)
        )
        slow_stats_tracker["q3_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.75)
        )
        slow_stats_tracker["d9_rollout_sum_rewards"].append(
            np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"]), 0.9)
        )

        slow_stats_tracker["mean_loss"].append(np.array(fast_stats_tracker["loss"]).mean())
        slow_stats_tracker["d1_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.1))
        slow_stats_tracker["q1_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.25))
        slow_stats_tracker["median_loss"].append(np.median(np.array(fast_stats_tracker["loss"])))
        slow_stats_tracker["q3_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.75))
        slow_stats_tracker["d9_loss"].append(np.quantile(np.array(fast_stats_tracker["loss"]), 0.9))

        slow_stats_tracker[r"%light_desynchro"].append(
            np.array(fast_stats_tracker["n_ors_light_desynchro"]).sum()
            / (np.array(fast_stats_tracker["race_time"]).sum() / misc.ms_per_run_step)
        )
        slow_stats_tracker[r"n_tmi_protection"].append(
            np.array(fast_stats_tracker["n_frames_tmi_protection_triggered"]).sum()
        )

        slow_stats_tracker[r"race_time_ratio"].append(np.array(fast_stats_tracker["race_time_ratio"]).mean())
        slow_stats_tracker[r"train_on_batch_duration"].append(
            np.array(fast_stats_tracker["train_on_batch_duration"]).mean()
        )

        # ===============================================
        #   MAKE THE PLOTS
        # ===============================================

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker[r"%light_desynchro"], label=r"%light_desynchro")
        ax.legend()
        fig.savefig(base_dir / "figures" / "light_desynchro.png")

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker[r"n_tmi_protection"], label=r"n_tmi_protection")
        ax.legend()
        fig.savefig(base_dir / "figures" / "tmi_protection.png")

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker[r"%race finished"], label=r"%race finished")
        ax.legend()
        fig.savefig(base_dir / "figures" / "race_finished.png")

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["d1_q_value_starting_frame"], "b", label="d1_q_value_starting_frame")
        ax.plot(slow_stats_tracker["q1_q_value_starting_frame"], "b", label="q1_q_value_starting_frame")
        ax.plot(slow_stats_tracker["median_q_value_starting_frame"], "b", label="median_q_value_starting_frame")
        ax.plot(slow_stats_tracker["q3_q_value_starting_frame"], "b", label="q3_q_value_starting_frame")
        ax.plot(slow_stats_tracker["d9_q_value_starting_frame"], "b", label="d9_q_value_starting_frame")
        ax.plot(slow_stats_tracker["d1_rollout_sum_rewards"], "r", label="d1_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["q1_rollout_sum_rewards"], "r", label="q1_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["median_rollout_sum_rewards"], "r", label="median_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["q3_rollout_sum_rewards"], "r", label="q3_rollout_sum_rewards")
        ax.plot(slow_stats_tracker["d9_rollout_sum_rewards"], "r", label="d9_rollout_sum_rewards")
        ax.legend()
        fig.savefig(base_dir / "figures" / "start_q.png")

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["mean_loss"], label="mean_loss")
        ax.plot(slow_stats_tracker["d1_loss"], label="d1_loss")
        ax.plot(slow_stats_tracker["q1_loss"], label="q1_loss")
        ax.plot(slow_stats_tracker["median_loss"], label="median_loss")
        ax.plot(slow_stats_tracker["q3_loss"], label="q3_loss")
        ax.plot(slow_stats_tracker["d9_loss"], label="d9_loss")
        ax.legend()
        fig.savefig(base_dir / "figures" / "loss.png")

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["eval_race_time"], label="eval_race_time")
        ax.plot(slow_stats_tracker["d1_race_time"], label="d1_race_time")
        ax.plot(slow_stats_tracker["q1_race_time"], label="q1_race_time")
        ax.plot(slow_stats_tracker["median_race_time"], label="median_race_time")
        ax.plot(slow_stats_tracker["q3_race_time"], label="q3_race_time")
        ax.plot(slow_stats_tracker["d9_race_time"], label="d9_race_time")
        ax.plot(slow_stats_tracker["min_race_time"], label="min_race_time")
        ax.legend()
        fig.savefig(base_dir / "figures" / "race_time.png")

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["train_on_batch_duration"], label="batch_duration")
        ax.legend()
        fig.savefig(base_dir / "figures" / "train_on_batch_duration.png")

        fig, ax = plt.subplots()
        ax.plot(slow_stats_tracker["race_time_ratio"], label="race_time_ratio")
        ax.legend()
        fig.savefig(base_dir / "figures" / "race_time_ratio.png")

        # ===============================================
        #   CLEANUP
        # ===============================================

        fast_stats_tracker["race_finished"] = fast_stats_tracker["race_finished"][-400:]
        fast_stats_tracker["race_time"] = fast_stats_tracker["race_time"][-400:]
        fast_stats_tracker["q_value_starting_frame"] = fast_stats_tracker["q_value_starting_frame"][-400:]
        fast_stats_tracker["rollout_sum_rewards"] = fast_stats_tracker["rollout_sum_rewards"][-400:]
        fast_stats_tracker["loss"] = fast_stats_tracker["loss"][-400:]
        fast_stats_tracker["n_ors_light_desynchro"] = fast_stats_tracker["n_ors_light_desynchro"][-400:]
        fast_stats_tracker["n_frames_tmi_protection_triggered"].clear()
        fast_stats_tracker["train_on_batch_duration"].clear()
        fast_stats_tracker["race_time_ratio"].clear()

        time_last_save = time.time()
        torch.save(model.state_dict(), save_dir / "weights.torch")
        torch.save(model2.state_dict(), save_dir / "weights2.torch")
        torch.save(optimizer.state_dict(), save_dir / "optimizer.torch")
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
# for i in range(20):
#     plt.imshow(rollout_results["frames"][i][0, :, :], cmap='gray')
#     plt.show()
