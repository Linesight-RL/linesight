import importlib
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import trackmania_rl.agents.noisy_iqn_pal2 as noisy_iqn_pal2
from trackmania_rl import buffer_management, misc, nn_utilities, tm_interface_manager
from trackmania_rl.experience_replay.basic_experience_replay import BasicExperienceReplay

base_dir = Path(__file__).resolve().parents[1]
run_name = "06"

save_dir = base_dir / "save" / run_name
save_dir.mkdir(parents=True, exist_ok=True)
tensorboard_writer = SummaryWriter(log_dir=str(base_dir / "tensorboard" / run_name))


layout = {
    "ABCDE": {
        "eval_q_values_starting_frame": ["Multiline", [f"eval_q_value_{i}_starting_frame" for i in range(len(misc.inputs))]],
        "race_time": [
            "Multiline",
            [
                "last400_min_race_time",
                "last400_d1_race_time",
                # "last400_q1_race_time",
                "last400_median_race_time",
                # "last400_q3_race_time",
                "last400_d9_race_time",
            ],
        ],
        # r"last400_%race finished": ["Multiline", [r"last400_%race finished"]],
        "loss": ["Multiline", ["laststep_mean_loss"]],
        "noisy_std": ["Multiline", [f"std_due_to_noisy_for_action{i}" for i in range(len(misc.inputs))]],
        "iqn_std": ["Multiline", [f"std_within_iqn_quantiles_for_action{i}" for i in range(len(misc.inputs))]],
        "laststep_race_time_ratio": ["Multiline", ["laststep_race_time_ratio"]],
        "observed_rollouts": [
            "Multiline",
            [
                "last400_d1_rollout_sum_rewards",
                "last400_median_rollout_sum_rewards",
                "last400_d9_rollout_sum_rewards",
            ]
            + [f"eval_q_value_{i}_starting_frame" for i in range(len(misc.inputs))],
        ],
        "race_time_with_eval": [
            "Multiline",
            [
                "last400_min_race_time",
                "eval_race_time",
                "last400_d1_race_time",
                # "last400_q1_race_time",
                "last400_median_race_time",
                # "last400_q3_race_time",
                "last400_d9_race_time",
            ],
        ],
    },
}
tensorboard_writer.add_custom_scalars(layout)

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
scaler = torch.cuda.amp.GradScaler()
buffer = BasicExperienceReplay(capacity=misc.memory_size)
fast_stats_tracker = defaultdict(list)
step_stats_history = []
# ========================================================
# Load existing stuff
# ========================================================
# noinspection PyBroadException
try:
    model1.load_state_dict(torch.load(save_dir / "weights1.torch"))
    model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
    optimizer1.load_state_dict(torch.load(save_dir / "optimizer1.torch"))
    print(" =========================     Weights loaded !     ================================")
except:
    print(" Could not load weights")

# noinspection PyBroadException
try:
    step_stats_history = joblib.load(save_dir / "step_stats_history.joblib")
    fast_stats_tracker = joblib.load(save_dir / "fast_stats_tracker.joblib")
    print(" =========================      Stats loaded !      ================================")
except:
    print(" Could not load stats")

# ========================================================
# Bring back relevant training history
# ========================================================
if len(step_stats_history) == 0:
    # No history, start from scratch
    cumul_number_frames_played = 0
    cumul_number_memories_generated = 0
    cumul_training_hours = 0
    cumul_number_batches_done = 0
    cumul_number_target_network_updates = 0
else:
    # Use previous known cumulative counters
    cumul_number_frames_played = step_stats_history[-1]["cumul_number_frames_played"]
    cumul_number_memories_generated = step_stats_history[-1]["cumul_number_memories_generated"]
    cumul_training_hours = step_stats_history[-1]["cumul_training_hours"]
    cumul_number_batches_done = step_stats_history[-1]["cumul_number_batches_done"]
    cumul_number_target_network_updates = step_stats_history[-1]["cumul_number_target_network_updates"]

    # cumul_number_batches_done = (misc.number_times_single_memory_is_used_before_discard * (cumul_number_memories_generated - misc.virtual_memory_size_start_learn)) // misc.batch_size
    # cumul_number_target_network_updates =  (cumul_number_batches_done * misc.batch_size) //  misc.number_memories_trained_on_between_target_network_updates

number_frames_played = 0
number_memories_generated = 0

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
    AL_alpha=misc.AL_alpha,
)

# ========================================================
# Training loop
# ========================================================
model1.train()
model1.V_head.eval()
model2.V_head.eval()
time_next_save = time.time() + misc.statistics_save_period_seconds
tmi = tm_interface_manager.TMInterfaceManager(
    base_dir=base_dir,
    running_speed=misc.running_speed,
    run_steps_per_action=misc.run_steps_per_action,
    max_time=misc.max_rollout_time_ms,
    interface_name="TMInterface0",
)

# iprofile = 0

while True:

    # iprofile += 1
    # if iprofile == 50:
    #     import sys
    #     sys.exit()

    # ===============================================
    #   PLAY ONE ROUND
    # ===============================================
    rollout_start_time = time.time()
    trainer.epsilon = (
        misc.high_exploration_ratio * misc.epsilon
        if cumul_number_memories_generated < misc.number_memories_generated_high_exploration
        else misc.epsilon
    )
    trainer.gamma = misc.gamma
    trainer.AL_alpha = misc.AL_alpha
    for param_group in optimizer1.param_groups:
        param_group['lr'] = misc.learning_rate
    rollout_results = tmi.rollout(
        exploration_policy=trainer.get_exploration_action,
        stats_tracker=fast_stats_tracker,
    )
    number_frames_played += len(rollout_results["frames"])
    cumul_number_frames_played += len(rollout_results["frames"])
    fast_stats_tracker["race_time_ratio"].append(fast_stats_tracker["race_time"][-1] / ((time.time() - rollout_start_time) * 1000))
    print("race time ratio  ", np.median(np.array(fast_stats_tracker["race_time_ratio"])))
    buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
        buffer, rollout_results, misc.n_steps, misc.gamma, misc.discard_non_greedy_actions_in_nsteps
    )
    number_memories_generated += number_memories_added
    cumul_number_memories_generated += number_memories_added
    print(f" NMG={cumul_number_memories_generated:<8}")

    # ===============================================
    #   LEARN ON BATCH
    # ===============================================
    while (
        cumul_number_memories_generated >= misc.memory_size_start_learn
        and cumul_number_batches_done * misc.batch_size
        <= misc.number_times_single_memory_is_used_before_discard * (cumul_number_memories_generated - misc.virtual_memory_size_start_learn)
    ):
        train_start_time = time.time()
        mean_q_values, loss = trainer.train_on_batch(buffer)
        fast_stats_tracker["train_on_batch_duration"].append(time.time() - train_start_time)
        fast_stats_tracker["loss"].append(loss)
        cumul_number_batches_done += 1
        print(f"B    {loss=:<8.2e}")

        # ===============================================
        #   UPDATE TARGET NETWORK
        # ===============================================
        if (
            misc.number_memories_trained_on_between_target_network_updates * cumul_number_target_network_updates
            <= cumul_number_batches_done * misc.batch_size
        ):
            cumul_number_target_network_updates += 1
            print("UPDATE")
            nn_utilities.soft_copy_param(model2, model1, misc.soft_update_tau)
            # model2.load_state_dict(model.state_dict())
    print("")

    # ===============================================
    #   STATISTICS EVERY NOW AND THEN
    # ===============================================
    if time.time() > time_next_save:  # every 15 minutes
        cumul_training_hours += misc.statistics_save_period_seconds / 3600

        # ===============================================
        #   FILL STEPS STATS HISTORY
        # ===============================================
        step_stats = {
            "number_frames_played": number_frames_played,
            "number_memories_generated": number_memories_generated,
            "training_hours": misc.statistics_save_period_seconds / 3600,
            "cumul_number_frames_played": cumul_number_frames_played,
            "cumul_number_memories_generated": cumul_number_memories_generated,
            "cumul_training_hours": cumul_training_hours,
            "cumul_number_batches_done": cumul_number_batches_done,
            "cumul_number_target_network_updates": cumul_number_target_network_updates,
            "gamma": misc.gamma,
            "n_steps": misc.n_steps,
            "epsilon": trainer.epsilon,
            "AL_alpha": trainer.AL_alpha,
            "learning_rate": misc.learning_rate,
            "discard_non_greedy_actions_in_nsteps": misc.discard_non_greedy_actions_in_nsteps,
            "reward_bogus_velocity": misc.reward_bogus_velocity,
            "reward_bogus_gas": misc.reward_bogus_gas,
            "reward_bogus_low_speed": misc.reward_bogus_low_speed,
            #
            r"last400_%race finished": np.array(fast_stats_tracker["race_finished"][-400:]).mean(),
            r"last400_%light_desynchro": np.array(fast_stats_tracker["n_ors_light_desynchro"][-400:]).sum()
            / (np.array(fast_stats_tracker["race_time"][-400:]).sum() / (misc.ms_per_run_step * misc.run_steps_per_action)),
            r"last400_%consecutive_frames_equal": np.array(fast_stats_tracker["n_two_consecutive_frames_equal"][-400:]).sum()
            / (np.array(fast_stats_tracker["race_time"][-400:]).sum() / (misc.ms_per_run_step * misc.run_steps_per_action)),
            #
            "laststep_mean_loss": np.array(fast_stats_tracker["loss"]).mean(),
            "laststep_n_tmi_protection": np.array(fast_stats_tracker["n_frames_tmi_protection_triggered"]).sum(),
            "laststep_race_time_ratio": np.median(np.array(fast_stats_tracker["race_time_ratio"])),
            "laststep_train_on_batch_duration": np.median(np.array(fast_stats_tracker["train_on_batch_duration"])),
            #
            "last400_min_race_time": np.array(fast_stats_tracker["race_time"][-400:]).min(initial=None),
            "last400_d1_race_time": np.quantile(np.array(fast_stats_tracker["race_time"][-400:]), 0.1),
            "last400_q1_race_time": np.quantile(np.array(fast_stats_tracker["race_time"][-400:]), 0.25),
            "last400_median_race_time": np.quantile(np.array(fast_stats_tracker["race_time"][-400:]), 0.5),
            "last400_q3_race_time": np.quantile(np.array(fast_stats_tracker["race_time"][-400:]), 0.75),
            "last400_d9_race_time": np.quantile(np.array(fast_stats_tracker["race_time"][-400:]), 0.9),
            #
            "last400_d1_value_starting_frame": np.quantile(np.array(fast_stats_tracker["value_starting_frame"][-400:]), 0.1),
            "last400_q1_value_starting_frame": np.quantile(np.array(fast_stats_tracker["value_starting_frame"][-400:]), 0.25),
            "last400_median_value_starting_frame": np.quantile(np.array(fast_stats_tracker["value_starting_frame"][-400:]), 0.5),
            "last400_q3_value_starting_frame": np.quantile(np.array(fast_stats_tracker["value_starting_frame"][-400:]), 0.75),
            "last400_d9_value_starting_frame": np.quantile(np.array(fast_stats_tracker["value_starting_frame"][-400:]), 0.9),
            #
            "last400_d1_rollout_sum_rewards": np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"][-400:]), 0.1),
            "last400_q1_rollout_sum_rewards": np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"][-400:]), 0.25),
            "last400_median_rollout_sum_rewards": np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"][-400:]), 0.5),
            "last400_q3_rollout_sum_rewards": np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"][-400:]), 0.75),
            "last400_d9_rollout_sum_rewards": np.quantile(np.array(fast_stats_tracker["rollout_sum_rewards"][-400:]), 0.9),
            #
        }

        for i in range(len(misc.inputs)):
            step_stats[f"last400_q_value_{i}_starting_frame"] = np.mean(fast_stats_tracker[f"q_value_{i}_starting_frame"][-400:])

        # TODO : add more recent loss than last400, that's too slow

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
        number_frames_played += len(rollout_results["frames"])
        cumul_number_frames_played += len(rollout_results["frames"])
        trainer.epsilon = misc.epsilon
        model1.train()
        model1.V_head.eval()
        buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
            buffer, rollout_results, misc.n_steps, misc.gamma, misc.discard_non_greedy_actions_in_nsteps
        )
        number_memories_generated += number_memories_added
        cumul_number_memories_generated += number_memories_added
        step_stats["eval_race_time"] = eval_stats_tracker["race_time"][-1]
        for i in range(len(misc.inputs)):
            step_stats[f"eval_q_value_{i}_starting_frame"] = eval_stats_tracker[f"q_value_{i}_starting_frame"][-1]

        print("EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL EVAL")

        # ===============================================
        #   SPREAD
        # ===============================================

        # Faire 100 tirages avec noisy, et les tau de IQN fixés
        tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")
        state_img_tensor = torch.as_tensor(np.expand_dims(rollout_results["frames"][0], axis=0)).to(
            "cuda", memory_format=torch.channels_last, non_blocking=True
        )
        state_float_tensor = torch.as_tensor(np.expand_dims(rollout_results["floats"][0], axis=0)).to("cuda", non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                list_of_q_values = []
                for i in range(400):
                    model1.reset_noise()
                    list_of_q_values.append(model1(state_img_tensor, state_float_tensor, misc.iqn_k, tau=tau)[0].cpu().numpy().mean(axis=0))

        for i, std in enumerate(list(np.array(list_of_q_values).astype(np.float32).std(axis=0))):
            step_stats[f"std_due_to_noisy_for_action{i}"] = std

        # Désactiver noisy, tirer des tau équitablement répartis
        model1.eval()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                per_quantile_output = model1(state_img_tensor, state_float_tensor, misc.iqn_k, tau=tau)[0]

        for i, std in enumerate(list(per_quantile_output.cpu().numpy().astype(np.float32).std(axis=0))):
            step_stats[f"std_within_iqn_quantiles_for_action{i}"] = std
        model1.train()
        model1.V_head.eval()

        # tensorboard_writer.add_scalars(
        #     main_tag="",
        #     tag_scalar_dict=step_stats,
        #     global_step=step_stats["cumul_number_memories_generated"],
        #     walltime=float(step_stats["cumul_training_hours"] * 3600),
        # )
        for k, v in step_stats.items():
            tensorboard_writer.add_scalar(
                tag=k,
                scalar_value=v,
                global_step=step_stats["cumul_number_memories_generated"],
                walltime=float(step_stats["cumul_training_hours"] * 3600),
            )

        tensorboard_writer.add_text(
            "times_summary",
            f"min {step_stats['last400_min_race_time']/1000:.2f} ; d1 {step_stats['last400_d1_race_time']/1000:.2f} ; median {step_stats['last400_median_race_time']/1000:.2f} ; d9 {step_stats['last400_d9_race_time']/1000:.2f} ",
            global_step=step_stats["cumul_number_memories_generated"],
            walltime=float(step_stats["cumul_training_hours"] * 3600),
        )
        step_stats_history.append(step_stats)

        # ===============================================
        #   Buffer stats
        # ===============================================

        print("Mean in buffer", np.array([experience.state_float for experience in buffer.buffer]).mean(axis=0))
        print("Std in buffer ", np.array([experience.state_float for experience in buffer.buffer]).std(axis=0))

        # ===============================================
        #   CLEANUP
        # ===============================================
        fast_stats_tracker["n_frames_tmi_protection_triggered"].clear()
        fast_stats_tracker["train_on_batch_duration"].clear()
        fast_stats_tracker["race_time_ratio"].clear()
        fast_stats_tracker["loss"].clear()

        for key, value in fast_stats_tracker.items():
            print(f"{len(value)} : {key}")  # FIXME
            fast_stats_tracker[key] = value[-400:]

        number_memories_generated = 0
        number_frames_played = 0

        # ===============================================
        #   SAVE
        # ===============================================
        torch.save(model1.state_dict(), save_dir / "weights1.torch")
        torch.save(model2.state_dict(), save_dir / "weights2.torch")
        torch.save(optimizer1.state_dict(), save_dir / "optimizer1.torch")
        joblib.dump(step_stats_history, save_dir / "step_stats_history.joblib")
        joblib.dump(fast_stats_tracker, save_dir / "fast_stats_tracker.joblib")

        # ===============================================
        #   RELOAD
        # ===============================================
        importlib.reload(misc)

        time_next_save += misc.statistics_save_period_seconds

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
# for i in range(40):
#     plt.imshow(rollout_results["frames"][i][0, :, :], cmap="gray")
#     plt.gcf().suptitle(f"{(i * 5) // 100} {(i * 5) % 100}")
#     plt.show()
