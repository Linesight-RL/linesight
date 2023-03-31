import collections
import datetime
import random
import time
import weakref
from functools import partial
from pathlib import Path

import dxcam
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

import trackmania_rl.agents.noisy_iqn as learning_algorithm
from trackmania_rl.experience_replay.prioritized_experience_replay import PrioritizedExperienceReplay
from trackmania_rl import buffer_management, misc, nn_utilities, rollout

base_dir = Path(__file__).resolve().parents[1]
save_dir = base_dir / "save"


Agent = learning_algorithm.Agent
learn_on_batch = learning_algorithm.learn_on_batch

torch.cuda.manual_seed_all(43)
torch.manual_seed(43)
random.seed(43)

plt.style.use("seaborn")

# ========================================================
# Create new stuff
# ========================================================
model = Agent(misc.float_input_dim, misc.float_hidden_dim).to("cuda")
model2 = Agent(misc.float_input_dim, misc.float_hidden_dim).to("cuda")
print(model)
optimizer = torch.optim.RAdam(model.parameters(), lr=misc.learning_rate)
buffer = PrioritizedExperienceReplay(
        capacity=misc.memory_size,
        sample_with_segments=misc.prio_sample_with_segments,
        prio_alpha=misc.prio_alpha,
        prio_beta=misc.prio_beta,
        prio_epsilon=misc.prio_epsilon,
    )
scaler = torch.cuda.amp.GradScaler()

# ========================================================
# Load existing stuff
# ========================================================
try:
    model.load_state_dict(torch.load(save_dir / "weights.torch"))
    model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
    print(" =========================     Weights loaded !      ================================")
except:
    print(" ========     Could not find weights file, left default initialization    ===========")
    # model2.load_state_dict(model.state_dict()) Why would we do that ???


# optimizer = torch.optim.SGD(model.parameters(), lr=misc.learning_rate, momentum=0.2)

# try:
#     # optimizer.load_state_dict(torch.load(save_dir / "optimizer.torch"))
# except:
#     print("Could not find optimizer file, left default initialization")

# buffer = joblib.load(save_dir / "buffer.joblib")

stats_tracker = {
    "race_finished": collections.deque(maxlen=100),
    "race_time": collections.deque(maxlen=100),
    "q_value_starting_frame": collections.deque(maxlen=100),
    "rollout_sum_rewards": collections.deque(maxlen=100),
    "loss": collections.deque(maxlen=200),
    "n_ors_light_desynchro": [],
    "n_frames_tmi_protection_triggered": [],
    "n_frames": [],
}
# number of frames total, number of frames where light desynchro, number of frames where heavy desynchro, number of frames where camera.grab did not have a new frame

stats_tracker_eval = {
    "race_finished": collections.deque(maxlen=1),
    "race_time": [],
    "q_value_starting_frame": collections.deque(maxlen=1),
    "rollout_sum_rewards": collections.deque(maxlen=1),
    "n_ors_light_desynchro": collections.deque(maxlen=1),
    "n_frames_tmi_protection_triggered": collections.deque(maxlen=1),
    "n_frames": collections.deque(maxlen=1),
}

slow_stats_tracker = {
    r"%race finished": [],
    "avg_race_time": [],
    "min_race_time": [],
    "median_race_time": [],
    "avg_q_value_starting_frame": [],
    "avg_rollout_sum_rewards": [],
    "avg_delta_q_starting_rollout_sum_rewards": [],
    "loss": [],
    r"%light_desynchro": [],
    r"%tmi_protection": [],
}


# ========================================================
# Training loop
# ========================================================
number_memories_generated = 0
number_batches_done = 0
number_target_network_updates = 0

lossbuffer = collections.deque(maxlen=100)

zoubidou = []
average_loss_every_5mn = []
# np.array([a.detach().cpu().numpy().ravel() for a in zoubidou]).shape


# %%

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
    print(datetime.datetime.now())

    rollout_results = tmi.rollout(
        exploration_policy=partial(
            learning_algorithm.get_exploration_action,
            model,
            misc.epsilon * (7 if buffer.tree.n_entries < misc.memory_size // 2 else 1),
        ),
        stats_tracker=stats_tracker,
    )
    # print("============================ ",max(rollout_results["rewards"]))
    buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
        buffer, rollout_results, misc.n_steps
    )
    number_memories_generated += number_memories_added

    print(f"{number_memories_generated=}")

    while (
        number_batches_done * misc.batch_size
        <= misc.number_times_single_memory_is_used_before_discard * number_memories_generated
    ):
        number_batches_done += 1

        if buffer.tree.n_entries > misc.memory_size // 2:
            mean_q_values, loss = learn_on_batch(model, model2, optimizer, scaler, buffer)
            stats_tracker["loss"].append(loss)
            # lossbuffer.append(loss)
            # zoubidou.append(mean_q_values)

        if (
            misc.number_memories_trained_on_between_target_network_updates * number_target_network_updates
            <= number_batches_done * misc.batch_size
        ):
            number_target_network_updates += 1
            # print("------- ------- SOFT UPDATE TARGET NETWORK")
            if buffer.tree.n_entries > misc.memory_size // 2:
                nn_utilities.soft_copy_param(model2, model, misc.soft_update_tau)
                # model2.load_state_dict(model.state_dict())

    if time.time() > time_last_save + 60 * 20:  # every 20 minutes
        slow_stats_tracker[r"%race finished"].append(np.array(stats_tracker["race_finished"]).mean())
        slow_stats_tracker["avg_race_time"].append(np.array(stats_tracker["race_time"]).mean())
        slow_stats_tracker["min_race_time"].append(np.array(stats_tracker["race_time"]).min())
        slow_stats_tracker["median_race_time"].append(np.median(np.array(stats_tracker["race_time"])))
        slow_stats_tracker["avg_q_value_starting_frame"].append(
            np.array(stats_tracker["q_value_starting_frame"]).mean()
        )
        slow_stats_tracker["avg_rollout_sum_rewards"].append(np.array(stats_tracker["rollout_sum_rewards"]).mean())
        slow_stats_tracker["avg_delta_q_starting_rollout_sum_rewards"].append(
            slow_stats_tracker["avg_q_value_starting_frame"][-1] - slow_stats_tracker["avg_rollout_sum_rewards"][-1]
        )
        slow_stats_tracker["loss"].append(np.array(stats_tracker["loss"]).mean())
        slow_stats_tracker[r"%light_desynchro"].append(np.array(stats_tracker["n_ors_light_desynchro"]).sum())
        slow_stats_tracker[r"%tmi_protection"].append(
            np.array(stats_tracker["n_frames_tmi_protection_triggered"]).sum()
        )

        stats_tracker["n_ors_light_desynchro"].clear()
        stats_tracker["n_frames_tmi_protection_triggered"].clear()
        stats_tracker["n_frames"].clear()

        plt.plot(slow_stats_tracker[r"%light_desynchro"], label=r"%light_desynchro")
        plt.legend()
        plt.show()
        plt.plot(slow_stats_tracker[r"%tmi_protection"], label=r"%tmi_protection")
        plt.legend()
        plt.show()
        plt.plot(slow_stats_tracker[r"%race finished"], label=r"%race finished")
        plt.legend()
        plt.show()

        plt.plot(slow_stats_tracker["avg_q_value_starting_frame"], label="avg_q_value_starting_frame")
        plt.plot(slow_stats_tracker["avg_rollout_sum_rewards"], label="avg_rollout_sum_rewards")
        plt.legend()
        plt.show()

        plt.plot(
            slow_stats_tracker["avg_delta_q_starting_rollout_sum_rewards"],
            label="avg_delta_q_starting_rollout_sum_rewards",
        )
        plt.legend()
        plt.show()

        plt.plot(
            slow_stats_tracker["loss"],
            label="loss",
        )
        plt.legend()
        plt.show()

        # Eval
        model.reset_noise()
        model.eval()
        rollout_results = tmi.rollout(
            exploration_policy=partial(learning_algorithm.get_exploration_action, model, -1),
            stats_tracker=stats_tracker_eval,
        )
        model.train()
        buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
            buffer, rollout_results, misc.n_steps
        )
        number_memories_generated += number_memories_added

        plt.plot(
            stats_tracker_eval["race_time"],
            label="eval_race_time",
        )
        plt.plot(slow_stats_tracker["avg_race_time"], label="avg_race_time")
        plt.plot(slow_stats_tracker["min_race_time"], label="min_race_time")
        plt.plot(slow_stats_tracker["median_race_time"], label="median_race_time")
        plt.legend()
        plt.show()

        # print("Average loss     every 5 minutes : ", np.array(lossbuffer).mean())
        # print(
        #     "Average Q values every 5 minutes : ",
        #     np.stack([np.array(zoubidoui) for zoubidoui in zoubidou[-200:]]).mean(axis=0),
        # )
        # print(
        #     "Overall avg Q    every 5 minutes : ",
        #     np.stack([np.array(zoubidoui) for zoubidoui in zoubidou[-200:]]).mean(),
        # )
        # average_loss_every_5mn.append(np.array(lossbuffer).mean())
        # print("SAVING MODEL AND OPTIMIZER")
        time_last_save = time.time()
        torch.save(model.state_dict(), save_dir / "weights.torch")
        torch.save(model2.state_dict(), save_dir / "weights2.torch")
        torch.save(optimizer.state_dict(), save_dir / "optimizer.torch")
        joblib.dump(slow_stats_tracker, save_dir / "slow_stats_tracker.joblib")
        joblib.dump(stats_tracker_eval, save_dir / "stats_tracker_eval.joblib")

    # if time.time() > time_last_buffer_save + 60 * 60 * 6:  # every 2 hours
    #     print("SAVING MODEL AND OPTIMIZER")
    #     time_last_buffer_save = time.time()
    #     joblib.dump(buffer, save_dir / "buffer.joblib")

# %%

model.reset_noise()

dxcam.__factory._camera_instances = weakref.WeakValueDictionary()
tmi = rollout.TMInterfaceManager(
    running_speed=misc.running_speed, run_steps_per_action=misc.run_steps_per_action, max_time=misc.max_rollout_time_ms
)
model.eval()
rollout_results = tmi.rollout(
    exploration_policy=partial(learning_algorithm.get_exploration_action, model, 0),
)
model.train()



#%%
for i in range(20):
    plt.imshow(rollout_results["frames"][i][0,:,:], cmap='gray')
    plt.show()