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
import trackmania_rl.agents.noisy_iqn_pal2 as noisy_iqn_pal2
from trackmania_rl import buffer_management, misc, nn_utilities, tm_interface_manager
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
# buffer = BasicExperienceReplay(capacity=misc.memory_size)
buffer = joblib.load(save_dir / "buffer.joblib")
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

print("Noisy STD : ", model1.A_head[0].std_init)


# ===============================================
#   LEARN ON BATCH
# ===============================================
train_start_time = time.time()
for i in range(100):
    mean_q_values, loss = trainer.train_on_batch(buffer)
    number_batches_done += 1
    # print("B ", end="")
    print(f"B {loss=:<12} {mean_q_values=}")

print("TIME ", time.time() - train_start_time)
