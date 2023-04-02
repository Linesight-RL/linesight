import datetime
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
# buffer = BasicExperienceReplay(capacity=misc.memory_size)
scaler = torch.cuda.amp.GradScaler()
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

buffer = joblib.load(save_dir / "buffer.joblib")

# ========================================================
# Training loop
# ========================================================

model.train()

for i in range(20):
    train_start_time = time.time()

    if i <= 2:
        start_time_2 = time.time()
    mean_q_values, loss = trainer.train_on_batch(buffer)
    print(f"{time.time() - train_start_time:>10.3f} {time.time() - start_time_2:>10.3f} {loss:.3f}")

    # nn_utilities.soft_copy_param(model2, model, misc.soft_update_tau)
