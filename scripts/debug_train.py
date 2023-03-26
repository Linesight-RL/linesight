from pathlib import Path

import torch

from trackmania_rl import buffer_management, misc, nn_management
from trackmania_rl.agents import ddqn
from trackmania_rl import rollout
from functools import partial
import datetime

base_dir = Path(__file__).resolve().parents[1]
save_dir = base_dir / "save"

buffer = buffer_management.get_buffer()

Agent = ddqn.Agent
learn_on_batch = ddqn.learn_on_batch

# ========================================================
# Load model and optimizer weights
# ========================================================
model = Agent(misc.float_input_dim, misc.float_hidden_dim).to("cuda")
model2 = Agent(misc.float_input_dim, misc.float_hidden_dim).to("cuda")
print(model)
try:
    model.load_state_dict(torch.load(save_dir / "weights.torch"))
    model2.load_state_dict(torch.load(save_dir / "weights2.torch"))
except:
    print("Could not find weights file, left default initialization")
    # model2.load_state_dict(model.state_dict()) Why would we do that ???


optimizer = torch.optim.RAdam(model.parameters(), lr=misc.learning_rate)

try:
    optimizer.load_state_dict(torch.load(save_dir / "optimizer.torch"))
except:
    print("Could not find optimizer file, left default initialization")

scaler = torch.cuda.amp.GradScaler()

# ========================================================
# Training loop
# ========================================================
number_memories_generated = 0
number_batches_done = 0
number_target_network_updates = 0

tmi = rollout.TMInterfaceManager(
    running_speed=misc.running_speed, run_steps_per_action=misc.run_steps_per_action, max_time=misc.max_rollout_time_ms,interface_name='TMInterface1'
)


print(datetime.datetime.now())

rollout_results = tmi.rollout(
    exploration_policy=partial(ddqn.get_exploration_action, model, misc.epsilon),
)

buffer, number_memories_added = buffer_management.fill_buffer_from_rollout_with_n_steps_rule(
    buffer, rollout_results, misc.n_steps
)
number_memories_generated += number_memories_added

print(f"{number_memories_generated=}")

number_batches_done += 1
print("------- LEARN ON BATCH")
learn_on_batch(model, model2, optimizer, scaler, buffer_management.sample(buffer, misc.batch_size))

# print(f"{model.state_dict()['img_head.6.weight'][0, 0].detach().cpu()=}")
# print(f"{model2.state_dict()['img_head.6.weight'][0, 0].detach().cpu()=}")

print(f"{model.state_dict()['dense_head.0.weight'][0, 0].detach().cpu()=}")
print(f"{model2.state_dict()['dense_head.0.weight'][0, 0].detach().cpu()=}")

print(f"{model.state_dict()['dense_head.0.bias']=}")
print(f"{model2.state_dict()['dense_head.0.bias']=}")


if (
    misc.number_memories_trained_on_between_target_network_updates * number_target_network_updates
    <= number_batches_done * misc.batch_size
):
    number_target_network_updates += 1
    print("------- ------- SOFT UPDATE TARGET NETWORK")
    nn_management.soft_copy_param(model2, model, misc.soft_update_tau)
