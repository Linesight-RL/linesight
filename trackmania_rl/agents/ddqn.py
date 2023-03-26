import numpy as np
import torch
import random
from .. import misc, nn_management, buffer_management


class Agent(torch.nn.Module):
    def __init__(self, float_inputs_dim, float_hidden_dim):
        super().__init__()

        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(16, 16), stride=8),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(8, 8), stride=4),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Flatten(),
        )
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )

        dense_input_dimension = misc.conv_head_output_dim + float_hidden_dim
        self.dense_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, misc.dense_hidden_dimension),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(misc.dense_hidden_dimension, len(misc.inputs)),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                nn_management.init_kaiming(m)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                nn_management.init_kaiming(m)
        nn_management.init_kaiming(self.dense_head[0])
        nn_management.init_xavier(self.dense_head[2])

    def forward(self, img_input, float_inputs):
        img_outputs = self.img_head((img_input - 128) / 128)
        float_outputs = self.float_feature_extractor(float_inputs)
        concat = torch.cat((img_outputs, float_outputs), 1)
        Q = self.dense_head(concat)
        return Q


class DummyAgent(torch.nn.Module):
    def __init__(self, float_inputs_dim, float_hidden_dim):
        super().__init__()
        self.thingy = torch.nn.Sequential(
            torch.nn.Linear(1, len(misc.inputs)),
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn_management.init_xavier(self.thingy[0])

    def forward(self, img_input, float_inputs):
        Q = self.thingy(torch.zeros((img_input.shape[0], 1)).to("cuda"))
        return Q


def learn_on_batch(
    model: Agent,
    model2: Agent,
    optimizer,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    buffer,
):
    # batch, idxs, is_weights = buffer.sample(misc.batch_size)
    batch = buffer_management.sample(buffer, misc.batch_size)

    optimizer.zero_grad(set_to_none=True)
    # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    state_img_tensor = torch.tensor(
        np.array([memory.state_img for memory in batch]), requires_grad=True, dtype=torch.float32
    ).to("cuda", memory_format=torch.channels_last, non_blocking=True)
    state_float_tensor = torch.tensor(
        np.array([memory.state_float for memory in batch]), dtype=torch.float32, requires_grad=True
    ).to("cuda", non_blocking=True)
    actions = torch.tensor(np.array([memory.action for memory in batch]), requires_grad=False).to(
        "cuda", non_blocking=True
    )
    rewards = torch.tensor(np.array([memory.reward for memory in batch]), requires_grad=False).to(
        "cuda", non_blocking=True
    )
    done = torch.tensor(np.array([memory.done for memory in batch]), requires_grad=False).to("cuda", non_blocking=True)
    next_state_img_tensor = torch.tensor(
        np.array([memory.next_state_img for memory in batch]), requires_grad=True, dtype=torch.float32
    ).to("cuda", memory_format=torch.channels_last, non_blocking=True)
    next_state_float_tensor = torch.tensor(
        np.array([memory.next_state_float for memory in batch]), dtype=torch.float32, requires_grad=True
    ).to("cuda", non_blocking=True)
    # is_weights = torch.as_tensor(is_weights).to("cuda", non_blocking=True)

    with torch.no_grad():
        outputs_next_action = torch.argmax(model(next_state_img_tensor, next_state_float_tensor), dim=1)
        outputs_target = torch.gather(
            model2(next_state_img_tensor, next_state_float_tensor),
            dim=1,
            index=torch.unsqueeze(outputs_next_action, 1),
        ).squeeze(dim=1)

        print("Mean value according to model2", outputs_target.mean())

        outputs_target = rewards + pow(misc.gamma, misc.n_steps) * outputs_target
        outputs_target = torch.where(done, rewards, outputs_target)

    outputs = model(state_img_tensor, state_float_tensor)

    mean_q_value = torch.mean(outputs, dim=0).detach().cpu()

    outputs = torch.gather(outputs, 1, torch.unsqueeze(actions.type(torch.int64), 1)).squeeze(1)

    print("Mean targets : ", torch.mean(outputs_target))
    print("Mean output  : ", torch.mean(outputs))
    print("Delta        : ", torch.mean(outputs_target - outputs))

    loss = torch.square(outputs_target - outputs)
    total_loss = torch.sum(loss)
    # total_loss = torch.mean(is_weights * loss)
    print(total_loss)
    # END WITH AMP
    total_loss.backward()

    # Clip gradient norm, from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py#L215
    torch.nn.utils.clip_grad_value_(model.parameters(), misc.clip_grad_value)

    optimizer.step()

    # scaler.scale(total_loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    # buffer.update(idxs, loss.detach().cpu().numpy().astype(np.float32))

    return mean_q_value


def get_exploration_action(model, epsilon, img_inputs, float_inputs):
    # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    with torch.no_grad():
        state_img_tensor = torch.tensor(np.expand_dims(img_inputs, axis=0)).to(
            "cuda", memory_format=torch.channels_last, non_blocking=True
        )
        state_float_tensor = torch.tensor(
            np.expand_dims((float_inputs - misc.float_inputs_mean) / misc.float_inputs_std, axis=0),
            dtype=torch.float32,
        ).to("cuda", non_blocking=True)

        q_values = model(state_img_tensor, state_float_tensor)[0].cpu().numpy()
    # end with AMP
    if random.random() < epsilon:
        return random.randrange(0, len(misc.inputs)), False
    else:
        return np.argmax(q_values), True
