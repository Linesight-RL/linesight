import numpy as np
import torch
import random
from .. import misc, nn_management


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
        img_input = (img_input - 128) / 128
        img_outputs = self.img_head(img_input)
        float_outputs = self.float_feature_extractor(float_inputs)
        concat = torch.cat((img_outputs, float_outputs), 1)
        Q = self.dense_head(concat)
        return Q


def learn_on_batch(
    model: Agent,
    model2: Agent,
    optimizer,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    batch,
):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
        done = torch.tensor(np.array([memory.done for memory in batch]), requires_grad=False).to(
            "cuda", non_blocking=True
        )
        next_state_img_tensor = torch.tensor(
            np.array([memory.next_state_img for memory in batch]), requires_grad=True, dtype=torch.float32
        ).to("cuda", memory_format=torch.channels_last, non_blocking=True)
        next_state_float_tensor = torch.tensor(
            np.array([memory.next_state_float for memory in batch]), dtype=torch.float32, requires_grad=True
        ).to("cuda", non_blocking=True)

        with torch.no_grad():
            outputs_next_action = torch.argmax(model(next_state_img_tensor, next_state_float_tensor), dim=1)
            outputs_target = torch.gather(
                model2(next_state_img_tensor, next_state_float_tensor),
                dim=1,
                index=torch.unsqueeze(outputs_next_action, 1),
            ).squeeze(dim=1)
            outputs_target = rewards + pow(misc.gamma, misc.n_steps) * outputs_target
            outputs_target = torch.where(done, rewards, outputs_target)

        outputs = model(state_img_tensor, state_float_tensor)
        outputs = torch.gather(outputs, 1, torch.unsqueeze(actions.type(torch.int64), 1)).squeeze(1)
        loss = torch.square(outputs_target - outputs)
        total_loss = torch.sum(loss)
        print(total_loss)

    total_loss.backward()
    optimizer.step()

    # scaler.scale(total_loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    return np.sqrt(total_loss.detach().cpu() / len(batch))


def get_exploration_action(model, epsilon, img_inputs, float_inputs):
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            state_img_tensor = torch.tensor(np.expand_dims(img_inputs, axis=0)).to(
                "cuda", memory_format=torch.channels_last, non_blocking=True
            )
            state_float_tensor = torch.tensor(
                np.expand_dims((float_inputs - misc.float_inputs_mean) / misc.float_inputs_std, axis=0),
                dtype=torch.float32,
            ).to("cuda", non_blocking=True)

            q_values = model(state_img_tensor, state_float_tensor)[0].cpu().numpy()

    if random.random() < epsilon:
        return random.randrange(0, len(misc.inputs)), False
    else:
        return np.argmax(q_values), True
