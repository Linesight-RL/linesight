import math
import random

import numpy as np
import torch

from .. import nn_utilities, noisy_linear
from ..experience_replay.experience_replay_interface import ExperienceReplayInterface


class Agent(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim,
        float_hidden_dim,
        conv_head_output_dim,
        dense_hidden_dimension,
        iqn_embedding_dimension,
        n_actions,
        float_inputs_mean,
        float_inputs_std,
    ):
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
        dense_input_dimension = conv_head_output_dim + float_hidden_dim

        self.A_head = torch.nn.Sequential(
            noisy_linear.NoisyLinear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            noisy_linear.NoisyLinear(dense_hidden_dimension // 2, n_actions),
        )
        self.V_head = torch.nn.Sequential(
            noisy_linear.NoisyLinear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            noisy_linear.NoisyLinear(dense_hidden_dimension // 2, 1),
        )
        self.iqn_fc = torch.nn.Linear(
            iqn_embedding_dimension, dense_input_dimension
        )  # There is no word in the paper on how to init this layer?
        self.lrelu = torch.nn.LeakyReLU()
        self.initialize_weights()

        self.iqn_embedding_dimension = iqn_embedding_dimension
        self.n_actions = n_actions

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                nn_utilities.init_kaiming(m)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_kaiming(m)
        # This was uninitialized in Agade's code
        nn_utilities.init_kaiming(self.iqn_fc)
        # A_head and V_head are NoisyLinear, already initialized

    def forward(self, img, float_inputs, num_quantiles, return_Q, tau=None):
        img_outputs = self.img_head((img.float() - 128) / 128)
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        concat = torch.cat((img_outputs, float_outputs), 1)
        if tau is None:
            tau = torch.cuda.FloatTensor(img.shape[0] * num_quantiles, 1).uniform_(0, 1)  # (batch_size * num_quantiles, 1) (random numbers)
        quantile_net = tau.expand(
            [-1, self.iqn_embedding_dimension]
        )  # (batch_size*num_quantiles, iqn_embedding_dimension) (still random numbers)
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device="cuda") * math.pi * quantile_net
        )  # (batch_size*num_quantiles, iqn_embedding_dimension)
        # (8 or 32 initial random numbers, expanded with cos to iqn_embedding_dimension)
        # (batch_size*num_quantiles, dense_input_dimension)
        quantile_net = self.iqn_fc(quantile_net)
        # (batch_size*num_quantiles, dense_input_dimension)
        quantile_net = self.lrelu(quantile_net)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat.repeat(num_quantiles, 1)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat * quantile_net

        A = self.A_head(concat)  # (batch_size*num_quantiles, n_actions)
        V = self.V_head(concat)  # (batch_size*num_quantiles, 1) #need to check this

        Q = V + A - A.mean(dim=-1, keepdim=True)

        return Q, tau

    def reset_noise(self):
        self.A_head[0].reset_noise()
        self.A_head[2].reset_noise()
        self.V_head[0].reset_noise()
        self.V_head[2].reset_noise()


# ==========================================================================================================================


def adjust_noise(model, is_reset):
    if is_reset:
        model.reset_noise()


class Trainer:
    __slots__ = (
        "model",
        "model2",
        "optimizer",
        "scaler",
        "batch_size",
        "iqn_k",
        "iqn_n",
        "iqn_kappa",
        "epsilon",
        "gamma",
        "n_steps",
        "AL_alpha",
    )

    def __init__(
        self,
        model: Agent,
        model2: Agent,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
        batch_size: int,
        iqn_k: int,
        iqn_n: int,
        iqn_kappa: float,
        epsilon: float,
        gamma: float,
        n_steps: int,
        AL_alpha: float,
    ):
        self.model = model
        self.model2 = model2
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.iqn_k = iqn_k
        self.iqn_n = iqn_n
        self.iqn_kappa = iqn_kappa
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_steps = n_steps
        self.AL_alpha = AL_alpha

    def train_on_batch(self, buffer: ExperienceReplayInterface):
        batch, idxs, is_weights = buffer.sample(self.batch_size)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            state_img_tensor = torch.tensor(np.array([memory.state_img for memory in batch])).to(
                "cuda", memory_format=torch.channels_last, non_blocking=True
            )
            state_float_tensor = torch.tensor(np.array([memory.state_float for memory in batch])).to("cuda", non_blocking=True)
            actions = torch.tensor(np.array([memory.action for memory in batch])).to("cuda", non_blocking=True)
            rewards = torch.tensor(np.array([memory.reward for memory in batch])).to("cuda", non_blocking=True)
            done = torch.tensor(np.array([memory.done for memory in batch])).to("cuda", non_blocking=True)  # type: ignore
            next_state_img_tensor = torch.tensor(np.array([memory.next_state_img for memory in batch])).to(
                "cuda", memory_format=torch.channels_last, non_blocking=True
            )
            next_state_float_tensor = torch.tensor(np.array([memory.next_state_float for memory in batch])).to("cuda", non_blocking=True)
            is_weights = torch.as_tensor(is_weights).to("cuda", non_blocking=True)

            with torch.no_grad():
                rewards = rewards.reshape(-1, 1).repeat(
                    [self.iqn_n, 1]
                )  # (batch_size*iqn_n, 1)     a,b,c,d devient a,b,c,d,a,b,c,d,a,b,c,d,...
                # (batch_size*iqn_n, 1)
                done = done.reshape(-1, 1).repeat([self.iqn_n, 1])
                actions = actions[:, None].repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                self.model2.reset_noise()
                outputs_target, tau_target = self.model2(next_state_img_tensor, next_state_float_tensor, self.iqn_n, True)
                # outputs_target  : (batch_size*iqn_n,n_actions)
                # tau_target : (batch_size*iqn_n, 1)
                V_next = torch.max(outputs_target, dim=1)[0]  # (batch_size*iqn_n,)
                AL_term2 = V_next - torch.gather(outputs_target, 1, actions.type(torch.int64)).squeeze(-1)  # (batch_size*iqn_n,)
                outputs_target = torch.max(outputs_target, dim=1, keepdim=True)[0]  # (batch_size*iqn_n, 1)
                AL_term2 = AL_term2.reshape([self.iqn_n, self.batch_size, 1])  # (iqn_n, batch_size, 1)
                AL_term2 = AL_term2.permute([1, 0, 2])  # (batch_size, iqn_n, 1)
                outputs_target = rewards + pow(self.gamma, self.n_steps) * outputs_target  # (batch_size*iqn_n, 1)
                outputs_target = torch.where(done, rewards, outputs_target)  # (batch_size*iqn_n, 1)s
                outputs_target = outputs_target.reshape([self.iqn_n, self.batch_size, 1])
                # (iqn_n, batch_size, 1)            outputs_target[0, :, :] contient 1 de chaque du batch d'entrée
                outputs_target = outputs_target.permute([1, 0, 2])
                # (batch_size, iqn_n, 1)            outputs_target[0, :, :] 8 copies du target correspondant au premier sample du batch
            with torch.no_grad():
                outputs_targetnet, tau = self.model2(state_img_tensor, state_float_tensor, self.iqn_n, True, tau_target)
                # outputs_targetnet  : (batch_size*iqn_n,n_actions)
                # tau                : (batch_size*iqn_n, 1)
                V = torch.max(outputs_targetnet, dim=1)[0]  # (batch_size*iqn_n,)
                # (batch_size*iqn_n,)
                AL_term = V - torch.gather(outputs_targetnet, 1, actions.type(torch.int64)).squeeze(-1)
                # (iqn_n, batch_size, 1)
                AL_term = AL_term.reshape([self.iqn_n, self.batch_size, 1])
                AL_term = AL_term.permute([1, 0, 2])  # (batch_size, iqn_n, 1)
                outputs_target -= self.AL_alpha * torch.minimum(AL_term, AL_term2)  # (batch_size, iqn_n, 1)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            self.model.reset_noise()

            outputs, _ = self.model(state_img_tensor, state_float_tensor, self.iqn_n, True)
            # outputs  : (batch_size*iqn_n,n_actions)

            mean_q_value = torch.mean(outputs, dim=0).detach().cpu()

            outputs = torch.gather(outputs, 1, actions.type(torch.int64))  # (batch_size*iqn_n, 1)

            # (iqn_n, batch_size, 1)
            outputs = outputs.reshape([self.iqn_n, self.batch_size, 1])
            outputs = outputs.permute([1, 0, 2])  # (batch_size, iqn_n, 1)

            # (batch_size, iqn_n, iqn_n, 1)    WTF ????????
            TD_Error = outputs_target[:, :, None, :] - outputs[:, None, :, :]

            # Huber loss
            loss = torch.where(
                torch.abs(TD_Error) <= self.iqn_kappa,
                0.5 * TD_Error**2,
                self.iqn_kappa * (torch.abs(TD_Error) - 0.5 * self.iqn_kappa),
            )

            tau = torch.reshape(tau, [self.iqn_n, self.batch_size, 1])
            tau = tau.permute([1, 0, 2])  # (batch_size, iqn_n, 1)
            # (batch_size, iqn_n, iqn_n, 1)
            tau = tau[:, None, :, :].expand([-1, self.iqn_n, -1, -1])
            loss = torch.where(TD_Error < 0, (tau - 1), tau) * loss / self.iqn_kappa  # pinball loss
            loss = torch.sum(loss, dim=2)  # (batch_size, iqn_n, 1)
            loss = torch.mean(loss, dim=1)  # (batch_size, 1)
            loss = loss[:, 0]  # (batch_size, )
            total_loss = torch.sum(is_weights * loss)  # total_loss.shape=torch.Size([])

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        buffer.update(idxs, loss.detach().cpu().numpy().astype(np.float32))

        return mean_q_value, total_loss.detach().cpu()

    def get_exploration_action(self, img_inputs, float_inputs):
        if self.epsilon > 0:
            # We are not evaluating
            self.model.reset_noise()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                state_img_tensor = torch.tensor(np.expand_dims(img_inputs, axis=0)).to(
                    "cuda", memory_format=torch.channels_last, non_blocking=True
                )
                state_float_tensor = torch.tensor(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
                q_values = self.model(state_img_tensor, state_float_tensor, self.iqn_k, True)[0].cpu().numpy().mean(axis=0)

        if False and random.random() < self.epsilon:
            return random.choice([1, 4, 7]), False, np.max(q_values)
        elif random.random() < self.epsilon:
            return random.randrange(0, self.model.n_actions), False, np.max(q_values)
        else:
            return np.argmax(q_values), True, np.max(q_values)
