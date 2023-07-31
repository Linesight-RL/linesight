import math
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
from torchrl.data import ReplayBuffer

from .. import misc  # TODO virer cet import
from .. import nn_utilities

class CReLU(torch.nn.Module):
    def __init__(self, inplace:bool=False):
        super(CReLU, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        x = torch.cat((x,-x),1)
        return torch.nn.functional.relu(x, inplace=self.inplace)

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
        self.iqn_embedding_dimension = iqn_embedding_dimension
        img_head_channels = [1, 16, 32, 64, 32]
        activation_function = torch.nn.LeakyReLU
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
            activation_function(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
            activation_function(inplace=True),
            torch.nn.Flatten(),
        )
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            activation_function(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            activation_function(inplace=True),
        )

        dense_input_dimension = conv_head_output_dim + float_hidden_dim

        self.A_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, n_actions),
        )
        self.V_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, 1),
        )

        self.iqn_fc = torch.nn.Linear(iqn_embedding_dimension, dense_input_dimension)
        self.lrelu = torch.nn.LeakyReLU() #Needs inplace?
        self.initialize_weights()

        self.n_actions = n_actions

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        activation_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)
        for module in [self.img_head, self.float_feature_extractor, self.A_head[:-1], self.V_head[:-1]]:
            for m in module:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    nn_utilities.init_orthogonal(m, activation_gain)
        nn_utilities.init_orthogonal(
            self.iqn_fc, np.sqrt(2) * activation_gain
        )  # Since cosine has a variance of 1/2, and we would like to exit iqn_fc with a variance of 1, we need a weight variance double that of what a normal leaky relu would need
        for module in [self.A_head[-1], self.V_head[-1]]:
            nn_utilities.init_orthogonal(module)

    def forward(
        self, img, float_inputs, num_quantiles: int, tau: Optional[torch.Tensor] = None, use_fp32: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = img.shape[0]
        img_outputs = self.img_head((img.to(torch.float32 if use_fp32 else torch.float16) - 128) / 128)  # PERF
        # img_outputs = torch.zeros(batch_size, misc.conv_head_output_dim).to(device="cuda") # Uncomment to temporarily mask the img_head
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        concat = torch.cat((img_outputs, float_outputs), 1)
        if tau is None:
            tau = torch.rand(
                size=(batch_size * num_quantiles, 1), device="cuda", dtype=torch.float32
            )  # (batch_size * num_quantiles, 1) (random numbers)
        quantile_net = torch.cos(
            torch.arange(1, self.iqn_embedding_dimension + 1, 1, device="cuda") * math.pi * tau
        )  # (batch_size*num_quantiles, 1)
        quantile_net = quantile_net.expand(
            [-1, self.iqn_embedding_dimension]
        )  # (batch_size*num_quantiles, iqn_embedding_dimension) (still random numbers)
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

        Q = V + A - A.mean(dim=-1).unsqueeze(-1)

        return Q, tau


# ==========================================================================================================================


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
        "epsilon_boltzmann",
        "gamma",
        "tau_epsilon_boltzmann",
        "tau_greedy_boltzmann",
        "IS_average",
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
        epsilon_boltzmann: float,
        gamma: float,
        tau_epsilon_boltzmann: float,
        tau_greedy_boltzmann: float,
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
        self.epsilon_boltzmann = epsilon_boltzmann
        self.gamma = gamma
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.tau_greedy_boltzmann = tau_greedy_boltzmann
        self.IS_average = deque([], maxlen=100)

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                batch, batch_info = buffer.sample(self.batch_size, return_info=True)
                (
                    state_img_tensor,
                    state_float_tensor,
                    actions,
                    rewards,
                    next_state_img_tensor,
                    next_state_float_tensor,
                    gammas_terminal,
                ) = batch
                if misc.prio_alpha > 0:
                    self.IS_average.append(batch_info["_weight"].mean())
                    IS_weights = torch.from_numpy(batch_info["_weight"] / np.mean(self.IS_average)).to("cuda", non_blocking=True)

                rewards = rewards.unsqueeze(-1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)     a,b,c,d devient a,b,c,d,a,b,c,d,a,b,c,d,...
                gammas_terminal = gammas_terminal.unsqueeze(-1).repeat([self.iqn_n, 1])# (batch_size*iqn_n, 1)
                actions = actions.unsqueeze(-1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                #
                #   Use model2 to evaluate the action chosen, per quantile.
                #
                q__stpo__model2__quantiles_tau2, tau2 = self.model2(
                    next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                )  # (batch_size*iqn_n,n_actions)
                #
                #   Use model to choose an action for next state.
                #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
                #
                if misc.use_ddqn:
                    a__tpo__model__reduced_repeated = (
                        self.model(
                            next_state_img_tensor,
                            next_state_float_tensor,
                            self.iqn_n,
                            tau=None,
                        )[0]
                        .reshape([self.iqn_n, self.batch_size, self.model.n_actions])
                        .mean(dim=0)
                        .argmax(dim=1, keepdim=True)
                        .repeat([self.iqn_n, 1])
                    )  # (iqn_n * batch_size, 1)
                    #
                    #   Build IQN target on tau2 quantiles
                    #
                    outputs_target_tau2 = rewards + gammas_terminal * q__stpo__model2__quantiles_tau2.gather(1, a__tpo__model__reduced_repeated) # (batch_size*iqn_n, 1)
                else:
                    outputs_target_tau2 = rewards + gammas_terminal * q__stpo__model2__quantiles_tau2.max(dim=1, keepdim=True)[0] # (batch_size*iqn_n, 1)

                #
                #   This is our target
                #
                outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                    0, 1
                )  # (batch_size, iqn_n, 1)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            q__st__model__quantiles_tau3, tau3 = self.model(
                state_img_tensor, state_float_tensor, self.iqn_n, tau=None
            )  # (batch_size*iqn_n,n_actions)

            outputs_tau3 = (
                q__st__model__quantiles_tau3.gather(1, actions).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
            )  # (batch_size, iqn_n, 1)

            TD_error = outputs_target_tau2[:, :, None, :] - outputs_tau3[:, None, :, :]
            # (batch_size, iqn_n, iqn_n, 1)    WTF ????????
            # Huber loss, my alternative
            loss = torch.where(
                torch.abs(TD_error) <= self.iqn_kappa,
                0.5 * TD_error**2,
                self.iqn_kappa * (torch.abs(TD_error) - 0.5 * self.iqn_kappa),
            )
            tau3 = tau3.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
            tau3 = tau3[:, None, :, :].expand([-1, self.iqn_n, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
            loss = (
                (torch.where(TD_error < 0, 1 - tau3, tau3) * loss / self.iqn_kappa).sum(dim=2).mean(dim=1)[:, 0]
            )  # pinball loss # (batch_size, )

            total_loss = torch.sum(IS_weights * loss if misc.prio_alpha > 0 else loss)

            if do_learn:
                self.scaler.scale(total_loss).backward()

                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), misc.clip_grad_norm).detach().cpu().item()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), misc.clip_grad_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = 0

            total_loss = total_loss.detach().cpu()
            if misc.prio_alpha > 0:
                buffer.update_priority(batch_info["index"], loss.detach().cpu().type(torch.float64))
        return total_loss, grad_norm

    def infer_model(self, img_inputs, float_inputs, tau=None):
        with torch.no_grad():
            state_img_tensor = torch.from_numpy(img_inputs).unsqueeze(0).to("cuda", memory_format=torch.channels_last, non_blocking=True)
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            q_values = (
                self.model(
                    state_img_tensor,
                    state_float_tensor,
                    self.iqn_k,
                    tau=tau,  # torch.linspace(0.05, 0.95, self.iqn_k, device="cuda")[:, None],
                    use_fp32=True,
                )[0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return q_values

    def get_exploration_action(self, img_inputs, float_inputs):
        q_values = self.infer_model(img_inputs, float_inputs).mean(axis=0)
        r = random.random()

        if r < self.epsilon:
            # Choose a random action
            get_argmax_on = np.random.randn(*q_values.shape)
        elif r < self.epsilon + self.epsilon_boltzmann:
            get_argmax_on = q_values + self.tau_epsilon_boltzmann * np.random.randn(*q_values.shape)
        else:
            get_argmax_on = q_values + ((self.epsilon + self.epsilon_boltzmann) > 0) * self.tau_greedy_boltzmann * np.random.randn(
                *q_values.shape
            )

        action_chosen_idx = np.argmax(get_argmax_on)
        greedy_action_idx = np.argmax(q_values)

        return (
            action_chosen_idx,
            action_chosen_idx == greedy_action_idx,
            np.max(q_values),
            q_values,
        )
