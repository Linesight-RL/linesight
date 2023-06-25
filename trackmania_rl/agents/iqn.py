import math
import random
from typing import Optional, Tuple

import numpy as np
import torch

from trackmania_rl.experience_replay.basic_experience_replay import ReplayBuffer

from .. import misc  # TODO virer cet import
from .. import nn_utilities


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
        img_head_channels = [1,16,32,64,32]
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
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
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, n_actions),
        )
        self.V_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, 1),
        )

        self.iqn_fc = torch.nn.Linear(
            iqn_embedding_dimension, dense_input_dimension
        )  # There is no word in the paper on how to init this layer?
        self.lrelu = torch.nn.LeakyReLU()
        self.initialize_weights()

        self.n_actions = n_actions

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                nn_utilities.init_kaiming(m,lrelu_neg_slope)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_kaiming(m,lrelu_neg_slope)
        nn_utilities.init_normal(self.iqn_fc,0,np.sqrt(2)*torch.nn.init.calculate_gain('leaky_relu', lrelu_neg_slope)/np.sqrt(self.iqn_embedding_dimension)) #Since cosine has a variance of 1/2, and we would like to exit iqn_fc with a variance of 1, we need a weight variance double that of what a normal leaky relu would need
        for m in self.A_head[:-1]:
            if isinstance(m, torch.nn.Linear):
                 nn_utilities.init_kaiming(m,lrelu_neg_slope)
        nn_utilities.init_xavier(self.A_head[-1])
        for m in self.V_head[:-1]:
            if isinstance(m, torch.nn.Linear):
                 nn_utilities.init_kaiming(m,lrelu_neg_slope)
        nn_utilities.init_xavier(self.V_head[-1])

    def forward(
        self, img, float_inputs, num_quantiles: int, tau: Optional[torch.Tensor] = None, use_fp32: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = img.shape[0]
        img_outputs = self.img_head((img.to(torch.float32 if use_fp32 else torch.float16) - 128) / 128)  # PERF
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        concat = torch.cat((img_outputs, float_outputs), 1)
        if tau is None:
            tau = torch.rand(
                size=(batch_size * num_quantiles, 1), device="cuda", dtype=torch.float32
            )  # (batch_size * num_quantiles, 1) (random numbers)
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
        "AL_alpha",
        "tau_epsilon_boltzmann",
        "tau_greedy_boltzmann",
        "execution_stream",
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
        AL_alpha: float,
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
        self.AL_alpha = AL_alpha
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.tau_greedy_boltzmann = tau_greedy_boltzmann
        self.execution_stream = torch.cuda.Stream()

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                (
                    state_img_tensor,
                    state_float_tensor,
                    new_actions,
                    new_n_steps,
                    rewards_per_n_steps,
                    next_state_img_tensor,
                    next_state_float_tensor,
                    gammas_per_n_steps,
                    minirace_min_time_actions,
                ) = buffer.sample(self.batch_size)
        with torch.cuda.stream(self.execution_stream):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    new_actions = new_actions.to(dtype=torch.int64)
                    new_n_steps = new_n_steps.to(dtype=torch.int64)
                    minirace_min_time_actions = minirace_min_time_actions.to(dtype=torch.int64)

                    new_xxx = (
                        torch.rand(size=minirace_min_time_actions.shape).to(device="cuda")
                        * (misc.temporal_mini_race_duration_actions - minirace_min_time_actions)
                    ).to(dtype=torch.int64, device="cuda")
                    temporal_mini_race_current_time_actions = misc.temporal_mini_race_duration_actions - 1 - new_xxx
                    temporal_mini_race_next_time_actions = temporal_mini_race_current_time_actions + new_n_steps

                    state_float_tensor[:, 0] = temporal_mini_race_current_time_actions
                    next_state_float_tensor[:, 0] = temporal_mini_race_next_time_actions

                    new_done = temporal_mini_race_next_time_actions >= misc.temporal_mini_race_duration_actions
                    possibly_reduced_n_steps = (
                        new_n_steps - (temporal_mini_race_next_time_actions - misc.temporal_mini_race_duration_actions).clip(min=0)
                    ).to(dtype=torch.int64)

                    rewards = rewards_per_n_steps.gather(1, (possibly_reduced_n_steps - 1).unsqueeze(-1)).repeat(
                        [self.iqn_n, 1]
                    )  # (batch_size*iqn_n, 1)     a,b,c,d devient a,b,c,d,a,b,c,d,a,b,c,d,...
                    # (batch_size*iqn_n, 1)
                    gammas_pow_nsteps = gammas_per_n_steps.gather(1, (possibly_reduced_n_steps - 1).unsqueeze(-1)).repeat([self.iqn_n, 1])
                    done = new_done.reshape(-1, 1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                    actions = new_actions[:, None]  # (batch_size, 1)
                    actions_n = actions.repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                    #
                    #   Use model to choose an action for next state.
                    #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
                    #
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
                    #   Use model2 to evaluate the action chosen, per quantile.
                    #
                    q__stpo__model2__quantiles_tau2, tau2 = self.model2(
                        next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                    )  # (batch_size*iqn_n,n_actions)

                    #
                    #   Build IQN target on tau2 quantiles
                    #
                    outputs_target_tau2 = torch.where(
                        done,
                        rewards,
                        rewards + gammas_pow_nsteps * q__stpo__model2__quantiles_tau2.gather(1, a__tpo__model__reduced_repeated),
                    )  # (batch_size*iqn_n, 1)

                    # # =============== BEG PAL ==============
                    # # V(x') dans PAL devient une distribution en IQN, on veux les quantiles de cette distribution. Il faut choisir une action, la meilleure action, et prendre les quantiles de cette action: notre V
                    # # Parceque si tu max par quantile, tu melange des quantiles de distributions differentes
                    # #
                    # #   PAL Term
                    # #
                    # pal_term_tau2 = q__stpo__model2__quantiles_tau2.gather(
                    #     1,
                    #     q__stpo__model2__quantiles_tau2.reshape(
                    #         [self.iqn_n, self.batch_size, self.model.n_actions]
                    #     )
                    #     .mean(dim=0)
                    #     .argmax(dim=1, keepdim=True)
                    #     .repeat([self.iqn_n, 1]),
                    # ) - q__stpo__model2__quantiles_tau2.gather(1, actions_n)
                    # # (batch_size*iqn_n, 1)
                    #
                    # #
                    # #   AL Term
                    # #
                    # q__st__model2__quantiles_tau2, tau2 = self.model2(
                    #     state_img_tensor, state_float_tensor, self.iqn_n, tau=tau2
                    # )  # (batch_size*iqn_n,n_actions)
                    # al_term_tau2 = q__st__model2__quantiles_tau2.gather(
                    #     1,
                    #     q__st__model2__quantiles_tau2.reshape(
                    #         [self.iqn_n, self.batch_size, self.model.n_actions]
                    #     )
                    #     .mean(dim=0)
                    #     .argmax(dim=1, keepdim=True)
                    #     .repeat([self.iqn_n, 1]),
                    # ) - q__st__model2__quantiles_tau2.gather(1, actions_n)
                    # # (batch_size*iqn_n, 1)
                    #
                    # outputs_target_tau2 -= self.AL_alpha * torch.minimum(
                    #     al_term_tau2, pal_term_tau2
                    # )
                    # # =============== END PAL ==============

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
                    q__st__model__quantiles_tau3.gather(1, actions_n).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
                )  # (batch_size, iqn_n, 1)

                TD_Error = outputs_target_tau2[:, :, None, :] - outputs_tau3[:, None, :, :]
                # (batch_size, iqn_n, iqn_n, 1)    WTF ????????
                # Huber loss, my alternative
                loss = torch.where(
                    torch.abs(TD_Error) <= self.iqn_kappa,
                    0.5 * TD_Error**2,
                    self.iqn_kappa * (torch.abs(TD_Error) - 0.5 * self.iqn_kappa),
                )
                tau3 = tau3.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
                tau3 = tau3[:, None, :, :].expand([-1, self.iqn_n, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
                loss = (
                    (torch.where(TD_Error < 0, 1 - tau3, tau3) * loss / self.iqn_kappa).sum(dim=2).mean(dim=1)[:, 0]
                )  # pinball loss # (batch_size, )

                total_loss = torch.sum(loss)  # total_loss.shape=torch.Size([])

            if do_learn:
                self.scaler.scale(total_loss).backward()

                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss = total_loss.detach().cpu()
        self.execution_stream.synchronize()
        return total_loss

    def get_exploration_action(self, img_inputs, float_inputs):
        with torch.cuda.stream(self.execution_stream):
            with torch.no_grad():
                state_img_tensor = img_inputs.unsqueeze(0).to("cuda", memory_format=torch.channels_last, non_blocking=True)
                state_float_tensor = torch.as_tensor(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
                q_values = (
                    self.model(
                        state_img_tensor,
                        state_float_tensor,
                        self.iqn_k,
                        tau=None,  # torch.linspace(0.05, 0.95, self.iqn_k, device="cuda")[:, None],
                        use_fp32=True,
                    )[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .mean(axis=0)
                )
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
