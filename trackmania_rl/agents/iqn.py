import math
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer

from .. import misc  # TODO virer cet import
from .. import nn_utilities


class FeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim,
        float_hidden_dim,
        float_inputs_mean,
        float_inputs_std,
    ):
        super().__init__()
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
        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        lrelu_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                nn_utilities.init_orthogonal(m, lrelu_gain)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_orthogonal(m, lrelu_gain)

    def forward(self, img, float_inputs, use_fp32: bool = False) -> torch.Tensor:
        img_outputs = self.img_head((img.to(torch.float32 if use_fp32 else torch.float16) - 128) / 128)  # PERF
        # img_outputs = torch.zeros(batch_size, misc.conv_head_output_dim).to(device="cuda") # Uncomment to temporarily mask the img_head
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        # (batch_size, dense_input_dimension) OK
        return torch.cat((img_outputs, float_outputs), 1)


class LogPolicyNetwork(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim,
        float_hidden_dim,
        conv_head_output_dim,
        dense_hidden_dimension,
        n_actions,
        float_inputs_mean,
        float_inputs_std,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(float_inputs_dim, float_hidden_dim, float_inputs_mean, float_inputs_std)
        activation_function = torch.nn.LeakyReLU
        dense_input_dimension = conv_head_output_dim + float_hidden_dim
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(dense_input_dimension, dense_hidden_dimension // 2),
            activation_function(inplace=True),
            torch.nn.Linear(dense_hidden_dimension // 2, n_actions),
            torch.nn.LogSoftmax(dim=-1),
        )
        self.n_actions = n_actions
        self.initialize_weights()

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        lrelu_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)
        for m in self.policy_head:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_orthogonal(m, lrelu_gain)
        self.feature_extractor.initialize_weights()

    def forward(self, img, float_inputs, use_fp32: bool = False) -> torch.Tensor:
        concat = self.feature_extractor(img, float_inputs, use_fp32)
        return torch.clamp(
            self.policy_head(concat), min=-23.0
        )  # If the policy goes below 1e-10, we clamp it to 1e-10. Since we return the log policy, the clamp is -23


class SoftIQNQNetwork(torch.nn.Module):
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
        self.feature_extractor = FeatureExtractor(float_inputs_dim, float_hidden_dim, float_inputs_mean, float_inputs_std)
        activation_function = torch.nn.LeakyReLU
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
        lrelu_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)
        nn_utilities.init_orthogonal(
            self.iqn_fc, np.sqrt(2) * lrelu_gain
        )  # Since cosine has a variance of 1/2, and we would like to exit iqn_fc with a variance of 1, we need a weight variance double that of what a normal leaky relu would need
        for m in self.A_head[:-1]:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_orthogonal(m, lrelu_gain)
        nn_utilities.init_orthogonal(self.A_head[-1])
        for m in self.V_head[:-1]:
            if isinstance(m, torch.nn.Linear):
                nn_utilities.init_orthogonal(m, lrelu_gain)
        nn_utilities.init_orthogonal(self.V_head[-1])
        self.feature_extractor.initialize_weights()

    def forward(
        self, img, float_inputs, num_quantiles: int, tau: Optional[torch.Tensor] = None, use_fp32: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = img.shape[0]
        concat = self.feature_extractor(img, float_inputs, use_fp32)
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

        soft_Q = V + A - A.mean(dim=-1).unsqueeze(-1)
        return soft_Q, tau


class LogAlphaSingletonNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.nn.parameter.Parameter(torch.ones(1, requires_grad=True) * np.log(misc.sac_alpha))

    def forward(self) -> torch.Tensor:
        return self.log_alpha


# ==========================================================================================================================


class Trainer:
    __slots__ = (
        "soft_Q_model",
        "soft_Q_model2",
        "soft_Q_optimizer",
        "soft_Q_scaler",
        "policy_model",
        "policy_optimizer",
        "policy_scaler",
        "logalpha_model",
        "logalpha_optimizer",
        "logalpha_scaler",
        "batch_size",
        "iqn_k",
        "iqn_n",
        "iqn_kappa",
        "gamma",
        "IS_average",
        "truncation_amplitude",
        "target_entropy",
        "epsilon",
    )

    def __init__(
        self,
        soft_Q_model: SoftIQNQNetwork,
        soft_Q_model2: SoftIQNQNetwork,
        soft_Q_optimizer: torch.optim.Optimizer,
        soft_Q_scaler: torch.cuda.amp.grad_scaler.GradScaler,
        policy_model: LogPolicyNetwork,
        policy_optimizer: torch.optim.Optimizer,
        policy_scaler: torch.cuda.amp.grad_scaler.GradScaler,
        logalpha_model: LogAlphaSingletonNetwork,
        logalpha_optimizer: torch.optim.Optimizer,
        logalpha_scaler: torch.cuda.amp.grad_scaler.GradScaler,
        batch_size: int,
        iqn_k: int,
        iqn_n: int,
        iqn_kappa: float,
        gamma: float,
        truncation_amplitude: float,
        target_entropy: float,  # This parameter is typically set to dim(action_space)
        epsilon: float,
    ):
        self.soft_Q_model = soft_Q_model
        self.soft_Q_model2 = soft_Q_model2
        self.soft_Q_optimizer = soft_Q_optimizer
        self.soft_Q_scaler = soft_Q_scaler

        self.policy_model = policy_model
        self.policy_optimizer = policy_optimizer
        self.policy_scaler = policy_scaler

        self.logalpha_model = logalpha_model
        self.logalpha_optimizer = logalpha_optimizer
        self.logalpha_scaler = logalpha_scaler

        self.batch_size = batch_size
        self.iqn_k = iqn_k
        self.iqn_n = iqn_n
        self.iqn_kappa = iqn_kappa

        self.gamma = gamma
        self.IS_average = deque([], maxlen=100)
        self.truncation_amplitude = truncation_amplitude
        self.target_entropy = target_entropy

        self.epsilon = epsilon

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        self.soft_Q_optimizer.zero_grad(set_to_none=True)
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.logalpha_optimizer.zero_grad(set_to_none=True)

        sac_alpha = self.logalpha_model().exp().item()
        # sac_alpha = misc.sac_alpha  # TODO temporary test

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                batch, batch_info = buffer.sample(self.batch_size, return_info=True)
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
                    entropies_per_n_steps,
                ) = batch
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                if misc.prio_alpha > 0:
                    self.IS_average.append(batch_info["_weight"].mean())
                    IS_weights = torch.from_numpy(batch_info["_weight"] / np.mean(self.IS_average)).to("cuda", non_blocking=True)
                new_actions = new_actions.to(dtype=torch.int64)
                new_n_steps = new_n_steps.to(dtype=torch.int64)
                minirace_min_time_actions = minirace_min_time_actions.to(dtype=torch.int64)

                new_xxx = (
                    torch.rand(size=minirace_min_time_actions.shape, device="cuda")
                    * (misc.temporal_mini_race_duration_actions - minirace_min_time_actions)
                ).to(dtype=torch.int64)
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
                entropies = entropies_per_n_steps.gather(1, (possibly_reduced_n_steps - 1).unsqueeze(-1)).repeat(
                    [self.iqn_n, 1]
                )  # (batch_size*iqn_n, 1)     a,b,c,d devient a,b,c,d,a,b,c,d,a,b,c,d,...
                # (batch_size*iqn_n, 1)
                gammas_pow_nsteps = gammas_per_n_steps.gather(1, (possibly_reduced_n_steps - 1).unsqueeze(-1)).repeat([self.iqn_n, 1])
                done = new_done.reshape(-1, 1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                actions = new_actions[:, None]  # (batch_size, 1)
                actions_n = actions.repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)

                #
                #   Use policy_model
                #
                # log_pi_stpo = self.policy_model(next_state_img_tensor, next_state_float_tensor)
                

                """
                If we want to "truncate" the target distribution as in TQC, this is where we would do it. Force tau2 to be sampled for [0 ; 0.9 [ instead of [0 ; 1 [ for example.
                """
                #
                #   Use model2 to evaluate the action chosen, per quantile.
                #
                tau2 = (
                    torch.rand(size=(misc.batch_size * self.iqn_n, 1), device="cuda", dtype=torch.float32) * self.truncation_amplitude
                )  # (batch_size * num_quantiles, 1) (random numbers)

                q__stpo__model2__quantiles_tau2, _ = self.soft_Q_model2(
                    next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=tau2
                )  # (batch_size*iqn_n,n_actions)

                #
                #   Use model to choose an action for next state.
                #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
                #
                q__stpo__model__reduced = (
                    self.model(
                        next_state_img_tensor,
                        next_state_float_tensor,
                        self.iqn_n,
                        tau=None,
                    )[0]
                    .reshape([self.iqn_n, self.batch_size, self.model.n_actions])
                    .mean(dim=0) # (batch_size, n_actions)
                )
                log_pi_stpo = torch.nn.functional.log_softmax(q__stpo__model__reduced / sac_alpha, dim=-1) # (batch_size, n_actions)

                rewards_and_entropy = rewards + sac_alpha * entropies
                target = rewards_and_entropy + gammas_pow_nsteps * (
                    (q__stpo__model2__quantiles_tau2 - (sac_alpha * log_pi_stpo).repeat(self.iqn_n, 1))
                    * log_pi_stpo.exp().repeat(self.iqn_n, 1)
                ).sum(dim=1, keepdim=True)
                # (batch_size*iqn_n, 1)

                #
                #   Build IQN target on tau2 quantiles
                #
                outputs_target_tau2 = torch.where(
                    done,
                    rewards_and_entropy,
                    target,
                )  # (batch_size*iqn_n, 1)

                #
                #   This is our target
                #
                outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                    0, 1
                )  # (batch_size, iqn_n, 1)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            q__st__model__quantiles_tau3, tau3 = self.soft_Q_model(
                state_img_tensor, state_float_tensor, self.iqn_n, tau=None
            )  # (batch_size*iqn_n,n_actions)

            outputs_tau3 = (
                q__st__model__quantiles_tau3.gather(1, actions_n).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
            )  # (batch_size, iqn_n, 1)

            TD_error = outputs_target_tau2[:, :, None, :] - outputs_tau3[:, None, :, :]
            # (batch_size, iqn_n, iqn_n, 1)    WTF ????????
            # Huber loss, my alternative
            loss_Q_network = torch.where(
                torch.abs(TD_error) <= self.iqn_kappa,
                0.5 * TD_error**2,
                self.iqn_kappa * (torch.abs(TD_error) - 0.5 * self.iqn_kappa),
            )
            tau3 = tau3.reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
            tau3 = tau3[:, None, :, :].expand([-1, self.iqn_n, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
            loss_Q_network = (
                (torch.where(TD_error < 0, 1 - tau3, tau3) * loss_Q_network / self.iqn_kappa).sum(dim=2).mean(dim=1)[:, 0]
            )  # pinball loss # (batch_size, )

            total_loss_Q_network = torch.sum(IS_weights * loss_Q_network if misc.prio_alpha > 0 else loss_Q_network)

            # ===============================
            # log_pi_st = self.policy_model(state_img_tensor, state_float_tensor)  # (batch_size, n_actions)
            # pi_st = log_pi_st.exp()
            # q__st__model__averaged_over_quantiles = q__st__model__quantiles_tau3.reshape(
            #     [self.iqn_n, self.batch_size, self.soft_Q_model.n_actions]
            # ).mean(
            #     dim=0
            # )  # (batch_size, n_actions)
            # loss_policy = ((sac_alpha * log_pi_st - q__st__model__averaged_over_quantiles) * pi_st).sum(axis=1)  # (batch_size, )
            # total_loss_policy = torch.sum(IS_weights * loss_policy if misc.prio_alpha > 0 else loss_policy)
            # ===============================

            q__st__model__reduced = (
                    q__st__model__quantiles_tau3
                    .reshape([self.iqn_n, self.batch_size, self.model.n_actions])
                    .mean(dim=0) # (batch_size, n_actions)
                )
            log_pi_st = torch.nn.functional.log_softmax(q__st__model__reduced / sac_alpha, dim=-1) # (batch_size, n_actions)
            pi_st = log_pi_st.exp()

            # ===============================

            # Apparently optimizing the log of alpha instead of alpha is standard
            policy_entropy = -(pi_st.detach() * log_pi_st.detach()).sum(axis=1)
            loss_entropy = -self.logalpha_model() * (-policy_entropy + self.target_entropy)  # (batch_size, )
            total_loss_entropy = torch.sum(IS_weights * loss_entropy if misc.prio_alpha > 0 else loss_entropy)
            # ===============================

            if do_learn:
                self.soft_Q_scaler.scale(total_loss_Q_network).backward(retain_graph=True)
                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.soft_Q_scaler.unscale_(self.soft_Q_optimizer)
                _ = torch.nn.utils.clip_grad_norm_(self.soft_Q_model.parameters(), misc.clip_grad_norm).detach().cpu().item()
                torch.nn.utils.clip_grad_value_(self.soft_Q_model.parameters(), misc.clip_grad_value)
                self.soft_Q_scaler.step(self.soft_Q_optimizer)
                self.soft_Q_scaler.update()

                # self.policy_scaler.scale(total_loss_policy).backward()
                # # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                # self.policy_scaler.unscale_(self.policy_optimizer)
                # _ = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), misc.clip_grad_norm).detach().cpu().item()
                # torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), misc.clip_grad_value)
                # self.policy_scaler.step(self.policy_optimizer)
                # self.policy_scaler.update()

                self.logalpha_scaler.scale(total_loss_entropy).backward()
                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.logalpha_scaler.unscale_(self.logalpha_optimizer)
                _ = torch.nn.utils.clip_grad_norm_(self.logalpha_model.parameters(), misc.clip_grad_norm).detach().cpu().item()
                torch.nn.utils.clip_grad_value_(self.logalpha_model.parameters(), misc.clip_grad_value)
                self.logalpha_scaler.step(self.logalpha_optimizer)
                self.logalpha_scaler.update()

            total_loss_Q_network = total_loss_Q_network.detach().cpu()
            # total_loss_policy = total_loss_policy.detach().cpu()
            total_loss_entropy = total_loss_entropy.detach().cpu()
            if misc.prio_alpha > 0:
                buffer.update_priority(batch_info["index"], loss_Q_network.detach().cpu().type(torch.float64))
        return total_loss_Q_network, 0, total_loss_entropy, policy_entropy.cpu().mean()

    def get_exploration_action(self, img_inputs, float_inputs):
        with torch.no_grad():
            state_img_tensor = torch.from_numpy(img_inputs).unsqueeze(0).to("cuda", memory_format=torch.channels_last, non_blocking=True)
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            policy = (
                torch.nn.functional.softmax(
                    (self.soft_Q_model(
                        state_img_tensor,
                        state_float_tensor,
                        use_fp32=True,
                    )[0]
                    .reshape([self.iqn_n, self.batch_size, self.model.n_actions])
                    .mean(dim=0)) / self.logalpha_model().exp().item() # (batch_size, n_actions)
                    , dim=-1
                )
                .cpu()
            )

        if self.epsilon >= 0:
            # Train
            if random.random() < self.epsilon:
                # Choose a random action
                action_chosen_idx = random.randrange(len(misc.inputs))
                action_followed_stochastic_policy = False
            else:
                action_dist = Categorical(policy)
                action_chosen_idx = action_dist.sample().item()
                action_followed_stochastic_policy = True
        else:
            # Eval
            action_chosen_idx = np.argmax(policy)
            greedy_action_idx = action_chosen_idx
            action_followed_stochastic_policy = False #TODO should I put True here ?

        return (
            action_chosen_idx,
            action_followed_stochastic_policy,
            policy.max().item(),  # TODO change name in tensorboard
            policy.numpy(),  # TODO change name in tensorboard
        )
