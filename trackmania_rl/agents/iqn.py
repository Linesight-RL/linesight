"""
In this file, we define:
    - The IQN_Network class, which defines the neural network's structure.
    - The Trainer class, which implements the IQN training logic in method train_on_batch.
    - The Inferer class, which implements utilities for forward propagation with and without exploration.
"""

import copy
import math
import random
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl import utilities


class IQN_Network(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim: int,
        float_hidden_dim: int,
        conv_head_output_dim: int,
        dense_hidden_dimension: int,
        iqn_embedding_dimension: int,
        n_actions: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
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
        self.iqn_fc = torch.nn.Sequential(torch.nn.Linear(iqn_embedding_dimension, dense_input_dimension), torch.nn.LeakyReLU(inplace=True))
        self.initialize_weights()

        self.n_actions = n_actions

        # States are not normalized when the method forward() is called. Normalization is done as the first step of the forward() method.
        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")

    def initialize_weights(self):
        lrelu_neg_slope = 1e-2
        activation_gain = torch.nn.init.calculate_gain("leaky_relu", lrelu_neg_slope)
        for module in [self.img_head, self.float_feature_extractor, self.A_head[:-1], self.V_head[:-1]]:
            for m in module:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    utilities.init_orthogonal(m, activation_gain)
        utilities.init_orthogonal(
            self.iqn_fc[0], np.sqrt(2) * activation_gain
        )  # Since cosine has a variance of 1/2, and we would like to exit iqn_fc with a variance of 1, we need a weight variance double that of what a normal leaky relu would need
        for module in [self.A_head[-1], self.V_head[-1]]:
            utilities.init_orthogonal(module)

    def forward(
        self, img: torch.Tensor, float_inputs: torch.Tensor, num_quantiles: int, tau: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method implements the forward pass through the IQN neural network.

        The neural network is structured with two input heads:
            - one for images, with Conv2D layers
            - one for float features with Dense layers

        The embedding extracted by these two input heads are concatenated, mixed (Hadamard product) with an embedding for IQN quantiles.

        A dueling network architecture (https://arxiv.org/abs/1511.06581) is implemented, two output heads predict:
            - the value of a (state, quantile) pair
            - the advantage of a (state, action, quantile) triplet

        The Value and Advantage heads are combined to return the Q values directly.

        Args:
            img: a torch.Tensor of shape (batch_size, 1, H, W) and type float16 or float32, depending on context.
            float_inputs: a torch.Tensor of shape (batch_size, float_input_dim) and type float16 or float32, depending on context.
            num_quantiles: the number of quantiles, defined as N or N' in the IQN paper (https://arxiv.org/pdf/1806.06923).
            tau: if not None, a torch.Tensor of shape (batch_size * num_quantiles) the specifies the exact quantiles for which the neural network should return Q values
                 if None, the method will sample tau randomly in num_quantiles regularly spaced segments, and symmetrically around 0.5.

        Returns:
            Q: a torch.Tensor of shape (batch_size * num_quantiles, 1) representing the Q values for a given (state, quantile) combination
            tau: a torch.Tensor of shape (batch_size * num_quantiles, 1) representing the quantiles used to make each prediction
        """
        batch_size = img.shape[0]
        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        concat = torch.cat((img_outputs, float_outputs), 1)  # (batch_size, dense_input_dimension)
        if tau is None:
            tau = (
                torch.arange(num_quantiles // 2, device="cuda", dtype=torch.float32).repeat_interleave(batch_size).unsqueeze(1)
                + torch.rand(size=(batch_size * num_quantiles // 2, 1), device="cuda", dtype=torch.float32)
            ) / num_quantiles  # (batch_size * num_quantiles // 2, 1) (random numbers)
            tau = torch.cat((tau, 1 - tau), dim=0)  # ensure that tau are sampled symmetrically
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
        concat = concat.repeat(num_quantiles, 1)
        # (batch_size*num_quantiles, dense_input_dimension)
        concat = concat * quantile_net

        A = self.A_head(concat)  # (batch_size*num_quantiles, n_actions)
        V = self.V_head(concat)  # (batch_size*num_quantiles, 1)

        Q = V + A - A.mean(dim=-1).unsqueeze(-1)

        return Q, tau

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


@torch.compile(disable=not config_copy.is_linux, dynamic=False)
def iqn_loss(targets: torch.Tensor, outputs: torch.Tensor, tau_outputs: torch.Tensor, num_quantiles: int, batch_size: int):
    """
    Implements the IQN loss as defined in the IQN paper (https://arxiv.org/pdf/1806.06923)

    Args:
        targets: a torch.Tensor of shape (batch_size, num_quantiles, 1)
        outputs: a torch.Tensor of shape (batch_size, num_quantiles, 1)
        tau_outputs: a torch.Tensor of shape (batch_size * num_quantiles, 1)
        num_quantiles: (int)
        batch_size: (int)

    Returns:
        loss: a torch.Tensor of shape (batch_size, )
    """
    TD_error = targets[:, :, None, :] - outputs[:, None, :, :]
    # (batch_size, iqn_n, iqn_n, 1)
    loss = torch.where(
        torch.lt(torch.abs(TD_error), config_copy.iqn_kappa),
        (0.5 / config_copy.iqn_kappa) * TD_error**2,
        (torch.abs(TD_error) - 0.5 * config_copy.iqn_kappa),
    )
    tau = tau_outputs.reshape([num_quantiles, batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
    tau = tau[:, None, :, :].expand([-1, num_quantiles, -1, -1])  # (batch_size, iqn_n, iqn_n, 1)
    loss = (torch.where(torch.lt(TD_error, 0), 1 - tau, tau) * loss).sum(dim=2).mean(dim=1)[:, 0]  # pinball loss # (batch_size, )
    return loss


class Trainer:
    __slots__ = (
        "online_network",
        "target_network",
        "optimizer",
        "scaler",
        "batch_size",
        "iqn_n",
        "typical_self_loss",
        "typical_clamped_self_loss",
    )

    def __init__(
        self,
        online_network: IQN_Network,
        target_network: IQN_Network,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        batch_size: int,
        iqn_n: int,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.iqn_n = iqn_n
        self.typical_self_loss = 0.01
        self.typical_clamped_self_loss = 0.01

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        """
        Implements one iteration of the training loop:
            1) Sample a batch of transitions from the replay buffer
            2) Calculate the IQN loss
            3) Obtain gradients through backpropagation
            4) Update the neural network weights using the optimizer

        The training loop may be configured to use DDQN-style updates with config.use_ddqn.

        Args:
            buffer: a ReplayBuffer object from which transitions are sampled. Currently, handles a basic buffer or a prioritized replay buffer.
            do_learn: a boolean indicating whether steps 3 and 4 should be applied. If these are not applied, the method only returns total_loss and grad_norm for logging purposes.

        Returns:
            total_loss: a float
            grad_norm: a float

        """
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
                if config_copy.prio_alpha > 0:
                    IS_weights = torch.from_numpy(batch_info["_weight"]).to("cuda", non_blocking=True)

                rewards = rewards.unsqueeze(-1).repeat(
                    [self.iqn_n, 1]
                )  # (batch_size*iqn_n, 1)     a,b,c,d becomes a,b,c,d,a,b,c,d,a,b,c,d,... (iqn_n times)
                gammas_terminal = gammas_terminal.unsqueeze(-1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                actions = actions.unsqueeze(-1).repeat([self.iqn_n, 1])  # (batch_size*iqn_n, 1)
                #
                #   Use target_network to evaluate the action chosen, per quantile.
                #
                q__stpo__target__quantiles_tau2, tau2 = self.target_network(
                    next_state_img_tensor, next_state_float_tensor, self.iqn_n, tau=None
                )  # (batch_size*iqn_n, n_actions)
                #
                #   Use online network to choose an action for next state.
                #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
                #
                if config_copy.use_ddqn:
                    a__tpo__online__reduced_repeated = (
                        self.online_network(
                            next_state_img_tensor,
                            next_state_float_tensor,
                            self.iqn_n,
                            tau=None,
                        )[0]
                        .reshape([self.iqn_n, self.batch_size, self.online_network.n_actions])
                        .mean(dim=0)
                        .argmax(dim=1, keepdim=True)
                        .repeat([self.iqn_n, 1])
                    )  # (iqn_n * batch_size, 1)
                    #
                    #   Build IQN target on tau2 quantiles
                    #
                    outputs_target_tau2 = rewards + gammas_terminal * q__stpo__target__quantiles_tau2.gather(
                        1, a__tpo__online__reduced_repeated
                    )  # (batch_size*iqn_n, 1)
                else:
                    outputs_target_tau2 = (
                        rewards + gammas_terminal * q__stpo__target__quantiles_tau2.max(dim=1, keepdim=True)[0]
                    )  # (batch_size*iqn_n, 1)

                #
                #   This is our target
                #
                outputs_target_tau2 = outputs_target_tau2.reshape([self.iqn_n, self.batch_size, 1]).transpose(
                    0, 1
                )  # (batch_size, iqn_n, 1)

            q__st__online__quantiles_tau3, tau3 = self.online_network(
                state_img_tensor, state_float_tensor, self.iqn_n, tau=None
            )  # (batch_size*iqn_n,n_actions)
            outputs_tau3 = (
                q__st__online__quantiles_tau3.gather(1, actions).reshape([self.iqn_n, self.batch_size, 1]).transpose(0, 1)
            )  # (batch_size, iqn_n, 1)

            loss = iqn_loss(outputs_target_tau2, outputs_tau3, tau3, config_copy.iqn_n, config_copy.batch_size)

            target_self_loss = torch.sqrt(
                iqn_loss(
                    outputs_target_tau2.detach(), outputs_target_tau2.detach(), tau2.detach(), config_copy.iqn_n, config_copy.batch_size
                )
            )

            self.typical_self_loss = 0.99 * self.typical_self_loss + 0.01 * target_self_loss.mean()

            correction_clamped = target_self_loss.clamp(min=self.typical_self_loss / config_copy.target_self_loss_clamp_ratio)

            self.typical_clamped_self_loss = 0.99 * self.typical_clamped_self_loss + 0.01 * correction_clamped.mean()

            loss *= self.typical_clamped_self_loss / correction_clamped

            total_loss = torch.sum(IS_weights * loss if config_copy.prio_alpha > 0 else loss)

            if do_learn:
                self.scaler.scale(total_loss).backward()

                # Gradient clipping : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), config_copy.clip_grad_norm).detach().cpu().item()
                )
                torch.nn.utils.clip_grad_value_(self.online_network.parameters(), config_copy.clip_grad_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = 0

            total_loss = total_loss.detach().cpu()
            if config_copy.prio_alpha > 0:
                mask_update_priority = torch.lt(state_float_tensor[:, 0], config_copy.min_horizon_to_update_priority_actions).detach().cpu()
                # Only update the transition priority if the transition was sampled with a sufficiently long-term horizon.
                buffer.update_priority(
                    batch_info["index"][mask_update_priority],
                    (outputs_tau3.mean(axis=1) - outputs_target_tau2.mean(axis=1))
                    .abs()[mask_update_priority]
                    .detach()
                    .cpu()
                    .type(torch.float64),
                )
        return total_loss, grad_norm


class Inferer:
    __slots__ = (
        "inference_network",
        "iqn_k",
        "epsilon",
        "epsilon_boltzmann",
        "tau_epsilon_boltzmann",
        "is_explo",
    )

    def __init__(self, inference_network, iqn_k, tau_epsilon_boltzmann):
        self.inference_network = inference_network
        self.iqn_k = iqn_k
        self.epsilon = None
        self.epsilon_boltzmann = None
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray, tau=None) -> npt.NDArray:
        """
        Perform inference of a single state through self.inference_network.

        Args:
            img_inputs_uint8:   a numpy array of shape (1, H, W) and dtype np.uint8
            float_inputs:       a numpy array of shape (float_input_dim, ) and dtype np.float32
            tau:                a torch.Tensor of shape (iqn_k,  1)

        Returns:
            q_values:           a numpy array of shape (iqn_k, 1)
        """
        with torch.no_grad():
            state_img_tensor = (
                torch.from_numpy(img_inputs_uint8)
                .unsqueeze(0)
                .to("cuda", memory_format=torch.channels_last, non_blocking=True, dtype=torch.float32)
                - 128
            ) / 128
            state_float_tensor = torch.from_numpy(np.expand_dims(float_inputs, axis=0)).to("cuda", non_blocking=True)
            q_values = (
                self.inference_network(
                    state_img_tensor,
                    state_float_tensor,
                    self.iqn_k,
                    tau=tau,  # torch.linspace(0.05, 0.95, self.iqn_k, device="cuda")[:, None],
                )[0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return q_values

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> Tuple[int, bool, float, npt.NDArray]:
        """
        Selects an action according to the exploration strategy.
        Implements epsilon-greedy exploration, as well as Boltzmann exploration, quantiles values are averaged.
        Configuration is done with self.epsilon (float), self.epsilon_boltzmann (float), self.tau_epsilon_boltzmann (float), and self.is_explo (bool).

        Args:
            img_inputs_uint8:   a numpy array of shape (1, H, W) and dtype np.uint8
            float_inputs:       a numpy array of shape (float_input_dim, ) and dtype np.float32

        Returns:
            action_chosen_idx:  an int indicating which exploration action is sampled
            is_greedy:          a bool indicating whether this action would have been chosen under a greedy policy
            V(state):           a float giving the value of the greedy action
            q_values:           a numpy array giving the q_values for all actions
        """

        q_values = self.infer_network(img_inputs_uint8, float_inputs).mean(axis=0)
        r = random.random()

        if self.is_explo and r < self.epsilon:
            # Choose a random action
            get_argmax_on = np.random.randn(*q_values.shape)
        elif self.is_explo and r < self.epsilon + self.epsilon_boltzmann:
            get_argmax_on = q_values + self.tau_epsilon_boltzmann * np.random.randn(*q_values.shape)
        else:
            get_argmax_on = q_values

        action_chosen_idx = np.argmax(get_argmax_on)
        greedy_action_idx = np.argmax(q_values)

        return (
            action_chosen_idx,
            action_chosen_idx == greedy_action_idx,
            np.max(q_values),
            q_values,
        )


def make_untrained_iqn_network(jit: bool, is_inference: bool) -> Tuple[IQN_Network, IQN_Network]:
    """
    Constructs two identical copies of the IQN network.

    The first copy is compiled (if jit == True) and is used for inference, for rollouts, for training, etc...
    The second copy is never compiled and **only** used to efficiently share a neural network's weights between processes.

    Args:
        jit: a boolean indicating whether compilation should be used
    """

    uncompiled_model = IQN_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        iqn_embedding_dimension=config_copy.iqn_embedding_dimension,
        n_actions=len(config_copy.inputs),
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    )
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else ("max-autotune" if is_inference else "max-autotune-no-cudagraphs")
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(uncompiled_model)
    else:
        model = copy.deepcopy(uncompiled_model)
    return (
        model.to(device="cuda", memory_format=torch.channels_last).train(),
        uncompiled_model.to(device="cuda", memory_format=torch.channels_last).train(),
    )
