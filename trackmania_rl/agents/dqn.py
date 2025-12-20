
"""
In this file, we define:
    - The DQN_Network class, which defines the neural network's structure.
    - The Trainer class, which implements the DQN training logic in method train_on_batch.
    - The Inferer class, which implements utilities for forward propagation with and without exploration.
"""

import copy
import math
import random
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl import utilities


class DQN_Network(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim: int,
        float_hidden_dim: int,
        conv_head_output_dim: int,
        dense_hidden_dimension: int,
        n_actions: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
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
        
        for module in [self.A_head[-1], self.V_head[-1]]:
            utilities.init_orthogonal(module)

    def forward(
        self, img: torch.Tensor, float_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        This method implements the forward pass through the DQN neural network.

        Args:
            img: a torch.Tensor of shape (batch_size, 1, H, W)
            float_inputs: a torch.Tensor of shape (batch_size, float_input_dim)

        Returns:
            Q: a torch.Tensor of shape (batch_size, n_actions)
        """
        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        concat = torch.cat((img_outputs, float_outputs), 1)  # (batch_size, dense_input_dimension)
        
        A = self.A_head(concat)  # (batch_size, n_actions)
        V = self.V_head(concat)  # (batch_size, 1)

        Q = V + A - A.mean(dim=-1).unsqueeze(-1)  # (batch_size, n_actions)

        return Q

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


class Trainer:
    __slots__ = (
        "online_network",
        "target_network",
        "optimizer",
        "scaler",
        "batch_size",
        "typical_self_loss",
        "typical_clamped_self_loss",
    )

    def __init__(
        self,
        online_network: DQN_Network,
        target_network: DQN_Network,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        batch_size: int,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size
        self.typical_self_loss = 0.01
        self.typical_clamped_self_loss = 0.01

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool):
        """
        Implements one iteration of the training loop.
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

                # DQN Target Calculation
                if config_copy.use_ddqn:
                    # Double DQN: Use online network to select action, target network to evaluate
                    next_q_online = self.online_network(next_state_img_tensor, next_state_float_tensor)
                    best_action = next_q_online.argmax(dim=1, keepdim=True)
                    
                    next_q_target = self.target_network(next_state_img_tensor, next_state_float_tensor)
                    max_next_q = next_q_target.gather(1, best_action)
                else:
                    # DQN: Use target network to select & evaluate
                    next_q_target = self.target_network(next_state_img_tensor, next_state_float_tensor)
                    max_next_q = next_q_target.max(dim=1, keepdim=True)[0]
                
                target_q = rewards.unsqueeze(-1) + gammas_terminal.unsqueeze(-1) * max_next_q
                
            # Current Q Calculation
            current_q_values = self.online_network(state_img_tensor, state_float_tensor)
            current_q = current_q_values.gather(1, actions.unsqueeze(-1))

            # Loss
            loss = F.smooth_l1_loss(current_q, target_q, reduction='none').squeeze(-1)
            
            # Since iqn_loss had some specific pinball loss, here we just use Huber/SmoothL1 or MSE.
            # Using smooth_l1_loss (Huber loss) is standard for DQN.  
             
            total_loss = torch.sum(IS_weights * loss if config_copy.prio_alpha > 0 else loss)

            if do_learn:
                self.scaler.scale(total_loss).backward()

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
                
                td_errors = (current_q - target_q).abs().squeeze(-1)
                
                buffer.update_priority(
                    batch_info["index"][mask_update_priority],
                    td_errors[mask_update_priority].detach().cpu().type(torch.float64)
                )
        return total_loss, grad_norm


class Inferer:
    __slots__ = (
        "inference_network",
        "epsilon",
        "epsilon_boltzmann",
        "tau_epsilon_boltzmann",
        "is_explo",
    )

    def __init__(self, inference_network, tau_epsilon_boltzmann):
        self.inference_network = inference_network
        self.epsilon = None
        self.epsilon_boltzmann = None
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann
        self.is_explo = None

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> npt.NDArray:
        """
        Perform inference of a single state through self.inference_network.
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
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )[0]
            
            return q_values

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> Tuple[int, bool, float, npt.NDArray]:
        
        q_values = self.infer_network(img_inputs_uint8, float_inputs)
        # q_values is (n_actions,)

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


def make_untrained_dqn_network(jit: bool, is_inference: bool) -> Tuple[DQN_Network, DQN_Network]:
    """
    Constructs two identical copies of the DQN network.
    """

    uncompiled_model = DQN_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
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
