import random
from copy import deepcopy
from typing import Any, Dict, Union

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch import Tensor
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.utils import INT_CLASSES, _to_numpy

from trackmania_rl import misc_copy

to_torch_dtype = {
    "uint8": torch.uint8,
    "float32": torch.float32,
    "int64": torch.int64,
    "float": torch.float32,
    "int": torch.int,
    "float64": torch.float32,
}


def fast_collate_cpu(batch, attr_name):
    elem = getattr(batch[0], attr_name)
    elem_array = hasattr(elem, "__len__")
    shape = (len(batch),) + (elem.shape if elem_array else ())
    data_type = elem.flat[0].dtype if elem_array else type(elem).__name__
    data_type = to_torch_dtype[str(data_type)]
    buffer = torch.empty(size=shape, dtype=data_type, pin_memory=True).numpy()
    source = [getattr(memory, attr_name) for memory in batch]
    buffer[:] = source[:]
    return buffer


def send_to_gpu(batch, attr_name):
    return torch.as_tensor(batch).to(
        non_blocking=True, device="cuda", memory_format=torch.channels_last if "img" in attr_name else torch.preserve_format
    )


def buffer_collate_function(batch):
    (
        state_img,
        state_float,
        state_potential,
        action,
        rewards,
        next_state_img,
        next_state_float,
        next_state_potential,
        gammas,
        terminal_actions,
        n_steps,
    ) = tuple(
        map(
            lambda attr_name: fast_collate_cpu(batch, attr_name),
            [
                "state_img",
                "state_float",
                "state_potential",
                "action",
                "rewards",
                "next_state_img",
                "next_state_float",
                "next_state_potential",
                "gammas",
                "terminal_actions",
                "n_steps",
            ],
        )
    )

    temporal_mini_race_current_time_actions = (
        np.abs(
            np.random.randint(
                low=-misc_copy.oversample_long_term_steps + misc_copy.oversample_maximum_term_steps,
                high=misc_copy.temporal_mini_race_duration_actions + misc_copy.oversample_maximum_term_steps,
                size=(len(state_img),),
                dtype=int,
            )
        )
        - misc_copy.oversample_maximum_term_steps
    ).clip(min=0)

    temporal_mini_race_next_time_actions = temporal_mini_race_current_time_actions + n_steps

    state_float[:, 0] = temporal_mini_race_current_time_actions
    next_state_float[:, 0] = temporal_mini_race_next_time_actions

    possibly_reduced_n_steps = n_steps - (temporal_mini_race_next_time_actions - misc_copy.temporal_mini_race_duration_actions).clip(min=0)

    terminal = (possibly_reduced_n_steps >= terminal_actions) | (
        temporal_mini_race_next_time_actions >= misc_copy.temporal_mini_race_duration_actions
    )

    gammas = np.take_along_axis(gammas, possibly_reduced_n_steps[:, None] - 1, axis=1).squeeze(-1)
    gammas = np.where(terminal, 0, gammas)

    rewards = np.take_along_axis(rewards, possibly_reduced_n_steps[:, None] - 1, axis=1).squeeze(-1)

    rewards += np.where(terminal, 0, gammas * next_state_potential)
    rewards -= state_potential

    state_img, state_float, action, rewards, next_state_img, next_state_float, gammas = tuple(
        map(
            lambda batch, attr_name: send_to_gpu(batch, attr_name),
            [
                state_img,
                state_float,
                action,
                rewards,
                next_state_img,
                next_state_float,
                gammas,
            ],
            [
                "state_img",
                "state_float",
                "action",
                "rewards",
                "next_state_img",
                "next_state_float",
                "gammas",
            ],
        )
    )

    state_img = (state_img.to(torch.float16) - 128) / 128
    next_state_img = (next_state_img.to(torch.float16) - 128) / 128

    if misc_copy.apply_randomcrop_augmentation:
        # Same transformation is applied for state and next_state.
        # Different transformation is applied to each element in a batch.
        i = random.randint(0, 2 * misc_copy.n_pixels_to_crop_on_each_side)
        j = random.randint(0, 2 * misc_copy.n_pixels_to_crop_on_each_side)
        state_img = transforms.functional.crop(
            transforms.functional.pad(state_img, padding=misc_copy.n_pixels_to_crop_on_each_side, padding_mode="edge"),
            i,
            j,
            misc_copy.H_downsized,
            misc_copy.W_downsized,
        )
        next_state_img = transforms.functional.crop(
            transforms.functional.pad(next_state_img, padding=misc_copy.n_pixels_to_crop_on_each_side, padding_mode="edge"),
            i,
            j,
            misc_copy.H_downsized,
            misc_copy.W_downsized,
        )

    if misc_copy.apply_horizontal_flip_augmentation:
        # Apply Horizontal Flipping
        use_horizontal_flip = torch.rand(len(state_img), device="cuda") < misc_copy.flip_augmentation_ratio
        state_img = torch.where(use_horizontal_flip[:, None, None, None], torch.flip(state_img, dims=(-1,)), state_img)  # state_img
        next_state_img = torch.where(
            use_horizontal_flip[:, None, None, None], torch.flip(next_state_img, dims=(-1,)), next_state_img
        )  # next_state_img
        # 0 Forward 1 Forward left 2 Forward right 3 Nothing 4 Nothing left 5 Nothing right 6 Brake 7 Brake left 8 Brake right 9 Brake and accelerate 10 Brake and accelerate left 11 Brake and accelerate right
        # becomes
        # 0 Forward 1 Forward right 2 Forward left 3 Nothing 4 Nothing right 5 Nothing left 6 Brake 7 Brake right 8 Brake left 9 Brake and accelerate 10 Brake and accelerate right 11 Brake and accelerate left
        action_flipped = torch.tensor([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10], device="cuda", dtype=torch.int64)
        action = torch.where(use_horizontal_flip, torch.gather(action_flipped, 0, action), action)

        # From SaiMoen on TMI Discord, the order of wheels in simulation_state is fl, fr, br, bl

        def float_inputs_horizontal_symmetry(floats):
            floats_flipped = floats.clone()
            floats_flipped[:, misc_copy.flip_indices_floats_before_swap] = floats_flipped[
                :, misc_copy.flip_indices_floats_after_swap
            ]  # Swap left right features
            floats_flipped[:, misc_copy.indices_floats_sign_inversion] *= -1  # Multiply by -1 relevant coordinates
            return floats_flipped

        state_float = torch.where(use_horizontal_flip[:, None], float_inputs_horizontal_symmetry(state_float), state_float)
        next_state_float = torch.where(use_horizontal_flip[:, None], float_inputs_horizontal_symmetry(next_state_float), next_state_float)

    return (
        state_img,
        state_float,
        action,
        rewards,
        next_state_img,
        next_state_float,
        gammas,
    )


class CustomPrioritizedSampler(PrioritizedSampler):
    """
    Custom Prioritized Sampler which implements a slightly modified behavior compared to torchrl's original implementation.

    A memory's default priority is based on all memories' average priority,
    instead of the maximum priority seen since the beginning of training.
    """

    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        reduction: str = "max",
        default_priority_ratio: float = 2.0,
    ) -> None:
        super(CustomPrioritizedSampler, self).__init__(max_capacity, alpha, beta, eps, dtype, reduction)
        self._average_priority = None
        self._default_priority_ratio = default_priority_ratio
        self._uninitialized_memories = 0.0

    @property
    def default_priority(self) -> float:
        if self._average_priority is None:
            self._uninitialized_memories += 1.0
            return 0.4
        else:
            return self._default_priority_ratio * self._average_priority

    def sample(self, storage: Storage, batch_size: int) -> tuple[Tensor, dict[str, Any]]:
        if len(storage) == 0:
            raise RuntimeError("Cannot sample from an empty storage.")
        p_sum = self._sum_tree.query(0, len(storage))
        self._average_priority = p_sum / len(storage)
        if p_sum <= 0:
            raise RuntimeError("negative p_sum")
        mass = np.random.uniform(0.0, p_sum, size=batch_size)
        index = self._sum_tree.scan_lower_bound(mass)
        if not isinstance(index, np.ndarray):
            index = np.array([index])
        if isinstance(index, torch.Tensor):
            index.clamp_max_(len(storage) - 1)
        else:
            index = np.clip(index, None, len(storage) - 1)
        if self._uninitialized_memories > 0.0:
            return index, {"_weight": 0.5 * np.ones(len(index))}
        else:
            weight = np.power((len(storage) * self._sum_tree[index] / p_sum), -self._beta)
            return index, {"_weight": weight}

    def update_priority(self, index: Union[int, torch.Tensor], priority: Union[float, torch.Tensor]) -> None:
        """Updates the priority of the data pointed by the index.

        Args:
            index (int or torch.Tensor): indexes of the priorities to be
                updated.
            priority (Number or torch.Tensor): new priorities of the
                indexed elements.

        """
        if isinstance(index, INT_CLASSES):
            if not isinstance(priority, float):
                if len(priority) != 1:
                    raise RuntimeError(f"priority length should be 1, got {len(priority)}")
                priority = priority.item()
                self._uninitialized_memories -= 0.5
        else:
            if not (isinstance(priority, float) or len(priority) == 1 or len(index) == len(priority)):
                raise RuntimeError("priority should be a number or an iterable of the same " "length as index")
            index = _to_numpy(index)
            priority = _to_numpy(priority)
            # We track the _approximate_ number of memories in the buffer that have default priority :
            self._uninitialized_memories -= 0.3 * len(index)
        priority = np.power(priority + self._eps, self._alpha)
        self._sum_tree[index] = priority

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_alpha": self._alpha,
            "_beta": self._beta,
            "_eps": self._eps,
            "_average_priority": self._average_priority,
            "_default_priority_ratio": self._default_priority_ratio,
            "_sum_tree": deepcopy(self._sum_tree),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._alpha = state_dict["_alpha"]
        self._beta = state_dict["_beta"]
        self._eps = state_dict["_eps"]
        self._average_priority = state_dict["_average_priority"]
        self._default_priority_ratio = state_dict["_default_priority_ratio"]
        self._sum_tree = state_dict.pop("_sum_tree")


def copy_buffer_content_to_other_buffer(source_buffer: ReplayBuffer, target_buffer: ReplayBuffer) -> None:
    assert source_buffer._storage.max_size <= target_buffer._storage.max_size

    target_buffer.extend(source_buffer._storage)

    if isinstance(source_buffer._sampler, CustomPrioritizedSampler) and isinstance(target_buffer._sampler, CustomPrioritizedSampler):
        target_buffer._sampler._average_priority = source_buffer._sampler._average_priority
        target_buffer._sampler._uninitialized_memories = source_buffer._sampler._uninitialized_memories

    if isinstance(source_buffer._sampler, PrioritizedSampler) and isinstance(target_buffer._sampler, PrioritizedSampler):
        for i in range(len(source_buffer)):
            target_buffer._sampler._sum_tree[i] = source_buffer._sampler._sum_tree.at(i)


def make_buffers(buffer_size: int) -> tuple[ReplayBuffer, ReplayBuffer]:
    buffer = ReplayBuffer(
        storage=ListStorage(buffer_size),
        batch_size=misc_copy.batch_size,
        collate_fn=buffer_collate_function,
        prefetch=1,
        sampler=CustomPrioritizedSampler(buffer_size, misc_copy.prio_alpha, misc_copy.prio_beta, misc_copy.prio_epsilon, torch.float64)
        if misc_copy.prio_alpha > 0
        else RandomSampler(),
    )
    buffer_test = ReplayBuffer(
        storage=ListStorage(int(buffer_size * misc_copy.buffer_test_ratio)),
        batch_size=misc_copy.batch_size,
        collate_fn=buffer_collate_function,
        sampler=CustomPrioritizedSampler(buffer_size, misc_copy.prio_alpha, misc_copy.prio_beta, misc_copy.prio_epsilon, torch.float64)
        if misc_copy.prio_alpha > 0
        else RandomSampler(),
    )
    return buffer, buffer_test


def resize_buffers(buffer: ReplayBuffer, buffer_test: ReplayBuffer, new_buffer_size: int) -> tuple[ReplayBuffer, ReplayBuffer]:
    new_buffer, new_buffer_test = make_buffers(new_buffer_size)
    copy_buffer_content_to_other_buffer(buffer, new_buffer)
    copy_buffer_content_to_other_buffer(buffer_test, new_buffer_test)
    return new_buffer, new_buffer_test
