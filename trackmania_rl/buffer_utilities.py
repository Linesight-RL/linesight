import numpy as np
import torch

from . import misc

to_torch_dtype = {"uint8": torch.uint8, "float32": torch.float32, "int64": torch.int64, "float": torch.float64, "int": torch.int}


def fast_collate_cpu(batch, attr_name):
    elem = getattr(batch[0], attr_name)
    elem_array = hasattr(elem, "__len__")
    shape = (len(batch),) + (elem.shape if elem_array else ())
    data_type = elem.flat[0].dtype if elem_array else type(elem).__name__
    data_type = to_torch_dtype[str(data_type)]
    buffer = torch.empty(size=shape, dtype=data_type, pin_memory=True).numpy()
    np.copyto(buffer, [getattr(memory, attr_name) for memory in batch])
    return buffer


def send_to_gpu(batch, attr_name):
    return torch.as_tensor(batch).to(
        non_blocking=True, device="cuda", memory_format=torch.channels_last if "img" in attr_name else torch.preserve_format
    )


def buffer_collate_function(batch):
    state_img, state_float, action, rewards, next_state_img, next_state_float, gammas, terminal_actions, n_steps = tuple(
        map(
            lambda attr_name: fast_collate_cpu(batch, attr_name),
            [
                "state_img",
                "state_float",
                "action",
                "rewards",
                "next_state_img",
                "next_state_float",
                "gammas",
                "terminal_actions",
                "n_steps",
            ],
        )
    )
    if misc.apply_horizontal_flip_augmentation:
        # Apply Horizontal Flipping
        use_horizontal_flip = torch.rand(len(state_img), device="cuda") < misc.flip_augmentation_ratio
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
            floats_flipped[:, misc.flip_indices_floats_before_swap] = floats_flipped[
                :, misc.flip_indices_floats_after_swap
            ]  # Swap left right features
            floats_flipped[:, misc.indices_floats_sign_inversion] *= -1  # Multiply by -1 relevant coordinates
            return floats_flipped

        state_float = torch.where(use_horizontal_flip[:, None], float_inputs_horizontal_symmetry(state_float), state_float)
        next_state_float = torch.where(use_horizontal_flip[:, None], float_inputs_horizontal_symmetry(next_state_float), next_state_float)

    temporal_mini_race_current_time_actions = np.random.randint(
        low=0, high=misc.temporal_mini_race_duration_actions, size=(len(state_img),), dtype=int
    )
    temporal_mini_race_next_time_actions = temporal_mini_race_current_time_actions + n_steps

    state_float[:, 0] = temporal_mini_race_current_time_actions
    next_state_float[:, 0] = temporal_mini_race_next_time_actions

    possibly_reduced_n_steps = n_steps - (temporal_mini_race_next_time_actions - misc.temporal_mini_race_duration_actions).clip(min=0)

    terminal = (possibly_reduced_n_steps >= terminal_actions) | (
        temporal_mini_race_next_time_actions >= misc.temporal_mini_race_duration_actions
    )

    gammas = np.take_along_axis(gammas, possibly_reduced_n_steps[:, None] - 1, axis=1).squeeze(-1)
    gammas = np.where(terminal, 0, gammas)

    rewards = np.take_along_axis(rewards, possibly_reduced_n_steps[:, None] - 1, axis=1).squeeze(-1)

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

    return (
        state_img,
        state_float,
        action,
        rewards,
        next_state_img,
        next_state_float,
        gammas,
    )
