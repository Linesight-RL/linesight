import numpy as np
import torch

from . import misc


def fast_collate(batch, attr_name):
    return torch.as_tensor(np.array([getattr(memory, attr_name) for memory in batch])).to(
        non_blocking=True, device="cuda", memory_format=torch.channels_last if "img" in attr_name else torch.preserve_format
    )


def fast_collate2(batch, attr_name):
    if "img" in attr_name:
        images = torch.empty(
            (len(batch), 1, misc.H_downsized, misc.W_downsized), device="cuda", dtype=torch.uint8, memory_format=torch.channels_last
        )
        for i, memory in enumerate(batch):
            images[i].copy_(getattr(memory, attr_name), non_blocking=True)
        return images
    else:
        return fast_collate(batch, attr_name)


def buffer_collate_function(batch, sampling_stream):
    with torch.cuda.stream(sampling_stream):
        state_img, state_float, action, n_steps, rewards, next_state_img, next_state_float, gammas, minirace_min_time_actions = tuple(
            map(
                lambda attr_name: fast_collate2(batch, attr_name),
                [
                    "state_img",
                    "state_float",
                    "action",
                    "n_steps",
                    "rewards",
                    "next_state_img",
                    "next_state_float",
                    "gammas",
                    "minirace_min_time_actions",
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
            next_state_float = torch.where(
                use_horizontal_flip[:, None], float_inputs_horizontal_symmetry(next_state_float), next_state_float
            )

    batch_done_event = sampling_stream.record_event()
    return (
        state_img,
        state_float,
        action,
        n_steps,
        rewards,
        next_state_img,
        next_state_float,
        gammas,
        minirace_min_time_actions,
    ), batch_done_event
