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
        batch = tuple(
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
    batch_done_event = sampling_stream.record_event()
    return batch, batch_done_event
