import numpy as np
import torch

from . import misc

sampling_stream = torch.cuda.Stream()


def fast_collate(batch, attr_name):
    return torch.as_tensor(np.array([getattr(memory, attr_name) for memory in batch])).to(
        non_blocking=True, device="cuda", memory_format=torch.channels_last if "img" in attr_name else torch.preserve_format
    )


def fast_collate2(batch, attr_name):
    if "img" in attr_name:
        images = torch.empty((len(batch), 1, misc.H, misc.W), device="cuda", dtype=torch.uint8, memory_format=torch.channels_last)
        for i, memory in enumerate(batch):
            images[i].copy_(getattr(memory, attr_name), non_blocking=True)
        return images
    else:
        return fast_collate(batch, attr_name)


def buffer_collate_function(batch):
    with torch.cuda.stream(sampling_stream):
        batch = tuple(
            map(
                lambda attr_name: fast_collate2(batch, attr_name),
                [
                    "state_img",
                    "state_float",
                    "action",
                    "reward",
                    "gamma_pow_nsteps",
                    "done",
                    "next_state_img",
                    "next_state_float",
                ],
            )
        )
    batch_done_event = sampling_stream.record_event()
    # sampling_stream.synchronize()
    return batch, batch_done_event
