import warnings
from collections import deque
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.data.replay_buffers.utils import INT_CLASSES
from torchrl.data.replay_buffers.writers import RoundRobinWriter


# Modified from https://pytorch.org/rl/_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer
class ReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        collate_fn: Optional[Callable] = None,
        prefetch: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self._storage = ListStorage(max_size=capacity)
        self._storage.attach(self)
        self._sampler = RandomSampler()
        self._writer = RoundRobinWriter()
        self._writer.register_storage(self._storage)
        self._collate_fn = collate_fn
        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = deque()

        if batch_size is None and prefetch:
            raise ValueError(
                "Dynamic batch-size specification is incompatible "
                "with multithreaded sampling. "
                "When using prefetch, the batch-size must be specified in "
                "advance. "
            )
        if batch_size is None and hasattr(self._sampler, "drop_last") and self._sampler.drop_last:
            raise ValueError(
                "Samplers with drop_last=True must work with a predictible batch-size. "
                "Please pass the batch-size to the ReplayBuffer constructor."
            )
        self._batch_size = batch_size
        self.sampling_stream = torch.cuda.Stream()

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(" f"storage={self._storage}, " f"sampler={self._sampler}, " f"writer={self._writer}" ")"

    def add(self, data: Any) -> int:
        index = self._writer.add(data)
        self._sampler.add(index)
        return index

    def _extend(self, data: Sequence) -> torch.Tensor:
        index = self._writer.extend(data)
        self._sampler.extend(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        return self._extend(data)

    def update_priority(
        self,
        index: Union[int, torch.Tensor],
        priority: Union[int, torch.Tensor],
    ) -> None:
        self._sampler.update_priority(index, priority)

    def _sample(self, batch_size: int) -> Tuple[Any, dict, Any]:
        index, info = self._sampler.sample(self._storage, batch_size)
        info["index"] = index
        data = self._storage[index]
        if not isinstance(index, INT_CLASSES) and self._collate_fn is not None:
            data, cuda_batch_event = self._collate_fn(data, self.sampling_stream)
        return data, info, cuda_batch_event

    def sample(self, batch_size: Optional[int] = None, return_info: bool = False) -> Any:
        if batch_size is not None and self._batch_size is not None and batch_size != self._batch_size:
            warnings.warn(
                f"Got conflicting batch_sizes in constructor ({self._batch_size}) "
                f"and `sample` ({batch_size}). Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments. "
                "The batch-size provided to the sample method "
                "will prevail."
            )
        elif batch_size is None and self._batch_size is not None:
            batch_size = self._batch_size
        elif batch_size is None:
            raise RuntimeError(
                "batch_size not specified. You can specify the batch_size when "
                "constructing the replay buffer, or pass it to the sample method. "
                "Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments."
            )
        if not self._prefetch:
            ret = self._sample(batch_size)
        else:
            if len(self._prefetch_queue) == 0:
                ret = self._sample(batch_size)
            else:
                ret = self._prefetch_queue.popleft()

            while len(self._prefetch_queue) < self._prefetch_cap:
                fut = self._sample(batch_size)
                self._prefetch_queue.append(fut)

        ret[2].synchronize()
        if return_info:
            return ret[:2]
        return ret[0]

    def sync_prefetching(self):
        self.sampling_stream.synchronize()

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        self._sampler.mark_update(index)
