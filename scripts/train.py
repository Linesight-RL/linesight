import ctypes
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from trackmania_rl import misc
from trackmania_rl.multiprocess.collector_process import collector_process_fn
from trackmania_rl.multiprocess.learner_process import learner_process_fn

if __name__ == "__main__":
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    random_seed = 444
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    base_dir = Path(__file__).resolve().parents[1]
    save_dir = base_dir / "save" / misc.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare multi process utilities
    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queue = mp.Queue(misc.max_rollout_queue_size)
    model_queue = mp.Queue()

    # Start worker process
    collector_process = mp.Process(target=collector_process_fn, args=(rollout_queue, model_queue, shared_steps, base_dir, save_dir))
    collector_process.start()

    # Start learner process
    learner_process = mp.Process(target=learner_process_fn, args=(rollout_queue, model_queue, shared_steps, base_dir, save_dir))
    learner_process.start()

    collector_process.join()
    learner_process.join()
