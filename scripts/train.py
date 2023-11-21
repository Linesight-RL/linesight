import ctypes
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from art import tprint
from torch.multiprocessing import Lock

from trackmania_rl import misc
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl.multiprocess.collector_process import collector_process_fn
from trackmania_rl.multiprocess.learner_process import learner_process_fn

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
random_seed = 444
torch.cuda.manual_seed_all(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


def signal_handler(sig, frame):
    print("Received SIGINT signal. Killing all open Trackmania instances.")
    clear_tm_instances()

    for child in mp.active_children():
        child.kill()

    tprint("Bye bye!", font="tarty1")
    sys.exit()


def clear_tm_instances():
    if misc.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill TmForever.exe /F")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    clear_tm_instances()

    print("Run:")
    print("\n" * 2)
    tprint(misc.run_name, font="tarty4")
    print("\n" * 2)
    tprint("Linesight", font="tarty1")
    print("\n" * 2)
    print("Training is starting!")

    base_dir = Path(__file__).resolve().parents[1]
    save_dir = base_dir / "save" / misc.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = base_dir / "tensorboard" / misc.run_name

    # Copy Angelscript plugin to TMInterface dir
    shutil.copyfile(
        base_dir / "trackmania_rl" / "tmi_interaction" / "Python_Link.as",
        misc.target_python_link_path,
    )

    # Prepare multi process utilities
    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queues = [mp.Queue(misc.max_rollout_queue_size) for _ in range(misc.gpu_collectors_count)]
    shared_network_lock = Lock()
    game_spawning_lock = Lock()
    _, uncompiled_shared_network = make_untrained_iqn_network(jit=misc.use_jit)
    uncompiled_shared_network.share_memory()

    # Start learner process
    learner_process = mp.Process(
        target=learner_process_fn,
        args=(rollout_queues, uncompiled_shared_network, shared_network_lock, shared_steps, base_dir, save_dir, tensorboard_dir),
    )
    learner_process.start()

    time.sleep(1)

    # Start worker process
    collector_processes = [
        mp.Process(
            target=collector_process_fn,
            args=(
                rollout_queue,
                uncompiled_shared_network,
                shared_network_lock,
                game_spawning_lock,
                shared_steps,
                base_dir,
                save_dir,
                misc.base_tmi_port + process_number,
            ),
        )
        for rollout_queue, process_number in zip(rollout_queues, range(misc.gpu_collectors_count))
    ]
    for collector_process in collector_processes:
        collector_process.start()
        time.sleep(5)

    for collector_process in collector_processes:
        collector_process.join()
    learner_process.join()
