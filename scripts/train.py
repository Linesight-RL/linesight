import shutil
from pathlib import Path


def copy_configuration_file():
    # At the beginning of training, parameters are read from misc.py.
    # They are then copied from misc.py to misc_reloaded_copy.py.
    # During training, misc_reloaded_copy.py will be reloaded at regular time intervals.
    #
    # Training parameters modifications made during training in misc_reloaded_copy.py will be applied on the fly
    # without losing the existing content of the replay buffer.
    #
    # The content of main.py may be modified after starting a run: it will have no effect on the ongoing run.
    # This setup provides the possibility to:
    #   1) Modify training parameters on the fly
    #   2) Continue to code and modify misc.py without impacting an ongoing run.

    base_dir = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        base_dir / "config_files" / "misc.py",
        base_dir / "config_files" / "misc_copy.py",
    )


if __name__ == "__main__":
    copy_configuration_file()

# =======================================================================================================================
# Actual start of train.py, after copying misc.py
# =======================================================================================================================

import ctypes
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from art import tprint
from torch.multiprocessing import Lock

from config_files import misc_copy
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
    if misc_copy.is_linux:
        os.system("pkill -9 TmForever.exe")
    else:
        os.system("taskkill TmForever.exe /F")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    clear_tm_instances()

    base_dir = Path(__file__).resolve().parents[1]
    save_dir = base_dir / "save" / misc_copy.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_base_dir = base_dir / "tensorboard"

    # Copy Angelscript plugin to TMInterface dir
    shutil.copyfile(
        base_dir / "trackmania_rl" / "tmi_interaction" / "Python_Link.as",
        misc_copy.target_python_link_path,
    )

    print("Run:\n\n")
    tprint(misc_copy.run_name, font="tarty4")
    print("\n" * 2)
    tprint("Linesight", font="tarty1")
    print("\n" * 2)
    print("Training is starting!")

    if misc_copy.is_linux:
        os.system(f"chmod +x {misc_copy.linux_launch_game_path}")

    # Prepare multi process utilities
    shared_steps = mp.Value(ctypes.c_int64)
    shared_steps.value = 0
    rollout_queues = [mp.Queue(misc_copy.max_rollout_queue_size) for _ in range(misc_copy.gpu_collectors_count)]
    shared_network_lock = Lock()
    game_spawning_lock = Lock()
    _, uncompiled_shared_network = make_untrained_iqn_network(jit=misc_copy.use_jit)
    uncompiled_shared_network.share_memory()

    # Start learner process
    learner_process = mp.Process(
        target=learner_process_fn,
        args=(
            rollout_queues,
            uncompiled_shared_network,
            shared_network_lock,
            shared_steps,
            base_dir,
            save_dir,
            tensorboard_base_dir,
        ),
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
                misc_copy.base_tmi_port + process_number,
            ),
        )
        for rollout_queue, process_number in zip(rollout_queues, range(misc_copy.gpu_collectors_count))
    ]
    for collector_process in collector_processes:
        collector_process.start()

    for collector_process in collector_processes:
        collector_process.join()
    learner_process.join()
