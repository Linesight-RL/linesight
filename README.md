# Trackmania AI with Reinforcement Learning
This public repository contains a copy of the reinforcement learning training code we (pb4 and [Agade](https://github.com/Agade09)) are currently developing.
This code is a Work-in-Progress, constantly evolving. Do not expect this repository to be clean/finalized/easily usable.
*Updated on June 9th 2023, synced with commit 8d8c0660bf516305f898a20359a929a59994f6e6 in our private repository*

This codebase requires the game Trackmania Nations Forever as well as [TMInterface](https://donadigo.com/tminterface/). 

### Installation
The code is known to work with Python 3.10.
- `pip install -r requirements.txt`
- Install pytorch, we use version 2 with Cuda 11.8.
- `pip install -e .`

### How to start a run
#### Generate a "virtual checkpoints" file for your map
1) Run `python ./scripts/observe_manual_run_to_extract_checkpoints.py`
2) Play once through the map, staying near the centerline of the road. The script will save a file in `./maps/map.npy` containing the coordinates of "virtual checkpoints" on the map, spaced approximately 10 meters apart.

#### Start training
1) Edit the location of `map.npy` file at the top of `./scripts/train.py` at line `zone_centers = np.load(...)`
2) Open Trackmania Interface and load the map you wish to train on. Set the game resolution to 640x480.
3) Run `python ./scripts/train.py`
4) Follow training performance via the tensorboard interface.
5) Wait a long time...

### Benchmark
It is possible to reach 2:04:91 of this map: https://tmnf.exchange/trackshow/10460245
Video: https://www.youtube.com/watch?v=p5pq2UNOEnY

### Disclaimer
In this public repository, we changed *some* training hyperparameters in the file `./trackmania_rl/misc.py`compared to our private repository. This *probably* makes training less efficient. This is done such that a minimal understanding of the principles of reinforcement learning is necessary to reproduce our results.
The actual training hyperparameters may be released at a later date. In the meantime, we will answer questions of interested individuals who want to contact us directly.
The code *should* run as-is, do not hesitate to contact us if it appears that necessary files are missing.
