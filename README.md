# Linesight AI: Playing Trackmania with Reinforcement Learning

## Disclaimer

In this public repository, we have intentionally changed some training hyperparameters in the file `./trackmania_rl/misc.py` compared to our private repository to encourage a better understanding of reinforcement learning principles. Training may be inefficient or impossible with the current hyperparameters, we haven't even tested.

The actual training hyperparameters may be released at a later date. In the meantime, feel free to contact us if you have questions or encounter any issues with the code.

To actively participate and share your progress with this code, please join the TMInterface Discord community (https://discord.gg/tD4rarRYpj) first. You can then post your updates in the 'Issues' section on Github or join the conversation in our dedicated thread on the TMInterface Discord (https://discord.com/channels/847108820479770686/1150816026028675133)

**Please note:** This project is a research work-in-progress and may not receive active support for setup or usage.

Welcome to our Trackmania AI with Reinforcement Learning project. This repository contains the reinforcement learning training code developed by [pb4](https://github.com/pb4git) and [Agade](https://github.com/Agade09).

**Last update:** *Updated on June 9th, 2023 ; synced with commit 8d8c0660bf516305f898a20359a929a59994f6e6 in our private repository*

Please note that this codebase is constantly evolving, and it may not be clean, finalized, or easily usable. We intend to open up our code with all training hyperparameters for the wider community in the future, but for now, it's shared as-is for code reading purposes.

## Prerequisites

Before you get started, ensure you have the following prerequisites:

- An NVIDIA graphics card
- Trackmania Nations Forever
- [TMInterface](https://donadigo.com/tminterface/) (Version < 2.0.0). [Download TMInterface 1.4.3](https://donadigo.com/files/TMInterface/TMInterface_1.4.3_Setup.exe).
- Python 3.10 [Download Python 3.10](https://www.python.org/downloads/release/python-3100/)
- PyTorch we use version 2 with Cuda 11.8
    -This project requires PyTorch. You can install it using the following command. Please note that the installation URL may change, so if the command doesn't work, check the [official PyTorch website](https://pytorch.org/) for the latest instructions:
    ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Installation

To set up the project, follow these steps:

1. Clone this repository.
2. Install the required Python packages: `pip install -r requirements.txt`
3. Install the project as an editable package: `pip install -e .`

## Getting Started

### Generating "Virtual Checkpoints" for Your Map

To begin a run, follow these steps:

1. Run the script to generate "virtual checkpoints" for your map: `python ./scripts/observe_manual_run_to_extract_checkpoints.py`.
2. Play through the map, staying near the centerline of the road. The script will save a file in `./maps/map.npy` containing the coordinates of "virtual checkpoints" spaced approximately 10 meters apart.

### Starting Training

1. Edit the location of the `map.npy` file at the top of `./scripts/train.py`, specifically at line `zone_centers = np.load(...)`.
2. Open Trackmania Interface and load the map you wish to train on, setting the game resolution to 640x480.
3. Run the training script: `python ./scripts/train.py`.
4. Monitor training performance via the TensorBoard interface.
5. Be patient; training may take a significant amount of time.

## Benchmark

We have achieved a lap time of 2:04:91 on this [map](https://tmnf.exchange/trackshow/10460245). You can watch the [video](https://www.youtube.com/watch?v=p5pq2UNOEnY) for a demonstration.

## Acknowledgments

We would like to acknowledge the contributions of the community to this project. In particular, we want to highlight [ausstein's fork](https://github.com/ausstein/trackmania_rl_public), which extends the functionality of this project with support for multiple instances, additional parameters, and important fixes. Their work has been valuable to the project's development and versatility.
