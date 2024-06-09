Contributions
=============

We welcome contributions to the Linesight Trackmania AI project. If you're interested in contributing, please follow the guidelines below to ensure that your contributions are clear, well-formatted, and thoroughly tested.

Contribution opportunities
--------------------------

We list below improvements that can likely be made to the project.

**Documentation**

General documentation improvements for users or developers.

**Respawns**

All rollouts start from the beginning of a track. On challenging or high-risk tracks, the agent's memory buffer may contain limited transitions for the end of the track, which makes it difficult for the agent to learn evenly across all sections of the track.

This is likely a difficult contribution as it involves modifications in several parts of the repository at once (rollouts & statistics tracking). It is yet unclear how to best resolve this issue.

One step further in difficulty, one might refactor the agent-environment interactions to follow the Gymnasium API.

**Linux performance**

On Linux, the game runs much slower than windows. You can see this if you display FPS by pressing - then open the console with ` and type "set speed 50". Your FPS should reduce by orders of magnitude. We suspect this is some kind of bottleneck in wineserver but we do not know.

**Game reboot**

We observe that game instances slow down over time. As a mitigation, we have implemented code to restart the game instances every config.game_reboot_interval.

- There is a rare issue where a worker will crash when restarting the game instance.
- config.game_camera_number allows training an agent in cam 1/2/3. There is an issue where the cam change may not happen after a worker restarts the game instance. Due to this bug it is advised to train with game_camera_number 2 (the default cam).

**Neural network architecture**

We have performed limited tests to improve neural network architecture. There are opportunities to improve the neural network's vision head (currently DQN-style, could test IMPALA-style or other architectures). There are also opportunities to improve the virtual checkpoints inputs (currently a basic Dense architecture which does not benefit from the known sequential structure of virtual checkpoints).

One may also try out largely different neural network architectures and/or input features.

**Reinforcement learning algorithm**

This project implements a modified version of the IQN algorithm. One may also try to implement any other compatible reinforcement learning algorithm.

**"inputs_to_gbx" script**

We use a script ``scripts/tools/video_stuff/inputs_to_gbx.py`` to convert a batch of ``xxxx.inputs`` files into corresponding ``xxxx.Replay.Gbx`` files.  It works on Agade's computer but not on pb4's computer. The script was put together hastily, we would welcome an improved version.

**Methods to train on varied & diverse tracks**

- Hide useless visual features (flags, track decor, ...) with z-buffer masking *(done with ReShade with TMI1.4, but screenshots are now made BEFORE ReShade shaders are applied in TMI2)*
- Replace game textures with normal maps or depth maps *(done with ReShade with TMI1.4, but screenshots are now made BEFORE ReShade shaders are applied in TMI2)*
- Implement raycasting
- Implement an automatic track generator, such that the AI can train on an infinite number of maps. As a start, one may restrict this to flat "map5-style" tracks.
- anything else...

Guidelines
----------

- Before you start working on a contribution, please open an issue to discuss your proposed changes with the project maintainers. This will help avoid duplicate work and ensure that your contribution aligns with the project's goals.

- Please include a brief description of your changes in the pull request description. This will help us review your changes more quickly and efficiently.

- All code contributions must be formatted with ``ruff check --select I --fix . ; ruff format .``. This will ensure that your code is consistent with the project's style guidelines and easy to read.

- Contributions should be broken down into meaningful, self-contained chunks. Please avoid combining unrelated changes, such as refactoring and algorithmic improvements, in the same pull request. This will make it easier to review and test your changes.

- For contributions that involve algorithmic changes, we expect contributors to provide multiple runs that demonstrate both the absence of performance regressions and the presence of the expected improvements.
