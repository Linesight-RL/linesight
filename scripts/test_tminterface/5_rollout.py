# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:13:50 2023

@author: chopi
"""

import time

from matplotlib import pyplot as plt

from trackmania_rl.buffer_management import fill_buffer_from_rollout_with_n_steps_rule, get_buffer
from trackmania_rl.rollout import rollout

rv = rollout(running_speed=1, run_steps_per_action=10, max_time=10000)

buffer = get_buffer()
buffer = fill_buffer_from_rollout_with_n_steps_rule(buffer, rv, 3)


# for frame in rv['frames']:
#     plt.imshow(frame, interpolation="nearest")
#     plt.show()


# aaaaaaaaaaaa

# while True:
#     frames = rollout(running_speed=1,
#                      run_steps_per_action=10,
#                      max_time=20000)


#     print(len(frames))

#     for frame in frames:
#         plt.imshow(frame, interpolation="nearest")
#         plt.show()


# rv['simstates'][3].scene_mobil.engine.gear
