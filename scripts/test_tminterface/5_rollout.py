# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:13:50 2023

@author: chopi
"""

from trackmania_rl.rollout import rollout
import time

while True:
    frames = rollout(running_speed=1,
                     run_steps_per_action=10,
                     max_time=20000)


    from matplotlib import pyplot as plt
    
    print(len(frames))
    
    for frame in frames:
        plt.imshow(frame, interpolation="nearest")
        plt.show()
        

