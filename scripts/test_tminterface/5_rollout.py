# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:13:50 2023

@author: chopi
"""

from trackmania_rl.rollout import RolloutWorker
import time


rlw = RolloutWorker(running_speed=1)
print('RLW Created')
rlw.play_one_race(actor=None)
# time.sleep(0.01)
# rlw.play_one_race(actor=None)
# time.sleep(0.01)
# rlw.play_one_race(actor=None)


# from PIL import Image
# Image.fromarray(rlw.screenshots[0]).show()


from matplotlib import pyplot as plt

for frame in rlw.screenshots:
    plt.imshow(frame, interpolation="nearest")
    plt.show()