# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:13:50 2023

@author: chopi
"""

from trackmania_rl.rollout import RolloutWorker
import time


rlw = RolloutWorker()
print('RLW Created')
rlw.play_one_race(actor=None)
time.sleep(10)
rlw.play_one_race(actor=None)
time.sleep(10)
rlw.play_one_race(actor=None)