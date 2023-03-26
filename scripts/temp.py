# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:04:27 2023

@author: chopi
"""


from trackmania_rl.agents import noisy_iqn
from trackmania_rl import misc
import torch
import numpy as np

a = noisy_iqn.Agent(misc.float_input_dim, misc.float_hidden_dim).to("cuda")
a2 = noisy_iqn.Agent(misc.float_input_dim, misc.float_hidden_dim).to("cuda")
optimizer = torch.optim.RAdam(a.parameters(), lr=misc.learning_rate)
scaler = torch.cuda.amp.GradScaler()

import joblib
img = joblib.load("img.joblib")
floats = joblib.load("float.joblib")

buffer = joblib.load("buffer.joblib")


# img_tensor = torch.tensor(
#     np.array([img for i in range(3)]), requires_grad=True, dtype=torch.float32
# ).to("cuda", memory_format=torch.channels_last, non_blocking=True)

# float_tensor = torch.tensor(
#     np.array([floats for i in range(3)]), dtype=torch.float32, requires_grad=True
# ).to("cuda", non_blocking=True)

# a(img_tensor, float_tensor, 8, True)

noisy_iqn.learn_on_batch(a, a2, optimizer, scaler, buffer)