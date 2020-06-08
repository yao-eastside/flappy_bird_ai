import json
import os
import random
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import keras
import numpy as np
import skimage.color
import skimage.exposure
import skimage.transform
from keras import layers, models

from game import wrapped_flappy_bird as game


def logging(mode, t, time0, network, observe, epsilon, action_index, r_t, Q_sa, loss, total_loss, total_explore):
    # save progress every 10000 iterations
    # if t % 10_000 == 0:
    if t % 1_000 == 0:
        if mode == 'train':
            print("saving the model...")
            network.save(f"model-{t:08d}.h5", overwrite=True)

    # print info
    state = "train"
    if t <= observe:
        state = "observe"
    elif t > observe and t <= observe + total_explore:
        state = "explore"

    now = datetime.now()
    print(
        now,
        "timespent:", str(timedelta(seconds=time.time() - time0)),
        "TIMESTEP:", t,
        "STATE:", state,
        "EPSILON:", epsilon,
        "ACTION:", action_index,
        "REWARD:", r_t,
        "Q_MAX:" , np.max(Q_sa),
        "Loss:", loss,
        "avgloss:", total_loss/(t+1)
    )

