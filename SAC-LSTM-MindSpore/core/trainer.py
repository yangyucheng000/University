import os.path
import datetime
import numpy as np
from core.utils import preprocess
import x2ms_adapter
import mindspore


def train(model, ims, real_input_flag, configs, itr):
    cost = model(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = mindspore.numpy.flip(ims, axis=1).copy()
        cost += model(ims_rev, real_input_flag)
        cost = cost / 2

    return cost


