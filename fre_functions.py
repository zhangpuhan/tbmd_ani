""" This file contains frequently used functions """

import torch
import math
from constant import CUT_OFF


def f_c(rs_list):
    step_1 = torch.div(rs_list, CUT_OFF)
    step_2 = torch.mul(step_1, math.pi)
    step_3 = torch.cos(step_2)
    step_4 = torch.mul(step_3, 0.5)
    step_5 = torch.add(step_4, 0.5)
    return step_5


def exponential_map(manipulate_tensor, radial_sample_comb):
    return torch.exp((-(manipulate_tensor - radial_sample_comb[:, 1:]) ** 2.0).mul(radial_sample_comb[:, :1]))

