# -*- coding: utf-8 -*-

from scipy.optimize import fsolve
import numpy as np
import torch

def func(x):
    return [x[0] * np.cos(x[1]) - 4,
            x[1] * x[0] - x[1] - 5]

root = fsolve(func,[0,0])

print(root)