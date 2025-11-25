import torch.nn as nn
from pau import PAU


def init_sequential(layers, finish_with_activation=False):
    modules = []
    n = len(layers)
    for i in range(n-2):
        modules.append(nn.Linear(layers[i], layers[i+1]))
        modules.append(PAU(initial_shape='relu'))
    modules.append(nn.Linear(layers[-2], layers[-1]))
    if finish_with_activation:
        modules.append(PAU(initial_shape='relu'))
    return nn.Sequential(*modules)
