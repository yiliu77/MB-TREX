import copy

import torch
from torch import nn


def polyak_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))
