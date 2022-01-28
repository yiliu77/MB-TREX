import random
import os

import h5py
import numpy as np
from torch.utils import data


class MazeDataset(data.Dataset):
    def __init__(self, file_path, seq_length):
        self.f = h5py.File(file_path, 'r')
        self.seq_length = seq_length

    def __len__(self):
        return len(self.f["images"])

    def __getitem__(self, index):
        states, actions = self.f["images"][index] / 255, self.f["actions"][index]
        rand_index = random.randint(0, len(states) - self.seq_length)
        return states[rand_index: rand_index + self.seq_length], actions[rand_index: rand_index + self.seq_length], \
               states, actions
