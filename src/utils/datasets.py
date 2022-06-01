import random

import h5py
import numpy as np
from torch.utils import data


class SimplePointDataset(data.Dataset):
    def __init__(self, file_path, n_past, n_future):
        self.f = h5py.File(file_path, 'r')
        self.n_past = n_past
        self.n_future = n_future
        self.seq_length = self.n_past + self.n_future

    def __len__(self):
        return len(self.f["images"])

    def __getitem__(self, index):
        states, actions, length = self.f["images"][index] / 255, self.f["actions"][index], self.f["lengths"][index]
        states = np.concatenate([states[0][None, ...]] * self.n_past + [states[1:]], axis=0)
        actions = np.concatenate([np.zeros_like(actions[0])[None, ...]] * (self.n_past - 1) + [actions], axis=0)
        rand_index = random.randint(0, length - (self.n_future + 1))
        return states[rand_index: rand_index + self.seq_length], actions[rand_index: rand_index + self.seq_length]
