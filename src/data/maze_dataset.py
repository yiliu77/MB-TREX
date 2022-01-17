import random

import h5py
import numpy as np
from torch.utils import data


class MazeDataset(data.Dataset):
    def __init__(self, file_path, seq_length):
        print(file_path + "transition_data.hdf5")
        with h5py.File(file_path + "transition_data.hdf5", 'r') as h5_file:
            self.images = np.array(h5_file["images"])
            self.actions = np.array(h5_file["actions"])
        self.seq_length = seq_length

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        states, actions = self.images[index] / 255, self.actions[index]
        rand_index = random.randint(0, len(states) - self.seq_length)
        return states[rand_index: rand_index + self.seq_length], actions[rand_index: rand_index + self.seq_length], states, actions
