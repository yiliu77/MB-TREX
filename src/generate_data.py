import os
import sys

import h5py
import numpy as np
import yaml
from yaml import CLoader

import wandb
from models.sac import ContSAC
from utils.parser import create_env


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == "__main__":
    with open(sys.argv[1], "r") as stream:
        params = yaml.load(stream, CLoader)
    data_id = sys.argv[2]

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    env = create_env(params["env"])
    action_dim = env.action_space.shape[0]

    model = ContSAC(action_dim, env, "cuda", ent_adj=True)
    all_states, all_states_lowdim, all_actions, all_lengths = model.train(500, deterministic=False)

    if not os.path.isdir("saved/{}/data/".format(params["env"]["type"])):
        os.makedirs("saved/{}/data/".format(params["env"]["type"]))

    with h5py.File("saved/{}/data/transition_data{}.hdf5".format(params["env"]["type"], data_id), 'w') as f:
        f.create_dataset('images', data=all_states)
        f.create_dataset('actions', data=all_actions)
        f.create_dataset('lengths', data=all_lengths)
        f.close()

    with h5py.File("saved/{}/data/transition_data_lowdim{}.hdf5".format(params["env"]["type"], data_id), 'w') as f:
        f.create_dataset('images', data=all_states_lowdim)
        f.create_dataset('actions', data=all_actions)
        f.create_dataset('lengths', data=all_lengths)
        f.close()