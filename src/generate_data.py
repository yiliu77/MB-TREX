import sys

import os
import h5py
import numpy as np
import tqdm
import wandb
import yaml
from yaml import CLoader

from parser import create_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from models.sac import ContSAC

def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == "__main__":
    with open(sys.argv[1], "r") as stream:
        params = yaml.load(stream, CLoader)

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    env = create_env(params["env"])
    action_dim = env.action_space.shape[0]

    model = ContSAC(action_dim, env, "cuda", ent_adj=True)
    all_states, all_actions = model.train(1000, deterministic=False)

    if not os.path.isdir("saved/{}".format(params["env"]["type"])):
        os.makedirs("saved/{}".format(params["env"]["type"]))

    with h5py.File("saved/{}/transition_data1.hdf5".format(params["env"]["type"]), 'w') as f:
        f.create_dataset('images', data=all_states)
        f.create_dataset('actions', data=all_actions)
        f.close()

    # num_parallel = params["num_parallel"]
    # envs = SubprocVecEnv([lambda: create_env(params["env"]) for _ in range(num_parallel)])
    #
    # all_states, all_actions = [], []
    # num_games = 0
    # with tqdm.tqdm(range(params["num_games"])) as counter:
    #     while num_games < params["num_games"]:
    #         states, actions = [], []
    #         done = [False for _ in range(num_parallel)]
    #         # noinspection PyRedeclaration
    #         state = envs.reset()
    #         while not np.any(done):
    #             action = [envs.action_space.sample() for _ in range(num_parallel)]
    #             # while last_action is not None and not (-np.pi / 4 < angle_between(action, last_action) < np.pi / 4):
    #             #     action = env.action_space.sample()
    #             next_state, reward, done, info = envs.step(action)
    #
    #             states.append(state)
    #             actions.append(action)
    #             state = next_state
    #
    #         states = np.array(states)
    #         actions = np.array(actions)
    #         if len(states) != params["env"]["horizon"]:
    #             continue
    #
    #         num_games += num_parallel
    #         for i in range(num_parallel):
    #             all_states.append(states[:, i, :, :, :])
    #             all_actions.append(actions[:, i, :])
    #         counter.update(num_parallel)
    #
    # all_states = np.stack(all_states, axis=0)
    # all_actions = np.stack(all_actions, axis=0)
    #
    # print(all_states.shape, all_actions.shape)
    #
    # if not os.path.isdir("saved/{}".format(params["env"]["type"])):
    #     os.makedirs("saved/{}".format(params["env"]["type"]))
    #
    #
    # with h5py.File("saved/{}/transition_data_{}.hdf5".format(params["env"]["type"], params["note"]), 'w') as f:
    #     f.create_dataset('images', data=all_states)
    #     f.create_dataset('actions', data=all_actions)
    #     f.close()
