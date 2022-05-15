import sys

import numpy as np
import torch
from torch import nn
import yaml

from itertools import combinations, product
from models.trex import TRexCost
from parser import create_env
from models.human import LowDimHuman
from models.trex import TRexCost
import os
import datetime
import matplotlib.pyplot as plt
import cv2
from bc import BCEnsemble


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)
    torch.manual_seed(123456)
    np.random.seed(123456)
    logdir = os.path.join("saved/models/BC/maze", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cfg.yaml"), "w") as f:
        yaml.dump(params, f)
    env = create_env(params["env"])
    human = LowDimHuman(env, 0.005)

    demo_trajs = []
    demo_states = []
    demo_acs = []

    for i in range(4):
        demo = human.get_demo(length=30)
        demo_states.extend(demo["obs"][:-1])
        demo_acs.extend(demo["acs"])
        demo_trajs.append([demo["obs"][:-1], demo["acs"], demo["obs"][1:]])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    states_np = np.vstack(demo_states)
    actions_np = np.vstack(demo_acs)
    shuffle_idx = np.random.permutation(len(states_np))
    states = torch.from_numpy(states_np[shuffle_idx]).to(device).float()
    actions = torch.from_numpy(actions_np[shuffle_idx]).to(device).float()

    bc = BCEnsemble(device)
    epochs = 10000
    val_losses, loss_mean_pred = bc.train(states, actions, epochs)

    plt.plot(np.arange(1, epochs + 1), loss_mean_pred, label="ensemble prediction validation loss")
    plt.plot(np.arange(1, epochs + 1), np.array(val_losses).mean(axis=0), label="mean validation loss")
    # for i in range(len(val_losses)):
    #     plt.plot(np.arange(1, epochs + 1), np.array(val_losses)[i], label=str(i))
    plt.legend()    
    plt.savefig(os.path.join(logdir, "trainbc.png"))
    plt.close()

    eval_successes = 0
    final_costs = []
    for i in range(100):
        traj, success = bc.rollout_trajectory(env, return_success=True)
        final_costs.append(env.get_expert_cost([traj[2][-1]])[0])
        eval_successes += int(success)
    final_costs.append(eval_successes)
    final_costs = np.array(final_costs)
    np.save(os.path.join(logdir, "eval_costs"), final_costs)