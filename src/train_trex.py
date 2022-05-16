import os
import random
import sys
import time

import socket
import h5py
import numpy as np
import torch
import wandb
import yaml

from models.svg import SVG
from models.trex import TREX
from models.visualmpc import VisualMPC
from utils.input_utils import create_pref_validation
from utils.parser import create_env, create_human


def visualize_rnd(env, rnd, rnd_target, normalization):
    with torch.no_grad():
        x_bounds = [-0.3, 0.3]
        y_bounds = [-0.3, 0.3]

        obs_states, actions = [], []
        x_pts = 64
        y_pts = 64
        for y in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for x in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                obs = env.reset(pos=(x, y))
                obs_states.append(obs)
                # TODO check this add rew normalization

        obs_states = np.array(obs_states) / 255
        obs_states = ((obs_states - normalization.mean) / np.sqrt(normalization.var)).clip(-8, 8)
        obs_states = torch.as_tensor(obs_states).float().to("cuda")
        costs = torch.mean(torch.square(rnd_target(obs_states) - rnd(obs_states)), dim=1).detach().cpu().numpy()
        grid = (costs.reshape(y_pts, x_pts) - np.min(costs)) / (np.max(costs) - np.min(costs))
        grid_image = wandb.Image(grid[:, :, None] * 255 * 0.8 + 0.2 * env.reset(), caption="RND")
        return grid_image


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    env = create_env(params["env"])
    human = create_human(params["human"], env)

    cost_model = TREX(human, env.observation_space, env.action_space.shape[0], params["cost_model"])
    if not os.path.isdir("saved/{}/TREX/reward_visualization/".format(params["env"]["type"])):
        os.makedirs("saved/{}/TREX/reward_visualization/".format(params["env"]["type"]))
    env.visualize_rewards("saved/{}/TREX/reward_visualization/initial.png".format(params["env"]["type"]), cost_model)

    with h5py.File("saved/{}/data/transition_data6.hdf5".format(params["env"]["type"]), 'r') as f:
        all_states = np.array(f['images'])
        all_actions = np.array(f['actions'])
    print(all_actions.shape, all_states.shape, np.max(all_states), np.min(all_states))
    all_states = np.transpose(all_states, (1, 0, 4, 2, 3)) / 255
    all_actions = np.transpose(all_actions, (1, 0, 2))
    print(all_actions.shape, all_states.shape, np.max(all_states), np.min(all_states))

    test_data = create_pref_validation(all_states, human)

    video_model = SVG(env.observation_space, env.action_space.shape[0], params["video_model"])
    video_dir_name = f"./saved/{params['env']['type']}/{video_model.type}/"
    video_model.load(video_dir_name + "svg.pt")
    visualmpc = VisualMPC(video_model, cost_model.get_value, params["env"]["horizon"], env, params["visualmpc"])
    visualmpc.normalization.update(np.transpose(all_states[:, 0, ...], (0, 2, 3, 1)))

    all_query_states = np.empty((params["visualmpc"]["future_frames"], 0, all_states.shape[2], all_states.shape[3], all_states.shape[4]))
    mpc_states, mpc_actions = [], []
    for iteration in range(1000):
        traj_states = []
        avg_rewards = []
        for _ in range(3):
            wandb.log({"VisualMPC/RND Visualization": visualize_rnd(env, visualmpc.rnd, visualmpc.rnd_target, visualmpc.normalization),
                       "VisualMPC/Normalization": wandb.Image(visualmpc.normalization.mean[0])})
            states, actions = [], []
            state, done, t = env.reset(), False, 0
            while not done:
                action, query_states_list = visualmpc.act(state, t)
                next_state, reward, done, info = env.step(action)
                print(action, t, reward)
                states.append(state)
                actions.append(action)
                avg_rewards.append(reward)

                state = next_state
                t += 1
                if t % params["visualmpc"]["future_frames"] == 0:
                    for query_states in query_states_list:
                        all_query_states = np.concatenate([all_query_states, query_states[:, None, :, :, :]], axis=1)
            traj_states.append(np.max(np.stack(states), axis=0))
            mpc_states.append(states)
            mpc_actions.append(actions)
        avg_states = wandb.Image(np.mean(np.stack(traj_states, axis=0), axis=0), caption="Sequences")
        max_states = wandb.Image(np.max(np.stack(traj_states, axis=0), axis=0), caption="Sequences")
        wandb.log({"Mean Trajectories": avg_states, "Max Trajectories": max_states, "Avg Rewards": np.mean(avg_rewards)})

        # for _ in range(50):
        #     train_mpc_indices = np.random.choice(range(len(mpc_states)), size=min(len(mpc_states), params["video_model"]["batch_size"] // 2), replace=False)
        #     train_all_indices = np.random.choice(all_states.shape[1], size=params["video_model"]["batch_size"] - len(train_mpc_indices), replace=False)
        #     time_index = random.randint(0, all_states.shape[0] - (params["video_model"]["n_past"] + params["video_model"]["n_future"]))
        #
        #     train_video_states = np.array(mpc_states)[train_mpc_indices] / 255
        #     train_video_states = np.concatenate([train_video_states, all_states[:, train_all_indices, ...].transpose((1, 0, 3, 4, 2))], axis=0)
        #     print(train_video_states.shape, time_index, all_states.shape)
        #     train_video_states = train_video_states[:, time_index:, ...]
        #     print(train_video_states.shape, time_index, all_states.shape)
        #     train_video_actions = np.array(mpc_actions)[train_mpc_indices]
        #     train_video_actions = np.concatenate([train_video_actions, all_actions[:, train_all_indices, ...].transpose((1, 0, 2))], axis=0)
        #     print(train_video_actions.shape, time_index, all_actions.shape)
        #     train_video_actions = train_video_actions[:, time_index:, ...]
        #     print(train_video_actions.shape, time_index, all_actions.shape)
        #     video_model.train_step(torch.as_tensor(train_video_states), torch.as_tensor(train_video_actions))

        start = time.time()
        cost_model.train(all_query_states, 50, test_data=test_data)
        print("train total", time.time() - start)
        start = time.time()
        env.visualize_rewards("saved/{}/TREX/reward_visualization/image{}.png".format(params["env"]["type"], iteration), cost_model)
        print("visualize", time.time() - start)

        cost_model.save("saved/{}/TREX/save.pt".format(params["env"]["type"]))
