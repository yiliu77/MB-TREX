import os
import random
import sys
import time

import socket
import h5py
from itertools import combinations, product
import numpy as np
import torch
import wandb
import yaml

from models.svg import SVG
from models.trex import TREX
from models.visualmpc import VisualMPC
from models.bc import BCEnsemble
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

    wandb.init(project="MB-TREX", entity="yiliu77", config=params, name="TREX")

    env = create_env(params["env"])
    human = create_human(params["human"], env)

    cost_model = TREX(human, env.observation_space, env.action_space.shape[0], params["cost_model"])
    if not os.path.isdir("saved/{}/TREX/reward_visualization/".format(params["env"]["type"])):
        os.makedirs("saved/{}/TREX/reward_visualization/".format(params["env"]["type"]))
    env.visualize_rewards("saved/{}/TREX/reward_visualization/initial.png".format(params["env"]["type"]), cost_model)

    with h5py.File("saved/{}/data/transition_data1.hdf5".format(params["env"]["type"]), 'r') as f:
        all_states = np.array(f['images'])
        all_actions = np.array(f['actions'])
    # all_states = (0.299 * all_states[..., 0] + 0.587 * all_states[..., 1] + 0.114 * all_states[..., 2])[:, :, :, :, None]
    print(all_actions.shape, all_states.shape, np.max(all_states), np.min(all_states))
    all_states = np.transpose(all_states, (1, 0, 4, 2, 3)) / 255
    all_actions = np.transpose(all_actions, (1, 0, 2))
    print(all_actions.shape, all_states.shape, np.max(all_states), np.min(all_states))

    # test_data = create_pref_validation(all_states, human)

    video_model = SVG(env.observation_space, env.action_space.shape[0], params["video_model"])
    video_dir_name = f"./saved/{params['env']['type']}/{video_model.type}/"
    video_model.load(video_dir_name + "svg.pt")
    visualmpc = VisualMPC(video_model, cost_model.get_value, params["env"]["horizon"], env, params["visualmpc"])
    visualmpc.normalization.update(torch.as_tensor(np.transpose(all_states[:, 0, ...], (0, 2, 3, 1))).cuda())

    all_query_states = np.empty((params["visualmpc"]["future_frames"], 0, all_states.shape[2], all_states.shape[3], all_states.shape[4]))
    all_query_actions = np.empty((params["visualmpc"]["future_frames"], 0, all_actions.shape[2]))

    # # ############ PRETRAINING ############
    # bc = BCEnsemble(device)
    # demo_trajs = []
    # demo_states = []
    # demo_acs = []
    #
    # for i in range(params["cost_model"]["offline_demos"]):
    #     demo = human.get_demo()
    #     demo_states.extend(demo["obs"][:-1])
    #     demo_acs.extend(demo["acs"])
    #     demo_trajs.append([demo["obs"][:-1], demo["acs"], demo["obs"][1:]])
    #
    # bc_trajs = []
    # eps = np.linspace(0, 0.75, num=4)
    #
    # if params["cost_model"]["bc_comp_mode"] == "all":
    #     start = env.reset()
    #     costs = [[] for _ in range(len(eps))]
    #     for i, e in enumerate(eps):
    #         bc_trajs.append([])
    #         for k in range(5):
    #             states, acs = [], []
    #             s = start
    #             traj = []
    #             for t in range(env.horizon):
    #                 states.append(s)
    #                 state_torch = torch.from_numpy(s).float().to(device)
    #                 if np.random.random() < e:
    #                     a = torch.from_numpy(env.action_space.sample()).float().to(device)
    #                 else:
    #                     a = bc.predict(state_torch).squeeze()
    #                 acs.append(a.detach().cpu().squeeze().numpy())
    #                 next_s = video_model.predict_states(torch.unsqueeze(torch.cat((state_torch, a)), dim=0))
    #                 s = next_s.detach().cpu().squeeze().numpy()
    #             traj = [np.array(states), np.array(acs)]
    #             bc_trajs[-1].append(traj)
    #             costs[i].append(env.get_expert_cost([traj[0][-1]])[0])
    #
    #     costs = np.array(costs)
    #     avg_costs = np.mean(costs, axis=1)
    #     std_costs = np.std(costs, axis=1)
    #     states1 = []
    #     acs1 = []
    #     states2 = []
    #     acs2 = []
    #     bc_labels = []
    #
    #     for traj_set_1, traj_set_2 in combinations([demo_trajs] + bc_trajs, 2):
    #         for traj1, traj2 in product(traj_set_1, traj_set_2):
    #             states1.append(np.array(traj1)[0, :])
    #             acs1.append(np.array(traj1)[1, :])
    #             states2.append(np.array(traj2)[0, :])
    #             acs2.append(np.array(traj2)[1, :])
    #             bc_labels.append(0)

    # ############ TRAINING LOOP ############
    mpc_states, mpc_actions = [], []
    for iteration in range(1000):
        traj_states = []
        avg_rewards = []
        for _ in range(8):  # TODO
            wandb.log({# "VisualMPC/RND Visualization": visualize_rnd(env, visualmpc.rnd, visualmpc.rnd_target, visualmpc.normalization),
                       "VisualMPC/Normalization": wandb.Image(visualmpc.normalization.mean.cpu().numpy()[0])})
            states, actions = [], []
            state, done, t = env.reset()[0], False, 0
            all_query_states_list, all_query_actions_list = [], []
            while not done:
                start = time.time()
                action, query_states_list, query_actions_list = visualmpc.act(state, t)
                next_state, reward, done, info = env.step(action)
                print(t, action, reward, time.time() - start)
                states.append(state)
                actions.append(action)
                avg_rewards.append(reward)

                state = next_state
                if t % (params["visualmpc"]["future_frames"] // 2) == 0:
                    all_query_states_list.append((query_states_list, t))
                    all_query_actions_list.append((query_actions_list, t))
                t += 1

            for (query_states_list, curr_t1), (query_actions_list, curr_t2) in zip(all_query_states_list, all_query_actions_list):
                if curr_t1 + params["visualmpc"]["future_frames"] < t and curr_t2 + params["visualmpc"]["future_frames"] < t:
                    for query_states, query_actions in zip(query_states_list, query_actions_list):
                        all_query_states = np.concatenate([all_query_states, query_states[:, None, :, :, :]], axis=1)
                        all_query_actions = np.concatenate([all_query_actions, query_actions[:, None, :]], axis=1)

            states = np.array(states)
            all_trajectory_imgs = wandb.Image(np.reshape(states, (states.shape[0] * states.shape[1], states.shape[2], states.shape[3])), caption="Sequences")
            wandb.log({"All Trajectories": all_trajectory_imgs})
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
        cost_model.train(all_query_states, all_query_actions, 60, test_data=None)
        print("train total", time.time() - start)
        start = time.time()
        env.visualize_rewards("saved/{}/TREX/reward_visualization/image{}.png".format(params["env"]["type"], iteration), cost_model)
        print("visualize", time.time() - start)

        cost_model.save("saved/{}/TREX/save{}.pt".format(params["env"]["type"], iteration))
