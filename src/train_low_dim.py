import sys

import numpy as np
import torch
import wandb
import yaml

from itertools import combinations
from models.trex import TRexCost
from models.mpc import MPC
from parser import create_env
from models.human import LowDimHuman
from models.dynamics import PtModel
import os
import datetime
import matplotlib.pyplot as plt
import cv2


def plot_cost_function(cost_model, num_pts=100):
    x_bounds = [-0.3, 0.3]
    y_bounds = [-0.3, 0.3]

    states = []
    x_pts = num_pts
    y_pts = int(
        x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
    for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
        for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
            states.append([x, y])

    return cost_model(states).reshape(y_pts, x_pts).T

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    # wandb.init(project="MB-TREX", entity="yiliu77", config=params)
    logdir = os.path.join("saved/models/TREX/maze", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(logdir, exist_ok=True)
    env = create_env(params["env"])
    human = LowDimHuman(env, 0.005)
    demo_info = human.get_demo(env.horizon)
    query_states, query_actions = [], []
    # for demo_state, demo_action in zip(demo_states, demo_actions):
    #     query_states.append(demo_state[0:8])
    #     query_actions.append(demo_action[0:8])
    obs = env.reset()
    transitions = []
    for i in range(10000):

        if i % 50 == 0:
            env.reset()

        action = env.action_space.sample()
        next_obs, _, _, _ = env.step(action)
        transitions.append([obs, action, next_obs])
        obs = next_obs
    transitions = np.array(transitions)

    transition_params = params["dynamics_model"]
    # dir_name = f"./saved/models/{transition_params['type']}/{transition_params['note']}/"
    dynamics = PtModel(2, 2)
    test_loss = dynamics.train_dynamics(transitions, 25, transition_params["train_test_split"])
    print(test_loss)

    # TODO: intrinsic motivation using RND
    # TODO: down the line skill discovery methods

    # TODO: change from mean of ensemble to individual ensemble to improve diversity of CEM trajectories
    # TODO: rnd for video learning
    # TODO: add more metrics: measure variance of trex ensemble members, measure GT reward over time

    cost_model = TRexCost(lambda x, y: x, 2, params["cost_model"])
    env.visualize_rewards("initial.png", cost_model)

    agent_model = MPC(dynamics, cost_model, env, params["mpc"])
    paired_states1, paired_actions1, paired_states2, paired_actions2, labels = [], [], [], [], []
    num_success = 0
    for iteration in range(2):
        all_states = []
        state = env.reset()
        t = 0
        done = False

        cem_actions = []
        cem_states = []
        while not done:
            action, generated_actions, generated_states = agent_model.act(state)
            next_state, reward, done, info = env.step(action)
            cem_actions.append(generated_actions)
            cem_states.append(generated_states)
            # from matplotlib import pyplot as plt
            # plt.imshow(state)
            # plt.show()

            # if t % 4 == 0:
            #     query_states.append(generated_traj)
            #     query_actions.append(generated_actions)
            all_states.append(state)

            state = next_state
            t += 1

        if info['success']:
            num_success += 1
        print(f"ITER: {iteration} NUM SUCCESSES: {num_success}")
        time_step = np.random.choice(np.arange(t))
        query_actions = cem_actions[time_step]
        query_states = cem_states[time_step]


        # all_states = np.array(all_states)
        # # all_states = np.reshape(all_states, (all_states.shape[0] * all_states.shape[1], all_states.shape[2], all_states.shape[3]))
        #
        # # for states, actions in zip(query_states, query_actions):
        # #     print(states.shape, actions.shape)
        #
        # # query_states_np = np.transpose(np.array(query_states), (0, 1, 3, 4, 2))
        # query_actions_np = np.array(query_actions)

        indices = list(combinations(range(len(query_states)), 2))
        np.random.shuffle(indices)
        new_paired_states1, new_paired_actions1, new_paired_states2, new_paired_actions2, new_labels = [], [], [], [], []
        plot_queries = iteration % 5 == 0
        if plot_queries:
            fig, axs = plt.subplots(nrows=1, ncols=len(indices), figsize=(4 * len(indices), 4))
            fig.suptitle("Trajectory pairs at step " + str(time_step) + " of episode " + str(iteration))
            expert_cost = plot_cost_function(env.get_expert_cost)

        background = cv2.resize(env._get_obs(images=True), (100, 100))

        for i, index in enumerate(indices):
            new_paired_states1.append(query_states[index[0]])
            new_paired_states2.append(query_states[index[1]])
            new_paired_actions1.append(query_actions[index[0]])
            new_paired_actions2.append(query_actions[index[1]])
            label = human.query_preference(query_states[index[0]], query_states[index[1]])
            new_labels.append(label)
            if plot_queries:
                axs[i].imshow(background, extent=[0, 100, 100, 0])
                axs[i].imshow(expert_cost, alpha=0.6)

                traj1_x = query_states[index[0]][:, 0].T * 100 / 0.6 + 50
                traj1_y = query_states[index[0]][:, 1].T * 100 / 0.6 + 50
                traj2_x = query_states[index[1]][:, 0].T * 100 / 0.6 + 50
                traj2_y = query_states[index[1]][:, 1].T * 100 / 0.6 + 50
                if label == 0.5:
                    axs[i].plot(traj1_x, traj1_y, color="b")
                    axs[i].plot(traj2_x, traj2_y, color="b")
                else:
                    axs[i].plot(traj1_x, traj1_y, color="r" if label else "g")
                    axs[i].plot(traj2_x, traj2_y, color="g" if label else "r")
        if plot_queries:
            plt.savefig(os.path.join(logdir, "TREX_pairs_ep" + str(iteration) + ".png"), bbox_inches='tight')


        paired_states1 += new_paired_states1
        paired_actions1 += new_paired_actions1
        paired_states2 += new_paired_states2
        paired_actions2 += new_paired_actions2
        labels += new_labels

        cost_model.train(torch.stack(paired_states1), torch.stack(paired_actions1), torch.stack(paired_states2), torch.stack(paired_actions2), torch.tensor(labels), 1)  # TODO fix
        if iteration % 5 == 0:
            env.visualize_rewards(os.path.join(logdir, f"cost_ep{iteration}.png"), cost_model)
