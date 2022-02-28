import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.optim import Adam
import itertools
from models.sac import RunningMeanStd, RND, RNDTarget


class CostNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def get_cost(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        r = self.fc4(x)
        return r

    # N x statedim
    def cum_return(self, traj):
        x = F.relu(self.fc1(traj))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        r = self.fc4(x)
        sum_rewards = torch.sum(r)
        sum_abs_rewards = torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def get_abs_costs(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        r = self.fc4(x)
        return torch.abs(r)

    def forward(self, traj_i, traj_j):
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j


class TRexCost:
    def __init__(self, encoder, g_dim, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.params = params

        self.cost_models = [CostNetwork(g_dim, params["hidden_size"]).to(device=self.device) for _ in range(params["ensemble_size"])]
        self.cost_opts = [Adam(self.cost_models[i].parameters(), lr=params["lr"]) for i in range(len(self.cost_models))]

    def train(self, all_states1, all_actions1, all_states2, all_actions2, all_labels, num_epochs):
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for i in range(len(all_states1)):
                states1, actions1, states2, actions2 = all_states1[i], all_actions1[i], all_states2[i], all_actions2[i]
                states1 = torch.as_tensor(np.array(states1)).float().to(self.device)
                actions1 = torch.as_tensor(np.array(actions1)).float().to(self.device)
                states2 = torch.as_tensor(np.array(states2)).float().to(self.device)
                actions2 = torch.as_tensor(np.array(actions2)).float().to(self.device)

                states1 = states1.permute((0, 3, 1, 2))
                states2 = states2.permute((0, 3, 1, 2))  # TODO standardize this
                encodings1 = self.encoder(states1, actions1).detach()
                encodings2 = self.encoder(states2, actions2).detach()

                for j in range(len(self.cost_models)):
                    if all_labels[i] != 0.5:
                        outputs, abs_costs = self.cost_models[j](encodings1, encodings2)
                        outputs = outputs.unsqueeze(0)

                        train_loss = loss_fn(outputs, 1 - torch.as_tensor(all_labels[i]).long().to(self.device).unsqueeze(0)) + self.params["cost_reg"] * abs_costs

                        self.cost_opts[j].zero_grad()
                        train_loss.backward()
                        self.cost_opts[j].step()

    def get_value(self, states, actions):
        with torch.no_grad():
            enc_states = self.encoder(states, actions)
            abs_costs = torch.mean(torch.cat([self.cost_models[i].get_cost(enc_states) for i in range(len(self.cost_models))], dim=1), dim=1)
        return abs_costs

    def get_queries(self, query_states, query_actions, num_queries):
        enc_states = self.encoder(query_states, query_actions)
        rewards = torch.cat([self.cost_models[i].get_cost(enc_states) for i in range(len(self.cost_models))], dim=2)
        indices = itertools.combinations(range(query_states), 2)
        reward_pair1, reward_pair2 = rewards[indices[0]], rewards[indices[1]]
        probs = 1 / (1 + torch.exp(-(reward_pair1 - reward_pair2)))

        avg_probs = torch.mean(probs, dim=1)
        avg_entropy = -np.sum(avg_probs * torch.log2(avg_probs))
        ind_entropy = -np.sum(torch.cat([probs[i] * np.log2(probs[i]) for i in range(len(self.cost_models))]))
        return torch.argmax()

    def plot(self, ep):
        print("Plotting learned cost: ", ep)
        if self.env_name == 'maze':
            x_bounds = [-0.3, 0.3]
            y_bounds = [-0.3, 0.3]
        elif self.env_name == 'simplepointbot0':
            x_bounds = [-80, 20]
            y_bounds = [-10, 10]
        elif self.env_name == 'simplepointbot1':
            x_bounds = [-75, 25]
            y_bounds = [-75, 25]
        elif self.env_name == 'image_maze':
            x_bounds = [-0.05, 0.25]
            y_bounds = [-0.05, 0.25]
        else:
            raise NotImplementedError("Plotting unsupported for this envs")

        states = []
        if self.images:
            x_pts = 100
        else:
            x_pts = 100
        y_pts = int(
            x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                if self.images:
                    env.reset(pos=(x, y))
                    obs = process_obs(env._get_obs(images=True))
                    states.append(obs)
                else:
                    states.append([x, y])

        num_states = len(states)
        states = self.torchify(np.array(states))
        costs = self.get_value(states)

        grid = costs.detach().cpu().numpy()
        grid = grid.reshape(y_pts, x_pts)
        if self.env_name == 'simplepointbot0':
            plt.gca().add_patch(
                Rectangle(
                    (0, 25),
                    500,
                    50,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'))
        elif self.env_name == 'simplepointbot1':
            plt.gca().add_patch(
                Rectangle(
                    (45, 65),
                    10,
                    20,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'))

        if 'maze' in self.env_name:
            background = cv2.resize(env._get_obs(images=True), (x_pts, y_pts))
            plt.imshow(background)
            plt.imshow(grid.T, alpha=0.6)
        else:
            plt.imshow(grid.T)
        plt.savefig(
            osp.join(self.logdir, "trex_cost_" + str(ep)),
            bbox_inches='tight')