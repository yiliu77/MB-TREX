import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.optim import Adam


class CostNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

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

        self.cost_model = CostNetwork(g_dim, params["hidden_size"]).to(device=self.device)
        self.cost_opt = Adam(self.cost_model.parameters(), lr=params["lr"])

    def train(self, all_states1, all_actions1, all_states2, all_actions2, all_labels, num_epochs):
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for i in range(len(all_states1)):
                states1, actions1, states2, actions2 = all_states1[i], all_actions1[i], all_states2[i], all_actions2[i]
                states1 = torch.as_tensor(np.array(states1)).to(self.device)
                actions1 = torch.as_tensor(np.array(actions1)).to(self.device)
                states2 = torch.as_tensor(np.array(states2)).to(self.device)
                actions2 = torch.as_tensor(np.array(actions2)).to(self.device)

                encodings1 = self.encoder(states1, actions1)
                encodings2 = self.encoder(states2, actions2)

                outputs, abs_costs = self.cost_model(encodings1, encodings2)
                outputs = outputs.unsqueeze(0)

                train_loss = loss_fn(outputs, all_labels[i]) + self.params["cost_reg"] * abs_costs

                self.cost_opt.zero_grad()
                train_loss.backward()
                self.cost_opt.step()

    def get_value(self, states, actions):
        with torch.no_grad():
            enc_states = self.encoder(states, actions)
            abs_costs = self.cost_model.get_abs_costs(enc_states)
        return abs_costs

    def plot(self, ep):
        print("Plotting learned cost: ", ep)
        x_bounds = [-0.05, 0.25]
        y_bounds = [-0.05, 0.25]

        states = []
        x_pts = 100
        y_pts = int(x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                env.reset(pos=(x, y))
                obs = process_obs(env._get_obs(images=True))
                states.append(obs)

        num_states = len(states)
        states = self.torchify(np.array(states))
        costs = self.get_value(states)

        grid = costs.detach().cpu().numpy()
        grid = grid.reshape(y_pts, x_pts)

        background = cv2.resize(env._get_obs(images=True), (x_pts, y_pts))
        plt.imshow(background)
        plt.imshow(grid.T, alpha=0.6)
        plt.savefig(
            osp.join(self.logdir, "trex_cost_" + str(ep)),
            bbox_inches='tight')
