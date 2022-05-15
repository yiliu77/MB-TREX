import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.optim import Adam
from models.architectures.utils import get_affine_params
from itertools import combinations

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')



class CostNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 1)
        )

    # N x statedim
    def cum_return(self, traj):
        r = self.model(traj)
        sum_rewards = torch.sum(r)
        sum_abs_rewards = torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def get_abs_costs(self, states):
        r = self.model(states)
        return torch.abs(r)

    def get_cost(self, states):
        r = self.model(states)
        return r

    def forward(self, traj_i, traj_j):
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j


class CostEnsemble(CostNetwork):
    def __init__(self, num_nets, state_dim, hidden_size):
        super().__init__()

        self.lin0_w, self.lin0_b = get_affine_params(num_nets,
                                                     state_dim, hidden_size)

        self.lin1_w, self.lin1_b = get_affine_params(num_nets, hidden_size, hidden_size)

        self.lin2_w, self.lin2_b = get_affine_params(num_nets, hidden_size, hidden_size)

        self.lin3_w, self.lin3_b = get_affine_params(num_nets, hidden_size, 1)

        self.inputs_mu = nn.Parameter(
            torch.zeros(state_dim), requires_grad=False)
        self.inputs_sigma = nn.Parameter(
            torch.zeros(state_dim), requires_grad=False)

    def fit_input_stats(self, data):
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(
            TORCH_DEVICE).float()

    def get_cost(self, states):
        x = (states - self.inputs_mu) / self.inputs_sigma

        x = F.relu(x.matmul(self.lin0_w) + self.lin0_b)
        x = F.relu(x.matmul(self.lin1_w) + self.lin1_b)
        x = F.relu(x.matmul(self.lin2_w) + self.lin2_b)
        x = x.matmul(self.lin3_w) + self.lin3_b
        return x

    def get_abs_costs(self, states):
        return torch.abs(self.get_cost(states))

    def cum_return(self, traj):
        r = self.get_cost(traj)
        sum_rewards = torch.sum(r, axis=1)
        sum_abs_rewards = torch.sum(torch.abs(r), axis=1)
        return sum_rewards, sum_abs_rewards


class TRexCost:
    def __init__(self, encoder, g_dim, params, num_nets=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.params = params
        self.num_nets = num_nets
        if num_nets == 1:
            self.cost_model = CostNetwork(g_dim, params["hidden_size"]).to(self.device)
            self.cost_opt = Adam(self.cost_model.parameters(), lr=params["lr"])
        else:
            self.cost_models = [CostNetwork(g_dim, params["hidden_size"]).to(self.device) for _ in range(num_nets)]
            self.cost_opts = [Adam(cost.parameters()) for cost in self.cost_models]

    def train(self, all_states1, all_actions1, all_states2, all_actions2, all_labels, num_epochs):
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for i in range(len(all_states1)):
                states1, actions1, states2, actions2, label = all_states1[i], all_actions1[i], all_states2[i], all_actions2[i], all_labels[i]
                states1 = torch.as_tensor(np.array(states1)).float().to(self.device)
                actions1 = torch.as_tensor(np.array(actions1)).float().to(self.device)
                states2 = torch.as_tensor(np.array(states2)).float().to(self.device)
                actions2 = torch.as_tensor(np.array(actions2)).float().to(self.device)
                label = torch.as_tensor(np.array(label)).long().to(self.device)

                if self.params["use_images"]:
                    states1 = torch.permute(states1, (0, 3, 1, 2))
                    states2 = torch.permute(states2, (0, 3, 1, 2))

                encodings1 = self.encoder(states1, actions1)
                encodings2 = self.encoder(states2, actions2)

                member = i % self.num_nets

                if self.num_nets == 1:
                    outputs, abs_costs = self.cost_model(encodings1, encodings2)
                else:
                    outputs, abs_costs = self.cost_models[member](encodings1, encodings2)
                outputs = outputs.unsqueeze(0)

                train_loss = loss_fn(outputs, 1 - label.unsqueeze(0)) + self.params["cost_reg"] * abs_costs
                
                if self.num_nets == 1:
                    self.cost_opt.zero_grad()
                else:
                    self.cost_opts[member].zero_grad()
                train_loss.backward()
                if self.num_nets == 1:
                    self.cost_opt.step()
                else:
                    self.cost_opts[member].step()

    def get_value(self, states, actions):
        with torch.no_grad():
            enc_states = self.encoder(states, actions).detach()
            if self.num_nets == 1:
                costs = self.cost_model.get_cost(enc_states)
            else:
                idxs = np.random.permutation(np.arange(self.num_nets))
                costs_split = [self.cost_models[i].get_cost(enc_states[i::self.num_nets]).squeeze() for i in idxs]
                costs = torch.empty(states.shape[0]).to(self.device)
                for i, idx in enumerate(idxs):
                    costs[idx::self.num_nets] = costs_split[i]
        return costs

    def get_info_gain(self, trajs, M=10, c=1.0):
        traj_pairs = list(combinations(trajs, 2))
        traj_pair_MIs = []
        for traj_1, traj_2 in traj_pairs:
            rewards_1 = np.array([model.cum_return(traj_1)[0] for model in self.cost_models])
            reward_1_samples = np.random.normal(np.mean(rewards_1), np.std(rewards_1), M) 
            rewards_2 = np.array([model.cum_return(traj_2)[0] for model in self.cost_models])
            reward_2_samples = np.random.normal(np.mean(rewards_2), np.std(rewards_2), M)

            p_1 = 1/(1+ np.exp(-c * (reward_1_samples - reward_2_samples)))
            p_2 = 1 - p_1
            h = -1 * (p_1 * np.log(p_1) + p_2 * np.log(p_2))

            p_1_bar = np.mean(p_1)
            p_2_bar = np.mean(p_2)
            H_1 = -1 * (p_1_bar * np.log(p_1_bar) + p_2_bar * np.log(p_2_bar))
            H_2 = np.mean(h)

            MI = H_1 - H_2
            traj_pairs_MIs.append(MI)



