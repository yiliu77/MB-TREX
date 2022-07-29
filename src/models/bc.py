import sys

import numpy as np
import torch
from torch import nn


class BCModel(nn.Module):

    def __init__(self, state_size=2, action_size=2, hidden_size=32, lr=0.0001, wd=0.001):
        super(BCModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, action_size))
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def train_step(self, states, actions):
        self.optim.zero_grad()
        loss = self.criterion(self.forward(states), actions)
        loss.backward()
        self.optim.step()
        return loss.item()

    def val(self, states, actions):
        with torch.no_grad():
            loss = self.criterion(self.forward(states), actions)
        return loss.item()


class BCEnsemble:
    def __init__(self, device, num_nets=5, **params):
        self.num_nets = num_nets
        self.models = [BCModel(**params).to(device) for _ in range(num_nets)]
        self.device = device

    def train(self, states, actions, epochs, val_split=0.9):
        split = int(0.9 * len(states))
        train_in = states[:split]
        train_out = actions[:split]
        val_in = states[split:]
        val_out = actions[split:]

        datasets = []
        for i in range(self.num_nets):
            idx = np.random.permutation(np.random.choice(len(train_in), len(train_in)))
            datasets.append((train_in[idx], train_out[idx]))

        val_losses = [[] for i in range(self.num_nets)]
        loss_mean_pred = []
        patience = 10000
        k = 0
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for i, model in enumerate(self.models):
                model.train_step(*datasets[i])
                val_losses[i].append(model.val(val_in, val_out))

            loss_mean_pred.append(criterion(self.predict(val_in), val_out).item())
            if len(loss_mean_pred) > 1:
                if loss_mean_pred[-1] > loss_mean_pred[-2]:
                    k += 1
                else:
                    k = 0
                    if loss_mean_pred[-1] == min(loss_mean_pred):
                        best_params = [model.parameters() for model in self.models]
            if k > patience:
                break

        for i, model in enumerate(self.models):
            with torch.no_grad():
                for p, best_p in zip(model.parameters(), best_params[i]):
                    p.copy_(best_p)
        return val_losses, loss_mean_pred

    def predict(self, states):
        return self.predict_indiv(states).mean(axis=0)

    def predict_indiv(self, states):
        return torch.stack([model(states) for model in self.models])

    def rollout_trajectory(self, env, eps=0.0, start_pos=(), return_success=False):
        done = False
        state = env.reset(pos=start_pos)
        states = []
        acs = []
        while not done:
            states.append(state)
            state_torch = torch.from_numpy(state).float().to(self.device)
            if np.random.random() < eps:
                ac = env.action_space.sample()
            else:
                ac = self.predict(state_torch).cpu().detach().numpy()
            acs.append(ac)
            next_state, reward, done, info = env.step(ac)
            state = next_state
        if return_success:
            return [np.array(states[:-1]), np.array(acs[:-1]), np.array(states[1:])], info["success"]
        else:
            return [np.array(states[:-1]), np.array(acs[:-1]), np.array(states[1:])]