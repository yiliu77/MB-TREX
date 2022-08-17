import sys

import numpy as np
import torch
from torch import nn
import yaml

from itertools import combinations, product
from parser import create_env
from human.human import LowDimHuman
from models.trex import TRexCost
import os
import datetime
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    torch.manual_seed(123456)
    np.random.seed(123456)
    logdir = os.path.join("saved/models/BC/maze", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(logdir, exist_ok=True)
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

    bc_trajs = []
    eps = np.linspace(0, 0.75, num=4)
    start = env.reset()
    costs = [[] for _ in range(len(eps))]
    for i, e in enumerate(eps):
        bc_trajs.append([])
        for _ in range(100):
            traj = bc.rollout_trajectory(env, eps=e, start_pos=start)
            bc_trajs[-1].append(traj)
            costs[i].append(env.get_expert_cost([traj[2][-1]])[0])
    costs = np.array(costs)
    avg_costs = np.mean(costs, axis=1)
    std_costs = np.std(costs, axis=1)
    plt.figure()
    plt.errorbar(eps, avg_costs, yerr=std_costs)
    plt.xlabel("epsilon greedy")
    plt.ylabel("cost at final state")
    plt.savefig(os.path.join(logdir, "degredation.png"))
    plt.close()

    states1 = []
    acs1 = []
    states2 = []
    acs2 = []
    labels = []
    cost_model = TRexCost(lambda x, y: x, 2, params["cost_model"], num_nets=1)
    for traj_set_1, traj_set_2 in combinations([demo_trajs] + bc_trajs, 2):
        for traj1, traj2 in product(traj_set_1, traj_set_2):
            states1.append(np.array(traj1)[0, :])
            acs1.append(np.array(traj1)[1, :])
            states2.append(np.array(traj2)[0, :])
            acs2.append(np.array(traj2)[1, :])
            labels.append(0)

    cost_model.train(states1, acs1, states2, acs2, labels, 25)
    env.visualize_rewards(os.path.join(logdir, "cost.png"), cost_model)