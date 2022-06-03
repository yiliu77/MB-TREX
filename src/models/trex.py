import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from scipy import special
from itertools import combinations, product
import random

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')



class CostNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
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
        t, b, d = states.shape
        states = states.reshape(t * b, d)
        r = self.model(states)
        r = r.reshape(t, b)
        sum_rewards = torch.sum(r, dim=0)
        sum_sq_rewards = torch.sum(torch.square(r), dim=0)
        return sum_rewards, sum_sq_rewards

    def forward(self, traj_i, traj_j):
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j


class TRexCost:
    def __init__(self, human, state_dim, action_dim, cem_keep_iter, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.human = human
        self.params = params

        self.input_dim = state_dim if params["state_only"] else state_dim + action_dim
        self.preprocess = lambda s, a: s if params["state_only"] else lambda s, a: np.concatenate((s, a), axis=0)

        self.cost_models = [CostNetwork(self.input_dim, params["hidden_dim"]).to(device=self.device) for _ in
                            range(params["ensemble_size"])]
        self.cost_opts = [Adam(self.cost_models[i].parameters(), lr=params["lr"]) for i in range(len(self.cost_models))]

        self.cem_keep_per_iter = cem_keep_iter

        self.pref_states1 = None
        self.pref_states2 = None
        self.pref_labels = None

    def train(self, trajs1, trajs2, labels, num_epochs):
        if trajs1 is not None:
            if self.pref_states1 is None:
                self.pref_states1 = trajs1
                self.pref_states2 = trajs2
                self.pref_labels = labels
            else:
                self.pref_states1 = np.concatenate([self.pref_states1, trajs1], axis=0)
                self.pref_states2 = np.concatenate([self.pref_states2, trajs2], axis=0)
                self.pref_labels = np.concatenate([self.pref_labels, labels], axis=0)

        dataset = TensorDataset(torch.from_numpy(self.pref_states1), torch.from_numpy(self.pref_states1),
                                torch.from_numpy(self.pref_labels))
        dataloader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True, persistent_workers=True,
                                num_workers=4)
        ce_loss = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            for data_id, (pref_states1_batch, pref_states2_batch, pref_labels_batch) in enumerate(dataloader):
                pref_states1_batch = pref_states1_batch.to(self.device).float()
                pref_states2_batch = pref_states2_batch.to(self.device).float()
                pref_labels_batch = pref_labels_batch.to(self.device).float()

                for i, cost_model in enumerate(self.cost_models):
                    (costs1, sq_costs1), (costs2, sq_costs2) = cost_model.get_cost(
                        pref_states1_batch), cost_model.get_cost(pref_states2_batch)
                    cost_probs = torch.softmax(torch.stack([costs1, costs2], dim=1), dim=1)

                    loss = ce_loss(pref_labels_batch, cost_probs) + self.params["regularization"] * torch.mean(
                        sq_costs1 + sq_costs2)

                    self.opts[i].zero_grad()
                    loss.backward()
                    self.opts[i].step()


    def get_value(self, states, actions=None):
        if actions is not None:
            inputs = self.preprocess(states, actions) # for compatibility
        else:
            inputs = states
        with torch.no_grad():
            costs = torch.mean(torch.cat([self.cost_models[i].get_cost(inputs)[0].unsqueeze(1)
                                          for i in range(len(self.cost_models))], dim=1), dim=1)
        return costs


    def calc_distribution(self, query_states):
        with torch.no_grad():
            pair_indices = [random.sample(range(query_states.shape[1]), k=2) for _ in
                            range(self.params["num_generated_pairs"])]
            query_states = torch.as_tensor(query_states, dtype=torch.float32).to(self.device)
            cost_trajs = torch.cat(
                [self.cost_models[i].get_cost(query_states)[0].unsqueeze(1) for i in range(len(self.cost_models))],
                dim=1).cpu().numpy()

            all_distributions = []
            for triplet_index in pair_indices:
                distributions = []
                for i in range(len(self.cost_models)):
                    sampled_cost1 = cost_trajs[triplet_index[0]][i]
                    sampled_cost2 = cost_trajs[triplet_index[1]][i]
                    probs = special.softmax([sampled_cost1, sampled_cost2])
                    distributions.append(probs)
                all_distributions.append(distributions)
            all_distributions = np.array(all_distributions)
            return all_distributions, np.array(pair_indices)


    def gen_queries(self, all_states, all_actions):
        if self.params["state_only"]:
            trajectories = all_states
        else:
            trajectories = np.concatenate([all_states, all_actions], axis=0)

        num_queries = self.params["query_batch_size"]
        if self.params["query_technique"] == "random":
            pair_indices = np.asmatrix(list(combinations(np.arange(len(trajectories)), 2)))
            pair_idx = np.random.choice(pair_indices, size=num_queries, replace=False)
        elif self.params["query_technique"] == "infogain":
            all_distributions, pair_indices = self.calc_distribution(trajectories)
            mean = np.mean(all_distributions, axis=1)
            mean_entropy = -(mean[:, 0] * np.log2(mean[:, 0]) + mean[:, 1] * np.log2(mean[:, 1]))

            ind_entropy = np.zeros_like(mean_entropy)
            for i in range(all_distributions.shape[1]):
                ind_entropy += -(all_distributions[:, i, 0] * np.log2(all_distributions[:, i, 0] + 1e-5) +
                                 all_distributions[:, i, 1] * np.log2(all_distributions[:, i, 1] + 1e-5))
            score = mean_entropy - ind_entropy / all_distributions.shape[1]
            indices = np.argpartition(score, -num_queries)[-num_queries:]
            pair_idx = pair_indices[indices]
        elif self.params["query_technique"] == "cemiters":
            pair_indices = np.asmatrix(list(product(np.split(np.arange(len(trajectories)), len(trajectories) // self.cem_keep_per_iter))))
            pair_idx = np.random.choice(pair_indices, size=num_queries, replace=False)
        else:
            raise NotImplementedError

        pref_states1, pref_states2, pref_labels = None, None, None
        for traj1_index, traj2_index in pair_idx:
            query1 = trajectories[:, traj1_index, ...]
            query2 = trajectories[:, traj2_index, ...]
            label = self.human.query_preference(query1, query2)

            if pref_states1 is None:
                pref_states1 = query1[np.newaxis, :]
                pref_states2 = query2[np.newaxis, :]
                pref_labels = np.array([label])
            else:
                pref_states1 = np.concatenate([pref_states1, query1[np.newaxis, :]], axis=0)
                pref_states2 = np.concatenate([pref_states2, query2[np.newaxis, :]], axis=0)
                pref_labels = np.concatenate([pref_labels, np.array([label])], axis=0)
            print("Pairs", pref_states1.shape, pref_labels.shape)

        return pref_states1, pref_states2, pref_labels


class GTCost:
    def __init__(self, env):
        self.env = env

    def get_value(self, states, actions):
        device = states.device
        states = states.detach().cpu().numpy()
        actions = actions.detach().cpu().numpy()
        return torch.from_numpy(self.env.get_expert_cost(states, actions)).to(device)
    
    def train(self, *args):
        pass
