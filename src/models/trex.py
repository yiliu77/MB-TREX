import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from scipy import special
from torch import optim
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


class ConvBlock(nn.Module):
    def __init__(self, nin, nout):
        super(ConvBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, inputs):
        return self.network(inputs)


class UpConvBlock(nn.Module):
    def __init__(self, nin, nout):
        super(UpConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, inputs):
        return self.main(inputs)


class Encoder(nn.Module):
    def __init__(self, input_channels, action_dim, output_dim):
        super(Encoder, self).__init__()
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = ConvBlock(input_channels, nf)
        # state size. (nf) x 32 x 32
        self.c2 = ConvBlock(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = ConvBlock(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = ConvBlock(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
            nn.Conv2d(nf * 8, output_dim, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.BatchNorm2d(output_dim),
            nn.Tanh()
        )
        self.hidden = nn.Sequential(nn.Linear(output_dim + action_dim, output_dim),
                                    nn.Tanh())

    def forward(self, states, actions):
        h1 = self.c1(states)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = h5.view(h5.shape[0], -1)
        if actions is None:
            return h6
        h6 = torch.cat([h6, actions], dim=1)
        h6 = self.hidden(h6)
        return h6, [h1, h2, h3, h4]


class CostNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.encoder = Encoder(state_dim[2], action_dim, params["enc_dim"])
        # self.fc1 = nn.Linear(params["enc_dim"], params["hidden_dim"])
        # self.fc2 = nn.Linear(params["hidden_dim"], params["hidden_dim"])
        # self.fc3 = nn.Linear(params["hidden_dim"], params["hidden_dim"])
        # self.fc4 = nn.Linear(params["hidden_dim"], 1)
        self.lstm = nn.LSTM(input_size=params["enc_dim"], hidden_size=params["hidden_dim"], num_layers=1)
        self.fc = nn.Linear(params["hidden_dim"], 1)

        self.hidden_state = torch.randn(1, 1, params["hidden_dim"]).cuda()
        self.cell_state = torch.randn(1, 1, params["hidden_dim"]).cuda()

    def get_cost(self, states, actions):
        t, b, h, w, c = states.shape
        t, b, a = actions.shape
        states = states.reshape(b * t, h, w, c)
        actions = actions.reshape(b * t, a)
        x = self.encoder(states, actions)[0]
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # r = self.fc4(x)
        # r = r.reshape(t, b)
        # sum_rewards = torch.sum(r, dim=0)
        # sum_abs_rewards = torch.sum(torch.square(r), dim=0)
        # return sum_rewards, sum_abs_rewards
        x = torch.relu(x.reshape(t, b, -1))
        x = self.lstm(x, (self.hidden_state.repeat(1, b, 1), self.cell_state.repeat(1, b, 1)))[0]
        x = torch.relu(x.reshape(t * b, -1))
        r = self.fc(x)
        r = r.reshape(t, b)
        sum_rewards = torch.sum(r, dim=0)
        sum_abs_rewards = torch.sum(torch.square(r), dim=0)
        return sum_rewards, sum_abs_rewards


class TREX:
    def __init__(self, human, state_dim, action_dim, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.human = human
        self.params = params

        self.cost_models = [CostNetwork(state_dim, action_dim, params).to(device=self.device) for _ in range(params["ensemble_size"])]
        self.cost_opts = [Adam(self.cost_models[i].parameters(), lr=params["lr"]) for i in range(len(self.cost_models))]

        self.pref_states1 = None
        self.pref_states2 = None
        self.pref_actions1 = None
        self.pref_actions2 = None
        self.pref_labels = None

        self.opts = [optim.Adam(self.cost_models[i].parameters(), lr=self.params["lr"]) for i in range(len(self.cost_models))]

    def get_value(self, states, actions):
        with torch.no_grad():
            costs = torch.mean(torch.cat([self.cost_models[i].get_cost(states, actions)[0].unsqueeze(1)
                                          for i in range(len(self.cost_models))], dim=1), dim=1)
        return costs

    def calc_distribution(self, query_states, query_actions):
        total_all_distributions = []
        total_pair_indices = []

        with torch.no_grad():
            for _ in range(self.params["num_generated_pairs"] // 64):
                pair_indices = [random.sample(range(query_states.shape[1]), k=2) for _ in range(64)]
                query_states = torch.as_tensor(query_states, dtype=torch.float32).to(self.device)
                query_actions = torch.as_tensor(query_actions, dtype=torch.float32).to(self.device)
                cost_trajs = torch.cat([self.cost_models[i].get_cost(query_states, query_actions)[0].unsqueeze(1) for i in range(len(self.cost_models))], dim=1).cpu().numpy()

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
                print(all_distributions.shape, np.array(pair_indices).shape)
                total_all_distributions.append(all_distributions)
                total_pair_indices.append(np.array(pair_indices))

        print(len(total_all_distributions))
        print(np.concatenate(total_all_distributions, axis=0).shape)
        return np.concatenate(total_all_distributions, axis=0), np.concatenate(total_pair_indices, axis=0)

    def gen_query(self, query_states, query_actions):
        all_distributions, pair_indices = self.calc_distribution(query_states, query_actions)
        mean = np.mean(all_distributions, axis=1)

        if self.params["query_technique"] == "random":
            score = np.random.random((len(all_distributions),))
        elif self.params["query_technique"] == "infogain":
            mean_entropy = -(mean[:, 0] * np.log2(mean[:, 0]) + mean[:, 1] * np.log2(mean[:, 1]))

            ind_entropy = np.zeros_like(mean_entropy)
            for i in range(all_distributions.shape[1]):
                ind_entropy += -(all_distributions[:, i, 0] * np.log2(all_distributions[:, i, 0] + 1e-5) +
                                 all_distributions[:, i, 1] * np.log2(all_distributions[:, i, 1] + 1e-5))
            score = mean_entropy - ind_entropy / all_distributions.shape[1]
            wandb.log({"Misc/Pref/Query Score": np.max(score), "Misc/Pref/Query Mean Entropy Score": mean_entropy[np.argmax(score)],
                       "Misc/Pref/Query Ind Entropy Score": (ind_entropy / all_distributions.shape[1])[np.argmax(score)]})
        else:
            raise NotImplementedError

        num_queries = self.params["query_batch_size"]
        indices = np.argpartition(score, -num_queries)[-num_queries:]
        return zip(pair_indices[indices], score[indices])

    def train(self, query_states_np, query_actions_np, num_epochs, test_data=None):
        # test_pref_states1, test_pref_states2, test_pref_actions1, test_pref_actions2, test_pref_labels = test_data

        for (traj1_index, traj2_index), kl_max in self.gen_query(query_states_np, query_actions_np):
            query_states1 = query_states_np[:, traj1_index, ...]
            query_states2 = query_states_np[:, traj2_index, ...]
            query_actions1 = query_actions_np[:, traj1_index, ...]
            query_actions2 = query_actions_np[:, traj2_index, ...]
            label = self.human.query_preference(query_states1, query_states2)

            if self.pref_states1 is None:
                self.pref_states1 = query_states1[np.newaxis, :]
                self.pref_states2 = query_states2[np.newaxis, :]
                self.pref_actions1 = query_actions1[np.newaxis, :]
                self.pref_actions2 = query_actions2[np.newaxis, :]
                self.pref_labels = np.array([label])
            self.pref_states1 = np.concatenate([self.pref_states1, query_states1[np.newaxis, :]], axis=0)
            self.pref_states2 = np.concatenate([self.pref_states2, query_states2[np.newaxis, :]], axis=0)
            self.pref_actions1 = np.concatenate([self.pref_actions1, query_actions1[np.newaxis, :]], axis=0)
            self.pref_actions2 = np.concatenate([self.pref_actions2, query_actions2[np.newaxis, :]], axis=0)
            self.pref_labels = np.concatenate([self.pref_labels, np.array([label])], axis=0)
            print("Pairs", self.pref_states1.shape, self.pref_labels.shape, self.pref_actions1.shape)

        dataset = TensorDataset(torch.from_numpy(self.pref_states1), torch.from_numpy(self.pref_states2), torch.from_numpy(self.pref_actions1), torch.from_numpy(self.pref_actions2),
                                torch.from_numpy(self.pref_labels))
        dataloader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True, persistent_workers=True, num_workers=10)

        log_interval = 10
        for epoch in range(num_epochs):
            for data_id, (pref_states1_batch, pref_states2_batch, pref_actions1_batch, pref_actions2_batch, pref_labels_batch) in enumerate(dataloader):
                # print(pref_states1_batch.shape)
                pref_states1_batch = pref_states1_batch.to(self.device).float().permute((1, 0, 2, 3, 4))
                # print(pref_states1_batch.shape)
                pref_states2_batch = pref_states2_batch.to(self.device).float().permute((1, 0, 2, 3, 4))
                pref_actions1_batch = pref_actions1_batch.to(self.device).float().permute((1, 0, 2))
                # print(pref_actions2_batch)
                pref_actions2_batch = pref_actions2_batch.to(self.device).float().permute((1, 0, 2))
                # print(pref_actions2_batch)
                pref_labels_batch = pref_labels_batch.to(self.device).float()

                log_results = {}
                avg_results = {"Avg_Pref_Train_Accuracy": 0}
                for i, cost_model in enumerate(self.cost_models):
                    (costs1, sq_costs1), (costs2, sq_costs2) = cost_model.get_cost(pref_states1_batch, pref_actions1_batch), cost_model.get_cost(pref_states2_batch, pref_actions2_batch)
                    cost_probs = torch.softmax(torch.stack([costs1, costs2], dim=1), dim=1)

                    loss = pref_labels_batch * cost_probs[:, 0] + (1 - pref_labels_batch) * cost_probs[:, 1]
                    loss = -torch.mean(loss) + self.params["regularization"] * torch.mean(sq_costs1 + sq_costs2)

                    self.opts[i].zero_grad()
                    loss.backward()
                    self.opts[i].step()

                    if data_id == 0 and epoch % log_interval == 0:
                        with torch.no_grad():
                            log_results["Pref{}/Pref_Loss".format(i)] = loss.item()

                            if len(self.pref_states1) > 0:
                                train_pref1_cost = costs1.detach().cpu().numpy()
                                train_pref2_cost = costs2.detach().cpu().numpy()
                                train_pref_labels = pref_labels_batch.detach().cpu().numpy()
                                train_non_tie_indices = np.argwhere(train_pref_labels != 0.5)[:, 0]

                                train_accuracy = np.mean(np.equal(np.argmin(np.stack([train_pref1_cost, train_pref2_cost], axis=1), axis=1)[train_non_tie_indices],
                                                                  train_pref_labels[train_non_tie_indices]).astype(np.float32))
                                log_results["Pref{}/Pref_Train_Accuracy".format(i)] = train_accuracy
                                avg_results["Avg_Pref_Train_Accuracy"] += train_accuracy

                            # indices = np.random.choice(range(test_pref_states1.shape[1]), size=min(self.params["batch_size"], test_pref_states1.shape[1]))
                            # test_pref_states1_batch = test_pref_states1[:, indices, ...]
                            # test_pref_states2_batch = test_pref_states2[:, indices, ...]
                            # test_pref_labels_batch = test_pref_labels[indices].detach().cpu().numpy()
                            # test_pref1_cost = cost_model.get_cost(test_pref_states1_batch)[0].detach().cpu().numpy()
                            # test_pref2_cost = cost_model.get_cost(test_pref_states2_batch)[0].detach().cpu().numpy()
                            # test_non_tie_indices = np.argwhere(test_pref_labels_batch != 0.5)[:, 0]
                            #
                            # test_accuracy = np.mean(np.equal(np.argmin(np.stack([test_pref1_cost, test_pref2_cost], axis=1), axis=1)[test_non_tie_indices],
                            #                                  test_pref_labels_batch[test_non_tie_indices]).astype(np.float32))
                            # log_results["Pref{}/Pref_Test_Accuracy".format(i)] = test_accuracy
                            # avg_results["Avg_Pref_Test_Accuracy"] += test_accuracy

                if data_id == 0 and epoch % log_interval == 0:
                    for k, v in avg_results.items():
                        log_results[k] = v / len(self.cost_models)
                    wandb.log(log_results)

    def save(self, f):
        torch.save({
            "cost_model_{}".format(i): self.cost_models[i] for i in range(len(self.cost_models))
        }, f)

    def load(self, f):
        checkpoint = torch.load(f)
        self.cost_models = [checkpoint["cost_model_{}".format(i)] for i in range(len(self.cost_models))]
