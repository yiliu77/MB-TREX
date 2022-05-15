import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

class MPC:
    def __init__(self, dyn_prediction, trex_cost, env, params, rnd=None, dynamics_cuda=True, logdir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dynamics_cuda = dynamics_cuda
        self.dyn_prediction = dyn_prediction
        self.trex_cost = trex_cost
        self.horizon = env.horizon
        self.env = env
        self.cem_iters, self.cem_pop_size, self.cem_elites = params["cem_iters"], params["cem_pop_size"], params["cem_elites"]
        self.action_dim = self.env.action_space.shape[0]
        self.ac_bounds = [self.env.action_space.low[0], env.action_space.high[0]]
        self.rnd_weight = params["rnd_weight"]
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.ensemble = params["ensemble"]
        self.rnd = rnd
        self.mean = 0
        self.std = 1
        self.logdir = logdir

    def act(self, obs, t=0, plot=False):
        with torch.no_grad():
            obs = torch.tensor(obs).float()
            self.obs = obs
            num_steps = self.horizon - t
            iter_actions = []
            iter_trajs = []
            std = 1
            mean = 0
            if plot:
                fig, axs = plt.subplots(nrows=1, ncols=self.cem_iters, figsize=(4 * self.cem_iters, 4))
                fig.suptitle("CEM trajectories")
                background = cv2.resize(self.env._get_obs(images=True), (100, 100))
            for t in range(self.cem_iters):
                action_samples = torch.empty((self.cem_pop_size, num_steps * self.action_dim)).normal_(mean=0, std=1)
                action_samples = action_samples * std + mean
                action_samples = torch.clamp(action_samples, min=self.ac_bounds[0], max=self.ac_bounds[1])

                actions = action_samples.view(self.cem_pop_size, num_steps, self.action_dim)
                costs, trajs = self.compile_cost(actions, t)
                sortid = costs.argsort().cpu().numpy()
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[:self.cem_elites]

                if t == self.cem_iters - 1:
                    keep = np.random.choice(sortid[:self.cem_elites])
                else:
                    keep = np.random.choice(sortid[self.cem_elites:])
                iter_actions.append(actions[keep])
                iter_trajs.append(trajs[keep].cpu().numpy())
                mean = torch.mean(elites, dim=0)
                var = torch.var(elites, dim=0)
                std = torch.sqrt(var)
                if plot:
                    axs[t].imshow(background, extent=[0, 100, 100, 0])
                    axs[t].plot(trajs[sortid[:self.cem_elites]][0, :, 0].cpu().numpy() * 100 / 0.6 + 50, trajs[sortid[:self.cem_elites]][0, :, 1].cpu().numpy() * 100 / 0.6 + 50)
            if plot:
                plt.savefig(os.path.join(self.logdir, plot + ".png"), bbox_inches='tight')
                plt.close()
                    

        ac = mean.reshape(-1, self.action_dim)[0]
        return ac, iter_actions, iter_trajs

    def compile_cost(self, acs, member):
        # acs.shape = [CEM pop size, traj_length, action_dim]
        # states = torch.tile(self.obs, (1, acs.shape[0], 1))
        # costs = []
        # for ac in acs.view(acs.shape[1], acs.shape[0], acs.shape[2]):
        #     obs = states[-1]
        #     if self.ensemble:
        #         cost = self.trex_cost.get_value(obs, ac).detach().cpu().numpy()[member]
        #     else:
        #         cost = self.trex_cost.get_value(obs, ac).detach().cpu().numpy()
        #     costs.append(cost)
        #     next_obs = self.dyn_prediction(torch.cat((obs, ac), dim=1))
        #     states = torch.cat((states, torch.unsqueeze(next_obs, dim=0)), dim=0)
        # costs = self.torchify(costs)
        # return torch.sum(costs, dim=0).squeeze(), states
        states = self.predict_trajectories(self.obs, acs)[:, :-1, :]
        states_lst = states.reshape(self.cem_pop_size * self.horizon, -1)
        costs = self.trex_cost.get_value(states_lst, acs.view(-1, self.action_dim)).reshape(self.cem_pop_size, self.horizon).sum(dim=1)
        costs -= self.rnd_weight * self.rnd_cost(states_lst)
        return costs, states

    def predict_trajectories(self, start_obs, acs):
        """
        Predict trajectories given multiple action sequences and a starting state
        :param start_obs: starting state np.ndarray with shape (state dimension,)
        :param acs: action sequences np.ndarray with shape (# trajectories, trajectory length, action dimension)
        :return: Tensor of states of shape (# of trajectories, trajectory length + 1, state dimension)
        """
        # acs.shape = [CEM pop size, traj_length, action_dim]
        if self.dynamics_cuda:
            actions = acs.permute(1, 0, 2).to(self.device)
            start_obs = start_obs.to(self.device)
        else:
            actions = acs.permute(1, 0, 2)
        states = start_obs.repeat(1, acs.shape[0], 1)
        for ac in actions:
            obs = states[-1]
            next_obs = self.dyn_prediction(torch.cat((obs, ac), dim=1))
            states = torch.cat((states, torch.unsqueeze(next_obs, dim=0)), dim=0)
        return states.permute(1, 0, 2).to(self.device)


    def rnd_cost(self, states):
        raw_rnd = self.rnd.get_value(states)
        normalized_rnd = raw_rnd / torch.sqrt(self.rnd.stddev())
        self.rnd.update_stats(raw_rnd)
        return normalized_rnd.reshape(self.cem_pop_size, self.horizon)[:, -1]
