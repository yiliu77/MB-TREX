import numpy as np
import torch

class MPC:
    def __init__(self, dyn_prediction, cost_fn, env, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dyn_prediction = dyn_prediction
        self.trex_cost = cost_fn
        self.horizon = env.horizon
        self.env = env
        self.cem_iters, self.cem_pop_size, self.cem_elites = params["cem_iters"], params["cem_pop_size"], params["cem_elites"]
        self.action_dim = self.env.action_space.shape[0]
        self.ac_bounds = [self.env.action_space.low[0], env.action_space.high[0]]
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.ensemble = params["ensemble"]
        self.mean = 0
        self.std = 1

    def act(self, obs, t=0):
        with torch.no_grad():
            obs = torch.tensor(obs).to(self.device).float()
            self.obs = obs
            num_steps = self.horizon - t
            iter_actions = []
            iter_trajs = []
            std = 1
            mean = 0
            for t in range(self.cem_iters):
                action_samples = torch.empty((self.cem_pop_size, num_steps * self.action_dim)).normal_(mean=0, std=1)
                action_samples = action_samples * std + mean
                action_samples = torch.clamp(action_samples, min=self.ac_bounds[0], max=self.ac_bounds[1])

                actions = action_samples.view(self.cem_pop_size, num_steps, self.action_dim)
                costs, trajs = self.compile_cost(actions, t)
                sortid = costs.argsort()
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[:self.cem_elites]

                if t == self.cem_iters - 1:
                    keep = np.random.choice(sortid[:self.cem_elites])
                    # all_actions.append(actions[np.random.choice(sortid[:self.cem_elites])])
                else:
                    keep = np.random.choice(sortid[self.cem_elites:])
                    # all_actions.append(actions[np.random.choice(sortid[self.cem_elites:])])
                iter_actions.append(actions[keep])
                iter_trajs.append(trajs[keep])
                mean = torch.mean(elites, dim=0)
                var = torch.var(elites, dim=0)
                std = torch.sqrt(var)
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
        costs = self.trex_cost.get_value(states.reshape(acs.shape[0] * acs.shape[1], -1), acs.view(acs.shape[0] * acs.shape[1], -1))
        return costs.reshape(acs.shape[0], acs.shape[1], -1).sum(dim=1).squeeze(), states

    def predict_trajectories(self, start_obs, acs):
        """
        Predict trajectories given multiple action sequences and a starting state
        :param start_obs: starting state np.ndarray with shape (state dimension,)
        :param acs: action sequences np.ndarray with shape (# trajectories, trajectory length, action dimension)
        :return: Tensor of states of shape (# of trajectories, trajectory length + 1, state dimension)
        """
        # acs.shape = [CEM pop size, traj_length, action_dim]
        states = torch.tile(start_obs, (1, acs.shape[0], 1))
        actions = acs.view(acs.shape[1], acs.shape[0], acs.shape[2])
        for ac in actions:
            obs = states[-1]
            next_obs = self.dyn_prediction(torch.cat((obs, ac), dim=1))
            states = torch.cat((states, torch.unsqueeze(next_obs, dim=0)), dim=0)
        return states.view(states.shape[1], states.shape[0], states.shape[2])




