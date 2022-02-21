
import torch

class CEMOptimizer():
    def __init__(self, sol_dim, bounds, cost_fn, iters, pop_size, num_elites, init_mean=0, init_var=1):
        self.sol_dim, self.cost = sol_dim, cost_fn
        self.bounds = bounds
        self.iters = iters
        self.pop_size = pop_size
        self.num_elites = num_elites
        self.init_mean = init_mean
        self.init_var = init_var

    def solve(self):
        mean = self.init_mean
        std = torch.sqrt(self.init_var)
        all_actions = []
        for t in range(self.iters):
            action_samples = torch.empty((self.pop_size, self.sol_dim)).normal_(mean=0, std=1).to(self.device)
            action_samples = action_samples * std + mean
            action_samples = torch.truncate(action_samples, min=self.bounds[0], max=self.bounds[1])

            costs = self.cost(action_samples)
            sortid = costs.argsort()
            actions_sorted = action_samples[sortid]
            elites = actions_sorted[:self.num_elites]
            all_actions.append(actions_sorted)

            mean = torch.mean(elites, dim=0)
            var = torch.var(elites, dim=0)
            std = torch.sqrt(var)

        return mean, all_actions