from torch import nn
import torch
import numpy as np

OUTPUT_DIM = 128

class RND(nn.Module):
    def __init__(self, state_dim, output_dim=OUTPUT_DIM, device=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SELU(),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, output_dim)
        ).to(device)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        
        self.optim = torch.optim.Adam(self.parameters())
        self.target = RNDTarget(state_dim, output_dim, device=device)
        self.stats = RunningMeanStd(device=device)

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def get_value(self, x):
        return torch.norm(self.forward(x) - self.target(x), dim=1)

    @torch.no_grad()
    def update_stats_from_states(self, x):
        self.stats.update(self.get_value(x))

    def update_stats(self, x):
        self.stats.update(x)
    
    def train(self, x, num_epochs=1):
        loss_criterion = nn.MSELoss()
        for _ in range(num_epochs):
            loss = loss_criterion(self.forward(x), self.target.forward(x))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def stddev(self):
        return torch.sqrt(self.stats.var)

class RNDTarget(nn.Module):
    def __init__(self, input_dim, output_dim, device=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SELU(),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, output_dim)
        ).to(device)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
