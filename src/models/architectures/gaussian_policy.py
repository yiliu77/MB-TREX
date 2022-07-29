import torch
from torch import nn, distributions
from torch.nn import functional as F


# n_states -> split: n_actions * 2 (none)
class ContGaussianPolicy(nn.Module):
    def __init__(self, action_dim, action_range):
        super(ContGaussianPolicy, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.mu = nn.Sequential(nn.Linear(256, action_dim))
        self.log_std = nn.Sequential(nn.Linear(256, action_dim))

        action_low, action_high = action_range
        self.action_scale = torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32)
        self.action_bias = torch.as_tensor((action_high + action_low) / 2, dtype=torch.float32)

    def forward(self, states):
        states = states.permute((0, 3, 1, 2)) / 255
        embedding = self.embedding(states)
        mu, log_std = self.mu(embedding), self.log_std(embedding)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, states):
        mus, log_stds = self.forward(states)
        stds = torch.exp(log_stds)

        normal_dists = distributions.Normal(mus, stds)
        outputs = normal_dists.rsample()
        tanh_outputs = torch.tanh(outputs)
        actions = self.action_scale * tanh_outputs + self.action_bias
        mean_actions = self.action_scale * torch.tanh(mus) + self.action_bias

        log_probs = normal_dists.log_prob(outputs)
        # https://arxiv.org/pdf/1801.01290.pdf appendix C
        log_probs -= torch.log(
            self.action_scale * (torch.ones_like(tanh_outputs, requires_grad=False) - tanh_outputs.pow(2)) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)

        return actions, log_probs, mean_actions

    def to(self, *args, **kwargs):
        device = args[0]
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.embedding = self.embedding.to(device)
        self.mu = self.mu.to(device)
        self.log_std = self.log_std.to(device)
        return super(ContGaussianPolicy, self).to(device)
