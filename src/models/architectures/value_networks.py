import torch
from torch import nn


class ContTwinQNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2)),
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
        self.q_net1 = nn.Sequential(nn.Linear(256 + action_dim, 1))
        self.q_net2 = nn.Sequential(nn.Linear(256 + action_dim, 1))

    def forward(self, states, actions):
        states = torch.permute(states, (0, 3, 1, 2)) / 255

        states = self.embedding(states)
        q1_out, q2_out = self.q_net1(torch.cat([states, actions], dim=1)), self.q_net2(torch.cat([states, actions], dim=1))
        return torch.min(q1_out, q2_out), q1_out, q2_out
