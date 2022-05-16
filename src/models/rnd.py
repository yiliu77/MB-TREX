import numpy as np
from torch import nn


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RND(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2)),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.SELU(),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Linear(256, 256)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.model(x)


class RNDTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(2, 2)),
            nn.SELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.SELU(),
            nn.Linear(512, 256)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.model(x)
