import torch
from torch import nn
import numpy as np
from models.architectures.utils import swish

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class PtModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        in_features = state_size + action_size
        out_features = action_size
        self.in_features = in_features
        self.out_features = out_features

        self.fc0 = nn.Linear(in_features, 200)
        nn.init.trunc_normal_(self.fc0.weight, std=1.0 / (2.0 * np.sqrt(in_features)))
        nn.init.constant_(self.fc0.bias, 0)

        self.fc1 = nn.Linear(200, 200)
        nn.init.trunc_normal_(self.fc1.weight, std=1.0 / (2.0 * np.sqrt(200)))
        nn.init.constant_(self.fc1.bias, 0)

        self.fc2 = nn.Linear(200, 200)
        nn.init.trunc_normal_(self.fc2.weight, std=1.0 / (2.0 * np.sqrt(200)))
        nn.init.constant_(self.fc2.bias, 0)

        self.fc3 = nn.Linear(200, 2)
        nn.init.trunc_normal_(self.fc3.weight, std=1.0 / (2.0 * np.sqrt(200)))
        nn.init.constant_(self.fc3.bias, 0)

        self.inputs_mu = nn.Parameter(
            torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(
            torch.zeros(in_features), requires_grad=False)

        self.optim = torch.optim.Adam(self.parameters(), lr=3e-4, eps=1e-4)

    def compute_decays(self):

        lin0_decays = 0.00025 * (self.fc0.weight**2).sum() / 2.0
        lin1_decays = 0.0005 * (self.fc1.weight**2).sum() / 2.0
        lin2_decays = 0.0005 * (self.fc2.weight**2).sum() / 2.0
        lin3_decays = 0.00075 * (self.fc3.weight**2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(
            TORCH_DEVICE).float()

    def forward(self, inputs):
        # Transform inputs
        x = (inputs - self.inputs_mu) / self.inputs_sigma

        x = swish(self.fc0(x))
        x = swish(self.fc1(x))
        x = swish(self.fc2(x))
        x = self.fc3(x)

        return x

    def train_dynamics(self, data, epochs, train_test_split):
        # data = [[S, A, S'] * len(data) ]
        batch_size = 32
        split = int(train_test_split * len(data))
        train_data = data[:split]
        test_data = data[split:]
        train_in = np.concatenate((train_data[:, 0], train_data[:, 1]), axis=1)
        self.train_in = torch.from_numpy(train_in).to(TORCH_DEVICE).float()
        self.train_out = torch.from_numpy(train_data[:, 2]).to(TORCH_DEVICE).float()
        self.fit_input_stats(train_in)
        num_batches = len(train_data) // batch_size
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            idx = np.random.permutation(len(train_data))
            for i in range(num_batches):
                batch_idx = idx[num_batches * batch_size: (num_batches + 1) * batch_size]

                batch_in = self.train_in[batch_idx]
                batch_out = self.train_out[batch_idx]
                model_out = self.forward(batch_in)
                loss = loss_fn(model_out, batch_out)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        if train_test_split < 1.0:
            test_in = torch.from_numpy(np.concatenate((test_data[:, 0], test_data[:, 1]), axis=1)).to(TORCH_DEVICE).float()
            test_out = torch.from_numpy(test_data[:, 2]).to(TORCH_DEVICE).float()
            return loss_fn(self.forward(test_in), test_out)




