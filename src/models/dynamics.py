import torch
from torch import nn
import numpy as np
from models.architectures.utils import swish, truncated_normal
from tqdm import trange

TORCH_DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class PtModel(nn.Module):
    def __init__(self, state_size, action_size, lr=0.001):
        super().__init__()

        self.state_dim = state_size
        self.action_dim = action_size
        in_features = state_size + action_size
        out_features = state_size
        self.in_features = in_features
        self.out_features = out_features
        hidden_size = 400

        self.fc0 = nn.Linear(in_features, hidden_size)
        # nn.init.trunc_normal_(self.fc0.weight, std=1.0 / (2.0 * np.sqrt(in_features)))
        # nn.init.constant_(self.fc0.bias, 0)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # nn.init.trunc_normal_(self.fc1.weight, std=1.0 / (2.0 * np.sqrt(200)))
        # nn.init.constant_(self.fc1.bias, 0)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # nn.init.trunc_normal_(self.fc2.weight, std=1.0 / (2.0 * np.sqrt(200)))
        # nn.init.constant_(self.fc2.bias, 0)

        self.fc3 = nn.Linear(hidden_size, out_features)
        # nn.init.trunc_normal_(self.fc3.weight, std=1.0 / (2.0 * np.sqrt(200)))
        # nn.init.constant_(self.fc3.bias, 0)

        with torch.no_grad():
            for layer in [self.fc0, self.fc1, self.fc2, self.fc3]:
                layer.weight = nn.Parameter(truncated_normal(layer.weight.shape, 1 / (2.0 * np.sqrt(layer.weight.shape[0]))))
                layer.bias = nn.Parameter(torch.zeros(1, layer.out_features, dtype=torch.float32))
    
        self.inputs_mu = nn.Parameter(
            torch.zeros(1, in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(
            torch.zeros(1, in_features), requires_grad=False)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr) # eps=1e-4

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

        return x * self.inputs_sigma[:, self.state_dim] + self.inputs_mu[:, :self.state_dim] + inputs[:, :self.state_dim]

    def train_dynamics(self, observations, actions, epochs, batch_size=32, val_split=0.9, patience=100, update_stats=True):
        # data = [[S, A, S'] * len(data) ]
        split = int(val_split * len(actions))

        train_in = np.concatenate((observations[:split], actions[:split]), axis=1)
        self.train_in = torch.from_numpy(train_in).to(TORCH_DEVICE).float()
        self.train_out = torch.from_numpy(observations[1:split+1]).to(TORCH_DEVICE).float()
        if update_stats:
            self.fit_input_stats(train_in)

        val_in = np.concatenate((observations[split:-1], actions[split:]), axis=1)
        val_in = torch.from_numpy(val_in).to(TORCH_DEVICE).float()
        val_out = torch.from_numpy(observations[split+1:]).to(TORCH_DEVICE).float()
        
        num_batches = split // batch_size + 1
        loss_fn = torch.nn.MSELoss()
        epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
        k = 0
        val_losses = []
        best_params = self.parameters()
        for _ in epoch_range:
            idx = np.random.permutation(split)
            for i in range(num_batches):
                batch_idx = idx[i * batch_size: (i + 1) * batch_size]

                batch_in = self.train_in[batch_idx]
                batch_out = self.train_out[batch_idx]
                model_out = self.forward(batch_in)
                self.optim.zero_grad()
                loss = loss_fn(model_out, batch_out)
                loss.backward()
                self.optim.step()
            if val_split < 1:
                with torch.no_grad():
                    val_losses.append(loss_fn(self.forward(val_in), val_out).item())
                    if len(val_losses) > 1:
                        if val_losses[-1] > val_losses[-2]:
                            k += 1
                        else:
                            k = 0
                            if val_losses[-1] == min(val_losses):
                                best_params = self.parameters()
                    if k > patience:
                        break
        with torch.no_grad():
            for p, best_p in zip(self.parameters(), best_params):
                p.copy_(best_p)
        return val_losses


def predict_gt_dynamics(env, state_dim, input):
    input_ = input.numpy()
    next_obs = []
    for x in input_:
        ob, ac = x[:state_dim], x[state_dim:]
        env.reset(pos=ob)
        next_ob, _, _, _ = env.step(ac)
        next_obs.append(next_ob)
    return torch.from_numpy(np.array(next_obs)).float()

