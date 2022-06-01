import copy

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from torch import optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader


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
        self.hidden = nn.Sequential(nn.Linear(128 + action_dim, 128),
                                    nn.Tanh())

    def forward(self, input, action):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = h5.view(h5.shape[0], -1)

        if action is None:
            return h6

        h6 = torch.cat([h6, action], dim=1)
        h6 = self.hidden(h6)
        return h6, [h1, h2, h3, h4]


class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Decoder, self).__init__()
        self.input_channels = input_channels
        nf = 64
        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_channels, nf * 8, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2)
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = UpConvBlock(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = UpConvBlock(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = UpConvBlock(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, output_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.input_channels, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for _ in range(self.n_layers)])
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(self.batch_size, self.hidden_size).cuda(),
                           torch.zeros(self.batch_size, self.hidden_size).cuda()))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        return self.output(h_in)


class GaussianLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(GaussianLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(self.batch_size, self.hidden_size).cuda(),
                           torch.zeros(self.batch_size, self.hidden_size).cuda()))
        return hidden

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class SVG:
    type = "SVG"

    def __init__(self, state_dim, action_dim, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = params
        self.n_past = params["n_past"]
        self.n_future = params["n_future"]
        self.last_frame_skip = params["last_frame_skip"]
        self.beta = params["beta"]

        self.encoder = Encoder(state_dim[2], action_dim, params["enc_dim"]).to(self.device)
        self.decoder = Decoder(params["enc_dim"], state_dim[2]).to(self.device)

        self.frame_predictor = LSTM(params["enc_dim"] + params["z_dim"], params["enc_dim"], params["rnn_size"],
                                    params["predictor_rnn_layers"], params["batch_size"]).to(self.device)
        self.prior = GaussianLSTM(params["enc_dim"], params["z_dim"], params["rnn_size"], params["prior_rnn_layers"],
                                  params["batch_size"]).to(self.device)
        self.posterior = GaussianLSTM(params["enc_dim"], params["z_dim"], params["rnn_size"],
                                      params["posterior_rnn_layers"], params["batch_size"]).to(self.device)
        self.batch_size = params["batch_size"]

        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=params["lr"])
        self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=params["lr"])
        self.frame_predictor_opt = optim.Adam(self.frame_predictor.parameters(), lr=params["lr"])
        self.prior_opt = optim.Adam(self.prior.parameters(), lr=params["lr"])
        self.posterior_opt = optim.Adam(self.posterior.parameters(), lr=params["lr"])

        self.encoder_scheduler = ExponentialLR(self.encoder_opt, gamma=0.999)
        self.decoder_scheduler = ExponentialLR(self.decoder_opt, gamma=0.999)
        self.frame_predictor_scheduler = ExponentialLR(self.frame_predictor_opt, gamma=0.999)
        self.prior_scheduler = ExponentialLR(self.prior_opt, gamma=0.999)
        self.posterior_scheduler = ExponentialLR(self.posterior_opt, gamma=0.999)

        self.rec_loss = nn.MSELoss()

        self.log_iter = 0

    def kl_div(self, mu1, log_var1, mu2, log_var2):
        KLD = 0.5 * torch.log(log_var2.exp() / log_var1.exp()) + (log_var1.exp() + (mu1 - mu2).pow(2)) / (
                2 * log_var2.exp()) - 1 / 2
        return torch.mean(KLD)

    def create_encoding(self, states, actions):
        with torch.no_grad():
            if len(states.shape) == 4:
                return self.encoder(states, actions)[0].detach()  # TODO pull out action
            elif len(states.shape) == 5:
                reshaped_states = states.reshape(-1, states.shape[2], states.shape[3], states.shape[4])
                reshaped_actions = actions.reshape(-1, actions.shape[2])
                return self.encoder(reshaped_states, reshaped_actions)[0].reshape(states.shape[0], states.shape[1], -1).detach()
            else:
                raise NotImplementedError

    def predict_states(self, start_states, actions, horizon):
        assert len(start_states) == self.n_past
        with torch.no_grad():
            self.frame_predictor.hidden = self.frame_predictor.init_hidden()
            self.prior.hidden = self.prior.init_hidden()
            self.posterior.hidden = self.posterior.init_hidden()

            gen_seq = copy.deepcopy(start_states) # TODO fix actions being too short (n_past actions are actually n_future)

            x_in = None
            for i in range(horizon - 1):
                if i < self.n_past:
                    h, h_skip = self.encoder(start_states[i], actions[i])
                else:
                    h, h_skip = self.encoder(x_in, actions[i])

                if self.last_frame_skip or i < self.n_past:
                    h, skip = h, h_skip
                else:
                    h, _ = h, h_skip
                z_t, _, _ = self.prior(h)
                h = self.frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = self.decoder([h, skip]).detach()
                gen_seq.append(x_in)
            return torch.stack(gen_seq)

    def loss(self, sequences, actions):
        sequences = sequences.permute((1, 0, 4, 2, 3))
        actions = actions.permute((1, 0, 2))

        # initialize the hidden state.
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        mse = 0
        kld = 0
        for i in range(1, self.n_past + self.n_future):
            h, h_skip = self.encoder(sequences[i - 1], actions[i - 1])
            h_target, _ = self.encoder(sequences[i], actions[i])
            if self.last_frame_skip or i < self.n_past:
                h, skip = h, h_skip
            else:
                h, _ = h, h_skip
            _, mu_p, logvar_p = self.prior(h)
            z_t, mu, logvar = self.posterior(h_target)
            h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
            x_pred = self.decoder([h_pred, skip])
            mse += self.rec_loss(x_pred, sequences[i])
            kld += self.kl_div(mu, logvar, mu_p, logvar_p)

        loss = mse + kld * self.beta

        if self.log_iter % 20 == 0:
            wandb.log({"Loss": loss.item(), "MSE": mse.item(), "KLD": kld.item() * self.beta})
        self.log_iter += 1

        return loss

    def train_step(self, sequences, actions):
        self.encoder.train()
        self.decoder.train()
        self.frame_predictor.train()
        self.prior.train()
        self.posterior.train()

        sequences = sequences.to(self.device).float()
        actions = actions.to(self.device).float()

        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()
        self.frame_predictor_opt.zero_grad()
        self.prior_opt.zero_grad()
        self.posterior_opt.zero_grad()

        loss = self.loss(sequences, actions)
        loss.backward()

        self.encoder_opt.step()
        self.decoder_opt.step()
        self.frame_predictor_opt.step()
        self.prior_opt.step()
        self.posterior_opt.step()

        return loss.item()

    def train(self, num_epochs, dataset):
        train_len = int(self.params["train_test_split"] * len(dataset))
        val_len = int(len(dataset) - train_len)
        train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_data, num_workers=self.params["num_cores"],
                                  batch_size=self.params["batch_size"], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, num_workers=self.params["num_cores"], batch_size=self.params["batch_size"],
                                shuffle=True, drop_last=True)

        self.encoder.train()
        self.decoder.train()
        self.frame_predictor.train()
        self.prior.train()
        self.posterior.train()

        losses = []
        for _ in tqdm.tqdm(range(num_epochs)):
            losses = []
            for sequences, actions in train_loader:
                losses.append(self.train_step(sequences, actions))

            for sequences, actions in val_loader:
                sequences = sequences.to("cuda").float()
                actions = actions.to("cuda").float()
                self.validate(sequences, actions)
                break

            self.encoder_scheduler.step()
            self.decoder_scheduler.step()
            self.frame_predictor_scheduler.step()
            self.prior_scheduler.step()
            self.posterior_scheduler.step()

            wandb.log({"LR": self.encoder_opt.state_dict()['param_groups'][0]['lr']})

        return np.mean(losses)

    def validate(self, sequences, actions):
        with torch.no_grad():
            sequences = sequences.permute((1, 0, 4, 2, 3))
            actions = actions.permute((1, 0, 2))

            nsample = 10
            gen_seqs = [[] for _ in range(nsample)]
            gt_seqs = [sequences[i].cpu().numpy() for i in range(len(sequences))]

            for s in range(nsample):
                self.frame_predictor.hidden = self.frame_predictor.init_hidden()
                self.prior.hidden = self.prior.init_hidden()

                gen_seqs[s].append(sequences[0].cpu().numpy())
                x_in = sequences[0]
                for i in range(len(sequences) - 1):
                    h, h_skip = self.encoder(x_in, actions[i])
                    if self.last_frame_skip or i < self.n_past:
                        h, skip = h, h_skip
                    else:
                        h, _ = h, h_skip
                    if i < self.n_past:
                        x_in = sequences[i]
                        gen_seqs[s].append(x_in.cpu().numpy())
                    else:
                        z_t, _, _ = self.prior(h)
                        h = self.frame_predictor(torch.cat([h, z_t], 1)).detach()
                        x_in = self.decoder([h, skip]).detach()
                        gen_seqs[s].append(x_in.cpu().numpy())

            gen_seqs = np.transpose(np.array(gen_seqs), (0, 2, 1, 4, 5, 3))
            gt_seqs = np.transpose(np.array(gt_seqs), (1, 0, 3, 4, 2))

            gt_seqs = gt_seqs[:5]
            gt_seqs = np.transpose(gt_seqs, (0, 2, 1, 3, 4))
            for gen_seq in gen_seqs:
                gen_seq = gen_seq[:5]

                gen_seq = np.transpose(gen_seq, (0, 2, 1, 3, 4))
                comb_seqs = np.empty((gen_seq.shape[0] + gt_seqs.shape[0], gen_seq.shape[1], gen_seq.shape[2],
                                      gen_seq.shape[3], gen_seq.shape[4]),
                                     dtype=gen_seq.dtype)
                comb_seqs[0::2] = gt_seqs
                comb_seqs[1::2] = gen_seq
                comb_seqs = np.reshape(comb_seqs, (
                    comb_seqs.shape[0] * comb_seqs.shape[1], comb_seqs.shape[2] * comb_seqs.shape[3],
                    comb_seqs.shape[4]))

                gen_images = wandb.Image(comb_seqs, caption="Sequences")
                wandb.log({"Sequences": gen_images})

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.frame_predictor.eval()
        self.prior.eval()
        self.posterior.eval()

    def load(self, directory):
        checkpoint = torch.load(directory)
        self.encoder = checkpoint["encoder"]
        self.decoder = checkpoint["decoder"]
        self.frame_predictor = checkpoint["frame_predictor"]
        self.prior = checkpoint["prior"]
        self.posterior = checkpoint["posterior"]
        self.frame_predictor.batch_size = self.batch_size
        self.prior.batch_size = self.batch_size
        self.posterior.batch_size = self.batch_size

    def save(self, f):
        torch.save({
            "encoder": self.encoder,
            "decoder": self.decoder,
            "frame_predictor": self.frame_predictor,
            "prior": self.prior,
            "posterior": self.posterior
        }, f)
