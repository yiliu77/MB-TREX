import os
import random

import torch
import wandb
from torch.nn import functional
from torch.optim import Adam

from models.architectures.gaussian_policy import ContGaussianPolicy
from models.architectures.utils import polyak_update
from models.architectures.value_networks import ContTwinQNet
from models.replay_buffer import ReplayBuffer
from models.rnd import *


class ContSAC:
    def __init__(self, action_dim, env, device, memory_size=2e2, warmup_games=10, batch_size=64, lr=0.0001, gamma=0.99, tau=0.003, alpha=0.2,
                 ent_adj=False, target_update_interval=1, n_games_til_train=1, n_updates_per_train=2):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory_size = memory_size
        self.warmup_games = warmup_games
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

        self.env = env
        self.action_range = (env.action_space.low, env.action_space.high)
        self.policy = ContGaussianPolicy(action_dim, self.action_range).to(self.device)
        self.policy_opt = Adam(self.policy.parameters(), lr=lr)

        self.twin_q = ContTwinQNet(action_dim).to(self.device)
        self.twin_q_opt = Adam(self.twin_q.parameters(), lr=lr)
        self.target_twin_q = ContTwinQNet(action_dim).to(self.device)
        polyak_update(self.twin_q, self.target_twin_q, 1)

        self.rnd = RND().to(self.device)
        self.rnd_target = RNDTarget().to(self.device)
        self.rnd_opt = Adam(self.rnd.parameters(), lr=lr / 3)
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        self.normalization = RunningMeanStd(shape=(1, 64, 64, 3))
        self.reward_normalization = RunningMeanStd()

        self.tau = tau
        self.gamma = gamma
        self.n_until_target_update = target_update_interval
        self.n_games_til_train = n_games_til_train
        self.n_updates_per_train = n_updates_per_train

        self.alpha = alpha
        self.ent_adj = ent_adj
        if ent_adj:
            self.target_entropy = -len(self.action_range)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = Adam([self.log_alpha], lr=lr)

        self.total_train_steps = 0

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.as_tensor(state[np.newaxis, :].copy(), dtype=torch.float32).to(self.device)
            if deterministic:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]

    def calc_reward_in(self, states):
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        return torch.sum(torch.square(self.rnd_target(states) - self.rnd(states)), dim=1).detach().cpu().numpy()

    def train_step(self, states, actions, rewards, rewards_in, next_states, next_normalized_states, done_masks, index):
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
            # rewards = torch.as_tensor(rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
            rewards_in = torch.as_tensor(rewards_in[:, np.newaxis], dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
            next_normalized_states = torch.as_tensor(next_normalized_states, dtype=torch.float32).to(self.device)
            done_masks = torch.as_tensor(done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            next_q = self.target_twin_q(next_states, next_actions)[0]
            v = next_q - self.alpha * next_log_probs
            expected_q = rewards_in + done_masks * self.gamma * v  # 3, 10, 20

        # Q backprop
        q_val, pred_q1, pred_q2 = self.twin_q(states, actions)
        q_loss = functional.mse_loss(pred_q1, expected_q) + functional.mse_loss(pred_q2, expected_q)

        self.twin_q_opt.zero_grad()
        q_loss.backward()
        self.twin_q_opt.step()

        # Policy backprop
        s_action, s_log_prob, _ = self.policy.sample(states)
        policy_loss = self.alpha * s_log_prob - self.twin_q(states, s_action)[0]
        policy_loss = policy_loss.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        forward_fn = nn.MSELoss(reduction='none')
        self.rnd_opt.zero_grad()
        forward_loss = forward_fn(self.rnd(next_normalized_states), self.rnd_target(next_normalized_states).detach()).mean(-1)
        mask = torch.rand(len(forward_loss)).to(self.device)
        mask = (mask < 0.25).float().to(self.device)
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))

        forward_loss.backward()
        self.rnd_opt.step()

        if self.ent_adj:
            alpha_loss = -(self.log_alpha * (s_log_prob + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.alpha = self.log_alpha.exp()

        if self.total_train_steps % self.n_until_target_update == 0:
            polyak_update(self.twin_q, self.target_twin_q, self.tau)

        losses = {'Loss/Policy Loss': policy_loss.item(),
                  'Loss/Q Loss': q_loss.item(),
                  'Stats/Avg Q Val': q_val.mean().item(),
                  'Stats/Avg Q Next Val': next_q.mean().item(),
                  'Stats/Avg Alpha': self.alpha.item() if self.ent_adj else self.alpha,
                  "Loss/Forward Loss": forward_loss.item()}
        return losses

    def train(self, num_games, deterministic=False):
        all_states = []
        all_states_lowdim = []
        all_actions = []
        all_lengths = []

        self.policy.train()
        self.twin_q.train()
        for i in range(num_games):
            total_reward = 0
            total_reward_in = 0
            n_steps = 0
            done = False
            states, actions, rewards, reward_ins, next_states, done_masks = [], [], [], [], [], []
            states_lowdim = []
            state, info = self.env.reset()
            while not done:
                if i <= 2 * self.warmup_games or random.random() < 0.1:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(state, deterministic)

                next_state, reward, done, new_info = self.env.step(action)
                done_mask = 1.0 if n_steps == self.env._max_episode_steps - 1 else float(not done)

                reward_in = self.calc_reward_in(((next_state[np.newaxis, :] - self.normalization.mean) / np.sqrt(self.normalization.var)).clip(-8, 8))[0]

                states.append(state)
                states_lowdim.append(info["lowdim"])
                actions.append(action)
                rewards.append(reward)
                reward_ins.append(reward_in)
                next_states.append(next_state)
                done_masks.append(done_mask)

                n_steps += 1
                total_reward += reward
                total_reward_in += reward_in
                state = next_state
                info = new_info

            mean, std, count = np.mean(reward_ins), np.std(reward_ins), len(reward_ins)
            self.reward_normalization.update_from_moments(mean, std ** 2, count)
            reward_ins = np.array(reward_ins) / np.sqrt(self.reward_normalization.var)
            if i > self.warmup_games:
                for point in zip(states, actions, rewards, reward_ins, next_states, done_masks):
                    self.memory.add(*point)

            states = np.array(states)
            all_states.append(states)
            all_actions.append(np.array(actions))
            all_states_lowdim.append(np.array(states_lowdim))
            all_lengths.append(len(states))

            self.normalization.update(states)

            if i % 10 == 0:
                gen_images = wandb.Image(np.reshape(states, (states.shape[0] * states.shape[1], states.shape[2], states.shape[3])), caption="Sequences")
                # x_bounds = [-0.3, 0.3]
                # y_bounds = [-0.3, 0.3]
                #
                # obs_states, actions = [], []
                # x_pts = 64
                # y_pts = 64
                # for y in np.linspace(x_bounds[0], x_bounds[1], y_pts):
                #     for x in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                #         obs = self.env.reset(pos=(x, y))
                #         obs_states.append(obs)
                #
                # obs_states = np.array(obs_states)
                # obs_states = ((obs_states - self.normalization.mean) / np.sqrt(self.normalization.var)).clip(-8, 8)
                # obs_states = torch.as_tensor(obs_states).float().to("cuda")
                # costs = self.calc_reward_in(obs_states)
                # grid = costs.reshape(y_pts, x_pts) / np.max(costs)
                # grid_image = wandb.Image(grid[:, :, None] * 255 * 0.8 + 0.2 * self.env.reset(), caption="RND")
                # avg_images = wandb.Image(np.array(self.normalization.mean[0, :, :, :]), caption="Avg")
                # traverse_images = wandb.Image(np.max(states, axis=0), caption="Traverse")
                wandb.log({"Sequences": gen_images})

            if i >= 2 * self.warmup_games:
                if i % self.n_games_til_train == 0:
                    for j in range(n_steps * self.n_updates_per_train):
                        self.total_train_steps += 1
                        s, a, r, r_in, s_, d = self.memory.sample()
                        train_info = self.train_step(s, a, r, r_in, s_, ((s_ - self.normalization.mean) / np.sqrt(self.normalization.var)).clip(-8, 8), d, i)
                        if i % 5 == 0 and j == 0:
                            train_info['Env/Rewards'] = total_reward
                            train_info['Env/Reward Ins'] = total_reward_in
                            train_info['Env/N_Steps'] = n_steps
                            wandb.log(train_info)

            print("index: {}, steps: {}, total_rewards: {}".format(i, n_steps, total_reward))
        padded_all_states = np.zeros((len(all_states), np.max(all_lengths), all_states[0].shape[1], all_states[0].shape[2], all_states[0].shape[3]))
        padded_all_states_lowdim = np.zeros((len(all_states_lowdim), np.max(all_lengths), len(all_states_lowdim[0][0])))
        padded_all_actions = np.zeros((len(all_actions), np.max(all_lengths), all_actions[0].shape[1]))
        for i in range(len(padded_all_states)):
            padded_all_states[i, :len(all_states[i]), :, :, :] = all_states[i]
            padded_all_states_lowdim[i, :len(all_states[i]), :] = all_states_lowdim[i]
            padded_all_actions[i, :len(all_actions[i]), :] = all_actions[i]
        return padded_all_states, padded_all_states_lowdim, padded_all_actions, np.array(all_lengths)

    def eval(self, num_games, render=True):
        self.policy.eval()
        self.twin_q.eval()
        for i in range(num_games):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                action = self.get_action(state, deterministic=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            print(i, total_reward)

    def save_model(self, folder_name):
        path = 'saved_weights/' + folder_name
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), path + '/policy')
        torch.save(self.twin_q.state_dict(), path + '/twin_q_net')

    # Load model parameters
    def load_model(self, folder_name, device):
        path = 'saved_weights/' + folder_name
        self.policy.load_state_dict(torch.load(path + '/policy', map_location=torch.device(device)))
        self.twin_q.load_state_dict(torch.load(path + '/twin_q_net', map_location=torch.device(device)))

        polyak_update(self.twin_q, self.target_twin_q, 1)
