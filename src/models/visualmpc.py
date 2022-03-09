import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.optim import Adam
import wandb

from models.sac import RND, RNDTarget, RunningMeanStd


class VisualMPC:
    def __init__(self, video_prediction, cost_fn, horizon, env):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.video_prediction = video_prediction
        self.cost_fn = cost_fn
        self.rnd = RND().to(self.device)
        self.rnd_target = RNDTarget().to(self.device)
        self.rnd_opt = Adam(self.rnd.parameters(), lr=0.002)
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        self.normalization = RunningMeanStd(shape=(1, 64, 64, 3))
        self.reward_normalization = RunningMeanStd()

        self.horizon = horizon
        self.env = env

        self.sample_size = 32
        self.elite_size = 5

    def act(self, obs, t=1):
        obs = torch.as_tensor(obs / 255).to(self.device).float()
        obs = obs.permute((2, 0, 1)).unsqueeze(0)
        self.video_prediction.encoder.eval()

        action_samples = []
        for _ in range(self.sample_size):
            action_trajs = []
            for j in range(self.horizon):
                action_trajs.append(self.env.action_space.sample())
            action_trajs = np.stack(action_trajs)
            action_samples.append(action_trajs)
        action_samples = np.stack(action_samples)
        action_samples = torch.as_tensor(action_samples).to(self.device)
        action_samples = action_samples.permute(1, 0, 2)

        for itr in range(10):  # TODO
            curr_states = obs.repeat(self.sample_size, 1, 1, 1)
            trajectory, _ = self.video_prediction.predict_states(curr_states, action_samples, 8)  # self.horizon - t) # TODO check both time and future prediction

            reshaped_trajectory = trajectory[:8].reshape(-1, trajectory.shape[2], trajectory.shape[3], trajectory.shape[4])
            reshaped_actions = action_samples[:8].reshape(-1, action_samples.shape[2])
            costs = self.cost_fn(reshaped_trajectory, reshaped_actions)

            # Train RND
            permuted_trajectory = reshaped_trajectory.detach().permute((0, 2, 3, 1))
            normalized_states = ((permuted_trajectory - torch.as_tensor(self.normalization.mean).float().to(self.device)) /
                                 torch.as_tensor(np.sqrt(self.normalization.var)).float().to(self.device)).clip(-8, 8).detach()
            reward_ins = torch.sum(torch.square(self.rnd_target(normalized_states) - self.rnd(normalized_states)), dim=1).detach()
            costs -= (reward_ins - torch.as_tensor(self.reward_normalization.mean).float().to(self.device)) / \
                     torch.as_tensor(np.sqrt(self.reward_normalization.var)).float().to(self.device)

            reward_ins = reward_ins.cpu().numpy()
            mean, std, count = np.mean(reward_ins), np.std(reward_ins), len(reward_ins)
            self.reward_normalization.update_from_moments(mean, std ** 2, count)
            self.normalization.update(permuted_trajectory.cpu().numpy())  # TODO move normalization out

            forward_fn = nn.MSELoss(reduction='none')
            self.rnd_opt.zero_grad()
            forward_loss = forward_fn(self.rnd(normalized_states), self.rnd_target(normalized_states).detach()).mean(-1)
            mask = torch.rand(len(forward_loss)).to(self.device)
            mask = (mask < 0.25).float().to(self.device)
            forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
            forward_loss.backward()
            self.rnd_opt.step()

            costs = costs.reshape(8, -1)
            costs = torch.sum(costs, dim=0).squeeze()  # costs of all action sequences

            sortid = costs.argsort()
            actions_sorted = action_samples[:, sortid, :]
            actions_ranked = actions_sorted[:, :self.elite_size, :]

            visualized_traj = trajectory[:, sortid, :, :, :]
            visualized_traj = visualized_traj.permute((0, 1, 3, 4, 2))
            visualized_traj = visualized_traj.permute((0, 2, 1, 3, 4))
            visualized_traj = visualized_traj.reshape(visualized_traj.shape[0] * visualized_traj.shape[1], visualized_traj.shape[2] * visualized_traj.shape[3], visualized_traj.shape[4])
            traj_images = wandb.Image(visualized_traj.detach().cpu().numpy(), caption="Sequences")
            # wandb.log({"Sequences": gen_images})
            wandb.log({"VisualMPC/VideoCost_Iter_{}".format(itr): torch.mean(costs).item(),
                       "VisualMPC/RNDCost_Iter_{}".format(itr): -np.mean(reward_ins),
                       "VisualMPC/RNDLoss_Iter_{}".format(itr): forward_loss.item(),
                       "VisualMPC/EliteCost_Iter_{}".format(itr): np.mean(costs[sortid].detach().cpu().numpy()).item(),
                       "VisualMPC/Gen_Traj_Iter_{}".format(itr): traj_images})

            mean, std = actions_ranked.mean(1), actions_ranked.std(1)
            smp = torch.empty(action_samples.shape).normal_(mean=0, std=1).cuda()
            mean = mean.unsqueeze(1).repeat(1, self.sample_size, 1)
            std = std.unsqueeze(1).repeat(1, self.sample_size, 1)
            action_samples = smp * std + mean
            # TODO: Assuming action space is symmetric, true for maze and shelf for now
            action_samples = torch.clamp(
                action_samples,
                min=self.env.action_space.low[0],
                max=self.env.action_space.high[0])

        action = mean[0][0]
        mean_trajectory, _ = self.video_prediction.predict_states(curr_states, mean, self.horizon)
        return action.detach().cpu().numpy(), mean_trajectory.detach().cpu().numpy()[:8, 0, :, :, :], mean.detach().cpu().numpy()[:8, 0, :]
