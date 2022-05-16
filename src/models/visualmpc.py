import torch
import wandb
from torch.optim import Adam
import copy

from models.rnd import *


class VisualMPC:
    def __init__(self, video_prediction, cost_fn, horizon, env, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = params

        self.video_prediction = video_prediction
        self.cost_fn = cost_fn
        self.rnd = RND().to(self.device)
        self.rnd_target = RNDTarget().to(self.device)
        self.rnd_opt = Adam(self.rnd.parameters(), lr=0.005)
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        self.normalization = RunningMeanStd(shape=(1, 64, 64, 3))
        self.rnd_rew_normalization = RunningMeanStd()
        self.cost_normalization = RunningMeanStd()

        self.horizon = horizon
        self.env = env

        self.sample_size = 64
        self.elite_size = 7

        self.rnd_iteration = 30
        self.rnd_counter = -1
        self.max_rnd_coeff = 2.0
        self.min_rnd_coeff = 0.0

        self.past_states = []

    def act(self, obs, t=0, evaluate=False):
        if t == 0 and len(self.past_states) != 0:
            past_states = torch.as_tensor(np.concatenate(self.past_states, axis=0)).float().to(self.device)
            normalized_states = ((past_states - torch.as_tensor(self.normalization.mean).float().to(self.device)) /
                                 torch.as_tensor(np.sqrt(self.normalization.var)).float().to(self.device)).clip(-8, 8).detach()
            losses = []
            for _ in range(10):
                forward_fn = nn.MSELoss(reduction='none')
                self.rnd_opt.zero_grad()
                forward_loss = forward_fn(self.rnd(normalized_states), self.rnd_target(normalized_states).detach()).mean(-1)
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < 0.25).float().to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                forward_loss.backward()
                self.rnd_opt.step()
                losses.append(forward_loss.item())
            wandb.log({"VisualMPC/RNDLoss": np.mean(losses)})
            self.past_states = []

            self.rnd_counter += 1

        query_states_list = []
        with torch.no_grad():
            obs = torch.as_tensor(obs / 255).to(self.device).float()
            obs = obs.permute((2, 0, 1)).unsqueeze(0)
            curr_states = obs.repeat(self.sample_size, 1, 1, 1)
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

            average_trajectories = []
            prev_normalization = copy.deepcopy(self.normalization)
            prev_rnd_rew_normalization = copy.deepcopy(self.rnd_rew_normalization)
            # prev_cost_normalization = copy.deepcopy(self.cost_normalization)
            for itr in range(self.params["iter_per_action"]):
                trajectory = self.video_prediction.predict_states(curr_states, action_samples, self.params["future_frames"])[:self.params["future_frames"]]

                # Calculate cost and rnd rewards
                traj_cost = self.cost_fn(trajectory)

                permuted_states = trajectory.reshape(-1, trajectory.shape[2], trajectory.shape[3], trajectory.shape[4])
                permuted_states = permuted_states.detach().permute((0, 2, 3, 1))
                self.normalization.update(permuted_states.cpu().numpy())

                normalized_states = ((permuted_states - torch.as_tensor(prev_normalization.mean).float().to(self.device)) /
                                     torch.as_tensor(np.sqrt(prev_normalization.var)).float().to(self.device)).clip(-8, 8).detach()
                rnd_rewards = torch.sum(torch.square(self.rnd_target(normalized_states) - self.rnd(normalized_states)), dim=1).detach()
                rnd_rewards = torch.sum(rnd_rewards.reshape(trajectory.shape[0], -1), dim=0)

                # Calculate final cost
                normalized_traj_cost = traj_cost
                normalized_rnd_rewards = rnd_rewards / torch.as_tensor(np.sqrt(prev_rnd_rew_normalization.var)).float().to(self.device)
                rnd_coeff = max(self.min_rnd_coeff, self.max_rnd_coeff - self.rnd_counter * (self.max_rnd_coeff - self.min_rnd_coeff) / self.rnd_iteration)
                costs = normalized_traj_cost - (rnd_coeff * normalized_rnd_rewards if not evaluate else 0)
                wandb.log({"VisualMPC/RND Rewards": torch.mean(normalized_rnd_rewards).item(), "VisualMPC/Traj Costs": torch.mean(normalized_traj_cost).item(),
                           "VisualMPC/RND Coeff": rnd_coeff, "VisualMPC/Combined Cost": torch.mean(costs).item()})

                # Update normalization
                rnd_rewards_np = rnd_rewards.cpu().numpy()
                mean, std, count = np.mean(rnd_rewards_np), np.std(rnd_rewards_np), len(rnd_rewards_np)
                self.rnd_rew_normalization.update_from_moments(mean, std ** 2, count)
                traj_cost_np = traj_cost.cpu().numpy()
                mean, std, count = np.mean(traj_cost_np), np.std(traj_cost_np), len(traj_cost_np)
                self.cost_normalization.update_from_moments(mean, std ** 2, count)

                sortid = costs.argsort()
                actions_sorted = action_samples[:, sortid, :]
                actions_ranked = actions_sorted[:, :self.elite_size, :]

                average_trajectories.append(torch.mean(trajectory[:, sortid[:self.elite_size], ...], dim=1))
                query_states_list.append(trajectory.detach().cpu().numpy()[:, sortid[0], ...])

                mean, std = actions_ranked.mean(1), actions_ranked.std(1)
                smp = torch.empty(action_samples.shape).normal_(mean=0, std=1).cuda()
                mean = mean.unsqueeze(1).repeat(1, self.sample_size, 1)
                std = std.unsqueeze(1).repeat(1, self.sample_size, 1)
                action_samples = smp * std + mean
                action_samples = torch.clamp(
                    action_samples,
                    min=self.env.action_space.low[0],
                    max=self.env.action_space.high[0])

            action = mean[0][0]
            mean_trajectory = self.video_prediction.predict_states(curr_states, mean, self.horizon)[:self.params["future_frames"], 0, ...]
            mean_trajectory_np = mean_trajectory.detach().cpu().numpy()

            visualized_traj = torch.stack(average_trajectories, dim=0)
            visualized_traj = visualized_traj.permute((0, 1, 3, 4, 2))
            visualized_traj = visualized_traj.permute((0, 2, 1, 3, 4))
            visualized_traj = visualized_traj.reshape(visualized_traj.shape[0] * visualized_traj.shape[1], visualized_traj.shape[2] * visualized_traj.shape[3], visualized_traj.shape[4])
            traj_images = wandb.Image(visualized_traj.detach().cpu().numpy(), caption="Sequences")
            avg_images = wandb.Image(np.mean(mean_trajectory_np, axis=0).transpose((1, 2, 0)), caption="Sequences")
            wandb.log({"VisualMPC/Gen_Traj": traj_images,
                       "VisualMPC/Mean_Traj": avg_images})

            self.past_states.append(np.transpose(mean_trajectory_np, (0, 2, 3, 1)))

        query_states_list.append(mean_trajectory_np)
        return action.detach().cpu().numpy(), query_states_list
