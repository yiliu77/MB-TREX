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
        self.video_prediction.to_half()
        self.cost_fn = cost_fn
        self.rnd = RND().to(self.device)
        self.rnd_target = RNDTarget().to(self.device)
        self.rnd_opt = Adam(self.rnd.parameters(), lr=0.007)
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        self.normalization = RunningMeanStd(shape=(1, 64, 64, 1))
        self.rnd_rew_normalization = RunningMeanStd()
        self.normalization.cuda()
        self.rnd_rew_normalization.cuda()
        print(self.normalization.mean.shape)
        # self.cost_normalization = RunningMeanStd()

        self.horizon = horizon
        self.env = env

        self.sample_size = 64
        self.elite_size = 7

        self.rnd_iteration = 100
        self.rnd_counter = -1
        self.max_rnd_coeff = 3.0
        self.min_rnd_coeff = 0.1

        self.past_segments = []
        self.past_states = []
        self.past_actions = []

    def act(self, obs, t=0, evaluate=False):
        if t == 0 and len(self.past_segments) != 0:
            losses = []
            for _ in range(10):
                concatenated_segments = np.concatenate(self.past_segments, axis=0)
                indices = np.random.choice(range(len(concatenated_segments)), size=min(32, len(concatenated_segments)), replace=False)
                past_states = torch.as_tensor(concatenated_segments)[indices].float().to(self.device)
                normalized_states = ((past_states - self.normalization.mean.float().to(self.device)) /
                                     torch.sqrt(self.normalization.var).float().to(self.device)).clip(-8, 8).detach()

                forward_fn = nn.MSELoss(reduction='none')
                self.rnd_opt.zero_grad()
                forward_loss = forward_fn(self.rnd(normalized_states), self.rnd_target(normalized_states).detach()).mean(-1)
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < 0.25).float().to(self.device)
                forward_loss = (forward_loss * mask).mean() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                forward_loss.backward()
                self.rnd_opt.step()

                losses.append(forward_loss.item())
                if forward_loss.item() > 100:
                    print(past_states[0].shape)
                    error_dict = {"VisualMPC/Error" + str(i): wandb.Image(past_states[i, :, :, 0]) for i in range(len(past_states))}
                    wandb.log(error_dict)
            wandb.log({"VisualMPC/RNDLoss": np.mean(losses)})
            self.past_segments = []

            self.rnd_counter += 1

        with torch.no_grad():
            query_states_list = []
            query_actions_list = []

            obs = torch.as_tensor(obs / 255).to(self.device)
            obs = obs.permute((2, 0, 1)).unsqueeze(0)
            if t == 0:
                self.past_states = [obs.repeat(self.sample_size, 1, 1, 1).float()] * self.video_prediction.n_past
                self.past_actions = [np.zeros_like(self.env.action_space.sample())] * (self.video_prediction.n_past - 1)
            self.past_states.append(obs.repeat(self.sample_size, 1, 1, 1).float())
            self.past_states.pop(0)
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
            past_actions_tensor = torch.as_tensor(np.array(self.past_actions)).to(self.device).float().unsqueeze(1).repeat(1, self.sample_size, 1)
            for itr in range(self.params["iter_per_action"]):
                all_actions = (torch.cat([past_actions_tensor, action_samples], dim=0) if len(self.past_actions) > 0 else action_samples)[:self.params["future_frames"]].float()
                trajectory = self.video_prediction.predict_states(self.past_states, all_actions, self.params["future_frames"])[:self.params["future_frames"]].float()
                traj_cost = self.cost_fn(trajectory, all_actions)

                # Calc RND
                permuted_states = trajectory.reshape(-1, trajectory.shape[2], trajectory.shape[3], trajectory.shape[4])
                permuted_states = permuted_states.permute((0, 2, 3, 1))
                self.normalization.update(permuted_states)

                normalized_states = ((permuted_states - prev_normalization.mean) / torch.sqrt(prev_normalization.var)).clip(-8, 8).float()
                rnd_rewards = torch.mean(torch.square(self.rnd_target(normalized_states) - self.rnd(normalized_states)), dim=1).detach()
                rnd_rewards = torch.mean(rnd_rewards.reshape(trajectory.shape[0], -1), dim=0)

                # Update normalization
                mean, std, count = torch.mean(rnd_rewards), torch.std(rnd_rewards), len(rnd_rewards)
                self.rnd_rew_normalization.update_from_moments(mean, std ** 2, count)

                # Calculate final cost
                normalized_traj_cost = traj_cost
                normalized_rnd_rewards = rnd_rewards / torch.sqrt(self.rnd_rew_normalization.var).float().to(self.device)
                # TODO remove mean subtraction 
                rnd_coeff = max(self.min_rnd_coeff, self.max_rnd_coeff - self.rnd_counter * (self.max_rnd_coeff - self.min_rnd_coeff) / self.rnd_iteration)
                costs = normalized_traj_cost - rnd_coeff * normalized_rnd_rewards.detach()
                wandb.log({"VisualMPC/RND Rewards": torch.mean(normalized_rnd_rewards).item(), "VisualMPC/Traj Costs": torch.mean(normalized_traj_cost).item(),
                           "VisualMPC/RND Coeff": rnd_coeff, "VisualMPC/Combined Cost": torch.mean(costs).item(), "VisualMPC/Reward Normalization Variance": self.rnd_rew_normalization.var.item(),
                           "VisualMPC/RND Reward Max": torch.max(rnd_rewards).item(), "VisualMPC/RND Reward Min": torch.min(rnd_rewards),
                           "VisualMPC/RND Reward Std": torch.std(rnd_rewards).item(),
                           "VisualMPC/RND Target Max": torch.max(self.rnd_target(normalized_states)).item(), "VisualMPC/RND Target Min": torch.min(self.rnd_target(normalized_states)),
                           "VisualMPC/RND Max": torch.max(self.rnd(normalized_states)).item(), "VisualMPC/RND Min": torch.min(self.rnd(normalized_states))})

                # traj_cost_np = traj_cost.cpu().numpy()
                # mean, std, count = np.mean(traj_cost_np), np.std(traj_cost_np), len(traj_cost_np)
                # self.cost_normalization.update_from_moments(mean, std ** 2, count)
                # print(time.time() - start)

                sortid = costs.argsort()
                actions_sorted = action_samples[:, sortid, :]
                actions_ranked = actions_sorted[:, :self.elite_size, :]

                average_trajectories.append(torch.mean(trajectory[:, sortid[:self.elite_size], ...], dim=1))
                query_states_list.append(trajectory[:, sortid[0], :, :, :])
                query_actions_list.append(all_actions[:, sortid[0], ...])

                mean_actions, std_actions = actions_ranked.mean(1), actions_ranked.std(1)
                smp = torch.randn_like(action_samples)
                mean_actions = mean_actions.unsqueeze(1).repeat(1, self.sample_size, 1)
                std_actions = std_actions.unsqueeze(1).repeat(1, self.sample_size, 1)
                action_samples = smp * std_actions + mean_actions
                action_samples = torch.clamp(
                    action_samples,
                    min=self.env.action_space.low[0],
                    max=self.env.action_space.high[0])

            action = mean_actions[0][0]
            all_actions = torch.cat([past_actions_tensor, mean_actions], dim=0) if len(self.past_actions) > 0 else mean_actions
            mean_trajectory = self.video_prediction.predict_states(self.past_states, all_actions, self.horizon)[:self.params["future_frames"], 0, ...]
            mean_trajectory_np = mean_trajectory.detach().cpu().numpy()
            all_actions = all_actions[:self.params["future_frames"], 0, ...]

            visualized_traj = torch.stack(average_trajectories, dim=0)
            visualized_traj = visualized_traj.permute((0, 1, 3, 4, 2))
            visualized_traj = visualized_traj.permute((0, 2, 1, 3, 4))
            visualized_traj = visualized_traj.reshape(visualized_traj.shape[0] * visualized_traj.shape[1], visualized_traj.shape[2] * visualized_traj.shape[3], visualized_traj.shape[4])
            traj_images = wandb.Image(visualized_traj.detach().cpu().numpy(), caption="Sequences")
            avg_images = wandb.Image(np.transpose(mean_trajectory_np, (0, 2, 3, 1)).reshape(-1, mean_trajectory_np.shape[3], mean_trajectory_np.shape[1]), caption="Sequences")
            wandb.log({"VisualMPC/Gen_Traj": traj_images,
                       "VisualMPC/Mean_Traj": avg_images})

            self.past_segments.append(np.transpose(mean_trajectory_np, (0, 2, 3, 1)))

        query_states_list.append(mean_trajectory)
        query_actions_list.append(all_actions)
        self.past_actions.append(action.detach().cpu().numpy())
        self.past_actions.pop(0)

        for i, query_states in enumerate(query_states_list):
            query_states_list[i] = query_states.cpu().numpy()
        for i, query_actions in enumerate(query_actions_list):
            query_actions_list[i] = query_actions.cpu().numpy()
        return action.detach().cpu().numpy(), query_states_list, query_actions_list
