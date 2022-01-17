import numpy as np
import torch


class VisualMPC:
    def __init__(self, video_prediction, cost_fn, horizon, env):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.video_prediction = video_prediction
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.env = env

        self.sample_size = 128
        self.elite_size = 10

    def act(self, obs, t=1):
        with torch.no_grad():
            obs = torch.as_tensor(obs / 255).to(self.device).float()
            obs = torch.permute(obs, (2, 0, 1)).unsqueeze(0)
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
                trajectory, _ = self.video_prediction.predict_states(curr_states, action_samples, 8)#self.horizon - t) # TODO check both time and future prediction

                reshaped_trajectory = trajectory[:8].reshape(-1, trajectory.shape[2], trajectory.shape[3], trajectory.shape[4])
                reshaped_actions = action_samples[:8].reshape(-1, action_samples.shape[2])
                costs = self.cost_fn(reshaped_trajectory, reshaped_actions)
                costs = costs.reshape(8, -1)
                costs = torch.sum(costs, dim=0).squeeze()  # costs of all action sequences

                sortid = costs.argsort()
                actions_sorted = action_samples[:, sortid, :]
                actions_ranked = actions_sorted[:, :self.elite_size, :]

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
