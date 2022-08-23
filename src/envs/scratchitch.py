from assistive_gym.envs import ScratchItchPR2Env

import numpy as np
import pybullet as p

class ScratchItchPR2EnvMB(ScratchItchPR2Env):
    def __init__(self):
        super(ScratchItchPR2EnvMB, self).__init__()
        self.horizon=200
        self.setup_camera()

    def get_expert_reward(self, state, action):
        reward_distance = -np.linalg.norm(state[:3], state[10:13])
        reward_action = - np.linalg.norm(action)
        reward_force_scratch = 0
        if -reward_distance < 0.025 and np.linalg.norm(state[:3] - self._prev_target_contact_pos) > 0.01 and state[-1] < 10:
            reward_force_scratch = 5
            self._prev_target_contact_pos = state[:3]
        reward = self.config('distance_weight') * reward_distance + self.config('action_weight') * reward_action + self.config('scratch_reward_weight')*reward_force_scratch
        return reward

    def _get_obs(self, use_images=False):
        if use_images:
            # import pdb; pdb.set_trace()
            w, h, img, depth, _ = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix)
            # img[img[:, :, 0] > 200] = [255, 0, 0]
            return img
        obs = super()._get_obs()
        # handpicked_features = np.array([self.tool_force_at_target])
        # obs = np.concatenate([obs[:7], obs[10:13], obs[-1:]])
        # obs = np.concatenate((obs[7:10], handpicked_features))
        return obs

    def reset(self):
        np.random.seed(0)
        self._prev_target_contact_pos = np.zeros(3)
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
    
        # distance = np.linalg.norm(obs[7:10])
        tool_force_at_target = info['tool_force_at_target']
        handpicked_features = np.array([tool_force_at_target])
        obs = np.concatenate((obs, handpicked_features))
    
        return obs, reward, done, info

