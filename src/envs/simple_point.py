import os

import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim


class SimplePoint(Env, utils.EzPickle):
    def __init__(self, horizon, max_force):
        utils.EzPickle.__init__(self)
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'assets/simple_point.xml')

        self.name = "MazeH{}_MF{}".format(horizon, max_force)

        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = horizon
        self._max_episode_steps = self.horizon

        self.max_force = max_force

        self.observation_space = self._get_obs().shape
        self.action_space = Box(-max_force * np.ones(2), max_force * np.ones(2))

        self.steps = 0

        self.goal = np.zeros((2,))
        self.goal[0] = -0.1
        self.goal[1] = -0.25

    def get_goal(self):
        self.sim.data.qpos[0] = self.goal[0]
        self.sim.data.qpos[1] = self.goal[1]
        self.sim.forward()
        obs = self._get_obs()
        return obs

    def step(self, action):
        action = np.clip(np.array(action), -self.max_force, self.max_force)
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = action

        cur_obs = self._get_obs()
        for _ in range(500):
            self.sim.step()
        obs = self._get_obs()
        gt_state = np.concatenate([self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy()])
        self.sim.data.qvel[:] = 0

        self.steps += 1
        done = self.steps >= self.horizon  # or self.get_distance_score() < self.goal_thresh  # NOTE: not calling done signal when in goal set to make all eps fixed length?
        cost = self.get_distance_score()

        info = {
            "cost": cost,
            "state": cur_obs,
            "next_state": obs,
            "action": action,
            "gt_state": gt_state
        }
        return obs, -cost, done, info

    def _get_obs(self):
        rendered_img = self.sim.render(64, 64, camera_name="cam0")
        rendered_img[rendered_img[:, :, 0] > 200] = [255, 0, 0]
        return rendered_img

    def reset(self, pos=None):
        if pos is None:
            self.sim.data.qpos[0] = np.random.uniform(-0.2, 0.2)
            self.sim.data.qpos[1] = np.random.uniform(-0.2, 0.2)
        else:
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        self.sim.forward()

        self.steps = 0
        return self._get_obs()

    def get_distance_score(self):
        d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:]) ** 2))
        return d

    def visualize_rewards(self, filename, cost_model):
        x_bounds = [-0.25, 0.25]
        y_bounds = [-0.25, 0.25]

        states = []
        x_pts = 40
        y_pts = int(x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                obs = self.reset(pos=(x, y))
                states.append(obs)

        with torch.no_grad():
            states = torch.as_tensor(np.array(states)[None, :, :, :, :]).float().to("cuda")
            states = states.permute((0, 1, 4, 2, 3)) / 255
            costs = cost_model.get_value(states)
            costs = (costs - torch.min(costs)) / (torch.max(costs) - torch.min(costs))

            grid = costs.detach().cpu().numpy()
            grid = grid.reshape(y_pts, x_pts)

        background = cv2.resize(self.get_goal(), (x_pts, y_pts))
        plt.imshow(background)
        plt.imshow(grid.T, alpha=0.6)
        plt.colorbar()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def render(self, mode="human"):
        raise NotImplementedError
