import os

import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim


class Maze(Env, utils.EzPickle):
    def __init__(self, horizon, use_images, dense_rewards, max_force, goal_thresh):
        utils.EzPickle.__init__(self)
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'assets/simple_maze.xml')

        self.name = "Maze{}_H{}R{}MF{}GT{}".format("1D" if use_images else "2D", horizon, int(dense_rewards), max_force, goal_thresh)

        self.sim = MjSim(load_model_from_path(filename))
        self.horizon = horizon
        self._max_episode_steps = self.horizon

        self.use_images = use_images
        self.max_force = max_force
        self.dense_rewards = dense_rewards
        self.goal_thresh = goal_thresh

        if self.use_images:
            self.observation_space = self._get_obs().shape
        else:
            self.observation_space = Box(-0.3, 0.3, shape=self._get_obs().shape)
        self.action_space = Box(-max_force * np.ones(2), max_force * np.ones(2))

        self.steps = 0
        self.gain = 10

        self.goal = np.zeros((2,))
        self.goal[0] = 0.3
        self.goal[1] = -0.3

    def get_goal(self):
        self.sim.data.qpos[0] = self.goal[0]
        self.sim.data.qpos[1] = self.goal[1]
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
        if not self.dense_rewards:
            cost = (self.get_distance_score() > self.goal_thresh).astype(float)
        else:
            cost = self.get_distance_score()

        info = {
            "cost": cost,
            "state": cur_obs,
            "next_state": obs,
            "action": action,
            "success": int(cost < self.goal_thresh),
            "gt_state": gt_state
        }
        return obs, -cost, done, info

    def _get_obs(self):
        if self.use_images:
            rendered_img = self.sim.render(64, 64, camera_name="cam0")
            rendered_img[rendered_img[:, :, 0] > 200] = [255, 0, 0]
            return rendered_img
        # joint poisitions and velocities
        state = np.concatenate([self.sim.data.qpos[:].copy(), self.sim.data.qvel[:].copy()])
        return state[:2]  # State is just (x, y) now

    def reset(self, difficulty='h', check_constraint=True, pos=()):
        pos = tuple(pos)
        if len(pos):
            self.sim.data.qpos[0] = pos[0]
            self.sim.data.qpos[1] = pos[1]
        else:
            if difficulty is None:
                self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
            elif difficulty == 'e':
                self.sim.data.qpos[0] = np.random.uniform(0.14, 0.22)
            elif difficulty == 'm':
                self.sim.data.qpos[0] = np.random.uniform(-0.04, 0.04)
            elif difficulty == 'h':
                self.sim.data.qpos[0] = np.random.uniform(-0.22, -0.13)
            self.sim.data.qpos[1] = np.random.uniform(-0.25, 0.25)

        self.steps = 0
        # Randomize wal positions
        w1 = -0.08  # np.random.uniform(-0.2, 0.2)
        w2 = 0.08  # np.random.uniform(-0.2, 0.2)
        self.sim.model.geom_pos[5, 1] = 0.5 + w1
        self.sim.model.geom_pos[7, 1] = -0.25 + w1
        self.sim.model.geom_pos[6, 1] = 0.4 + w2
        self.sim.model.geom_pos[8, 1] = -0.25 + w2
        self.sim.forward()
        constraint = int(self.sim.data.ncon > 3)
        if constraint and check_constraint:
            if not len(pos):
                self.reset(difficulty)
        return self._get_obs()

    def get_distance_score(self):
        d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:]) ** 2))
        return d

    def render(self, mode="human"):
        raise NotImplementedError

    def visualize_rewards(self, filename, cost_model):
        action = self.action_space.sample()
        x_bounds = [-0.25, 0.25]
        y_bounds = [-0.25, 0.25]

        states, actions = [], []
        x_pts = 60
        y_pts = int(x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
        for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
            for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
                obs = self.reset(pos=(x, y))
                states.append(obs)
                actions.append(action)

        states = torch.as_tensor(np.array(states)).float().to("cuda")
        states = states.permute((0, 3, 1, 2))
        actions = torch.as_tensor(np.array(actions)).float().to("cuda")
        costs = cost_model.get_value(states, actions)

        grid = costs.detach().cpu().numpy()
        grid = grid.reshape(y_pts, x_pts)

        background = cv2.resize(self.reset(), (x_pts, y_pts))
        plt.imshow(background)
        plt.imshow(grid.T, alpha=0.6)
        plt.colorbar()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
