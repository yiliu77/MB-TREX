
import cv2

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


class Hopper(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, params):
        self.horizon = params["horizon"]
        self.num_steps = 0
        self.dims = params["dims"]
        self.use_images = params["use_images"]
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'assets/hopper.xml')
        mujoco_env.MujocoEnv.__init__(self, filename, 4)
        utils.EzPickle.__init__(self)
        self.observation_space = self.observation_space.shape
    

    def step(self, a):
        self.num_steps += 1
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        self.do_simulation(a, self.frame_skip)
        self.do_simulation(a, self.frame_skip)

        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        return self._get_obs(), reward, self.horizon == self.num_steps, {}

    def _get_obs(self, use_images=None):
        if use_images is None:
            use_images = self.use_images
        if use_images:
            rendered_img = self.render(mode="rgb_array")
            rendered_img = rendered_img[50: -50, 50: -50, :]
            rendered_img = cv2.resize(rendered_img, tuple(self.dims))
            return rendered_img
        else:
            return np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])

    def reset(self, pos=(), **kwargs):
        self.num_steps = 0
        pos = tuple(pos)
        if len(pos) == 0:
            qpos = self.init_qpos + self.np_random.uniform(
                low=-0.005, high=0.005, size=self.model.nq
            )
            qvel = self.init_qvel + self.np_random.uniform(
                low=-0.005, high=0.005, size=self.model.nv
            )
        else:
            qpos = np.array([self.init_qpos[0]] + list(pos[:5]))
            qvel = np.array(pos[5:])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def visualize_rewards(self, path, model):
        pass
    
    def _reward(self, a, ob):
        backroll = -ob[7]
        height = ob[0]
        vel_act = a[0] * ob[8] + a[1] * ob[9] + a[2] * ob[10]
        backslide = -ob[5]
        return backroll * (1.0 + .3 * height + .1 * vel_act + .05 * backslide)

    def get_expert_cost(self, states, actions):
        costs = []
        for i in range(len(states)):
            costs.append(self._reward(actions[i], states[i]))
        return -1 * np.array(costs)


