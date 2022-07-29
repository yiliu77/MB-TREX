import cv2

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


class Hopper(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, params):
        self._max_episode_steps = 60
        self.num_steps = 0
        self.dims = params["dims"]
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
        return self._get_obs(), reward, self._max_episode_steps == self.num_steps, {"lowdim": np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])}

    def _get_obs(self):
        rendered_img = self.render(mode="rgb_array")
        rendered_img = rendered_img[50: -50, 50: -50, :]
        rendered_img = cv2.resize(rendered_img, tuple(self.dims))
        rendered_img = (0.299 * rendered_img[..., 0] + 0.587 * rendered_img[..., 1] + 0.114 * rendered_img[..., 2])[:, :, None]
        return rendered_img

    def reset(self):
        self.num_steps = 0
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs(), {"lowdim": np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])}

    def visualize_rewards(self, path, model):
        pass
