import numpy as np


class Human:
    def query_preference(self, traj1, traj2):
        raise NotImplementedError


class SimplePointHuman(Human):
    def __init__(self, env):
        self.env = env
        self.goal = self.calc_coords(np.transpose(self.env.get_goal(), (2, 0, 1)) / 255)

    def query_preference(self, paired_states1, paired_states2):
        # Trajectories are L x C x H x W
        score1, score2 = 0, 0
        for j in range(paired_states1.shape[0]):
            frame_x, frame_y = self.calc_coords(paired_states1[j])
            score1 += -((frame_x - self.goal[0]) ** 2 + (frame_y - self.goal[1]) ** 2)

            frame_x, frame_y = self.calc_coords(paired_states2[j])
            score2 += -((frame_x - self.goal[0]) ** 2 + (frame_y - self.goal[1]) ** 2)
        label = 0 if score1 > score2 else 1
        return label

    @staticmethod
    def calc_coords(frame):
        state = np.logical_and(frame[0, :, :] > 0.5, frame[1, :, :] < 0.5)
        x, y = np.argwhere(state).mean(0)
        return x, y
