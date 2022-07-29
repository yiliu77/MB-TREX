import numpy as np
from human.simple_point_human import Human
import wandb


class MazeHuman(Human):
    def __init__(self, env):
        self.env = env
        self.initial_state = self.env.get_empty() / 256
        self.mid_point = self.calc_coords(np.transpose(self.env.reset(pos=(0.12, 0.12))[0], (2, 0, 1)) / 255)
        self.goal = self.calc_coords(np.transpose(self.env.get_goal(), (2, 0, 1)) / 255)
        print(self.goal, self.mid_point)

    def query_preference(self, paired_states1, paired_states2, validate=False):
        # Trajectories are L x C x W x H
        score1, score2 = 0, 0
        for j in range(paired_states1.shape[0]):
            frame_x, frame_y = self.calc_coords(paired_states1[j])
            if frame_y < self.mid_point[1]:
                score1 += -((frame_x - self.mid_point[0]) ** 2 + (frame_y - self.mid_point[1]) ** 2)
            else:
                score1 += 1e5 - ((frame_x - self.goal[0]) ** 2 + (frame_y - self.goal[1]) ** 2)

            frame_x, frame_y = self.calc_coords(paired_states2[j])
            if frame_y < self.mid_point[1]:
                score2 += -((frame_x - self.mid_point[0]) ** 2 + (frame_y - self.mid_point[1]) ** 2)
            else:
                score2 += 1e5 - ((frame_x - self.goal[0]) ** 2 + (frame_y - self.goal[1]) ** 2)

        label = 0 if score1 > score2 else 1
        if not validate:
            comb_seqs = np.empty((paired_states1.shape[2], paired_states1.shape[3] * 2, paired_states1.shape[1]), dtype=paired_states1.dtype)
            comb_seqs[:, :paired_states1.shape[3], ...] = np.transpose(np.max(paired_states1, axis=0), (1, 2, 0))
            comb_seqs[:, paired_states1.shape[3]:, ...] = np.transpose(np.max(paired_states2, axis=0), (1, 2, 0))
            wandb.log({"Queries": wandb.Image(comb_seqs, caption="Query Score {}, {}".format(score1 / paired_states1.shape[0], score2 / paired_states1.shape[0]))})
        return label

    def calc_coords(self, frame):
        from matplotlib import pyplot as plt
        diff = np.transpose(frame, (1, 2, 0)) - self.initial_state
        plt.imshow(diff)
        plt.show()
        x, y = np.argwhere(diff[:, :, 0] > 0.15).mean(0)
        if np.isnan(x):
            return 0, 0
        return x, y
