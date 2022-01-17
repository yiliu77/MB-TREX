import sys

import numpy as np
import torch
import wandb
import yaml

from models.svg import SVG
from models.visualmpc import VisualMPC
from parser import create_env

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    model_params = params["model"]
    dir_name = f"./saved/models/{model_params['type']}/{params['note']}/"
    video_prediction = SVG(model_params)
    video_prediction.load(dir_name + "saved_svg")

    env = create_env(params["env"])

    goal_state = env.reset(pos=env.goal)
    goal_state = torch.permute(torch.as_tensor(np.array(goal_state)), (2, 0, 1)).unsqueeze(0)
    goal_state = goal_state.to(device).float() / 255

    def cost_fn(trajectory):
        costs = np.zeros((trajectory.shape[0], trajectory.shape[1]))
        for i in range(trajectory.shape[0]):
            for j in range(trajectory.shape[1]):
                state = trajectory[i, j, :, :, :].permute(1, 2, 0)

                state = state[:, :, 0] > 0.7
                state = state.cpu().numpy()

                goal = goal_state.permute(0, 2, 3, 1)[0, :, :, 0] > 0.7
                goal = goal.cpu().numpy()

                x, y = np.argwhere(state).mean(0)
                goalx, goaly = np.argwhere(goal).mean(0)

                costs[i, j] = (goalx - x) ** 2 + (goaly - y) ** 2
        return torch.as_tensor(costs).to("cuda")

        # return torch.sum(torch.square(trajectory_encoding - video_prediction.encoder(goal_state, None).unsqueeze(0)), dim=2)

    model = VisualMPC(video_prediction, cost_fn, params["env"]["horizon"], env)
    while True:
        all_states = []
        state = env.reset()
        t = 1
        done = False
        while not done:
            action = model.act(state)
            # from matplotlib import pyplot as plt
            #
            # plt.imshow(state)
            # plt.show()
            next_state, reward, done, info = env.step(action)
            print(action, t, reward)

            all_states.append(state)
            state = next_state
            t += 1

        all_states = np.array(all_states)
        print(np.array(all_states).shape)
        all_states = np.reshape(all_states, (all_states.shape[0] * all_states.shape[1], all_states.shape[2], all_states.shape[3]))

        gen_images = wandb.Image(all_states, caption="Sequences")
        wandb.log({"Sequences": gen_images})
