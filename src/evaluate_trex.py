import sys

import torch
import yaml
import wandb
import numpy as np

from models.svg import SVG
from models.trex import TREX
from models.visualmpc import VisualMPC
from utils.parser import create_env, create_human

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    env = create_env(params["env"])
    human = create_human(params["human"], env)

    cost_model = TREX(human, env.observation_space, env.action_space.shape[0], params["cost_model"])
    cost_model.load("saved/{}/TREX/save.pt".format(params["env"]["type"]))
    env.visualize_rewards("saved/{}/TREX/evaluation.png".format(params["env"]["type"]), cost_model)

    video_model = SVG(env.observation_space, env.action_space.shape[0], params["video_model"])
    video_dir_name = f"./saved/{params['env']['type']}/{video_model.type}/"
    video_model.load(video_dir_name + "svg.pt")

    visualmpc = VisualMPC(video_model, cost_model.get_value, params["env"]["horizon"], env, params["visualmpc"])

    while True:
        all_states = []
        state = env.reset()
        t = 0
        done = False
        while not done:
            action = visualmpc.act(state, evaluate=True)[0]
            next_state, reward, done, info = env.step(action)
            print(action, t, reward)

            all_states.append(state)
            state = next_state
            t += 1

        avg_states = wandb.Image(np.mean(np.stack(all_states, axis=0), axis=0), caption="Sequences")
        max_states = wandb.Image(np.max(np.stack(all_states, axis=0), axis=0), caption="Sequences")
        wandb.log({"Mean Trajectories": avg_states, "Max Trajectories": max_states})