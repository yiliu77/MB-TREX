import sys

import numpy as np
import torch
import wandb
import yaml

from itertools import combinations
from models.trex import TRexCost
from models.svg import SVG
from models.visualmpc import VisualMPC
from parser import create_env
from models.human import MazeHuman


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    env = create_env(params["env"])
    human = MazeHuman(env)
    query_states, query_actions = human.get_demo(3)

    transition_params = params["transition_model"]
    dir_name = f"./saved/models/{transition_params['type']}/{transition_params['note']}/"
    video_prediction = SVG(transition_params)
    video_prediction.load(dir_name + "saved_svg")

    # TODO: demonstrations using GT
    # TODO: a star to different points
    # TODO: ensemble with different ensemble members

    # TODO: intrinsic motivation using RND
    # TODO: information gain metric for selecting queries instead of random selection / instead of using learned reward add exploration bonus as infogain
    # TODO: down the line skill discovery methods

    cost_model = TRexCost(video_prediction.create_encoding, transition_params["g_dim"], params["cost_model"])

    agent_model = VisualMPC(video_prediction, cost_model.get_value, params["env"]["horizon"], env)
    while True:
        all_states = []
        state = env.reset()
        t = 1
        done = False
        while not done:
            action, generated_traj, generated_actions = agent_model.act(state)
            next_state, reward, done, info = env.step(action)
            print(action, t, reward)

            query_states.append(generated_traj)
            query_actions.append(generated_actions)
            all_states.append(state)
            state = next_state
            t += 1

        all_states = np.array(all_states)
        all_states = np.reshape(all_states, (all_states.shape[0] * all_states.shape[1], all_states.shape[2], all_states.shape[3]))

        gen_images = wandb.Image(all_states, caption="Sequences")
        wandb.log({"Sequences": gen_images})
        for states, actions in zip(query_states, query_actions):
            print(len(states), len(actions))

        query_states = np.transpose(np.array(query_states), (0, 1, 3, 4, 2))
        query_actions = np.array(query_actions)
        indices = list(combinations(range(len(query_states)), 2))
        np.random.shuffle(indices)
        indices = indices[:8]

        paired_states1, paired_states2 = [], []
        for index in indices:
            paired_states1.append(query_states[index[0]])
            paired_states2.append(query_states[index[1]])
        print(np.array(paired_states1).shape)
        labels = human.query_preference(paired_states1, paired_states2)

        paired_states1 = np.array(paired_states1)
        paired_states2 = np.array(paired_states2)
        labels = np.array(labels)
        print(labels.shape, paired_states2.shape, paired_states1.shape)
        cost_model.train(paired_states1, paired_states2, labels) # TODO fix
