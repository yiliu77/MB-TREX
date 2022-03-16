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
    demo_states, demo_actions = human.get_demo(2)
    query_states, query_actions = [], []
    for demo_state, demo_action in zip(demo_states, demo_actions):
        query_states.append(demo_state[0:8])
        query_actions.append(demo_action[0:8])

    for _ in range(10):
        obs = env.reset()
        action = env.action_space.sample()
        random_states, random_actions = [], []
        for _ in range(8):
            next_obs, _, _, _ = env.step(action)
            obs = next_obs
            random_states.append(np.transpose(obs, (2, 0, 1)) / 255)
            random_actions.append(action)
        query_states.append(np.array(random_states))
        query_actions.append(np.array(random_actions))

    transition_params = params["video_model"]
    dir_name = f"./saved/models/{transition_params['type']}/{transition_params['note']}/"
    video_prediction = SVG(transition_params)
    video_prediction.load(dir_name + "saved_svg")
    video_prediction.eval()

    # TODO: intrinsic motivation using RND
    # TODO: down the line skill discovery methods

    # TODO: change from mean of ensemble to individual ensemble to improve diversity of CEM trajectories
    # TODO: add more metrics: measure variance of trex ensemble members, measure GT reward over time
    # TODO: metric for w/o RND, w/o ensemble visualmpc (individual vs ensemble)
    # TODO: visualize each cem step
    # TODO show ground truth env vs visual mpc trajectory
    # TODO: cem with ground truth
    # TODO: rerun CEM with 15 frames ahead instead of 8

    cost_model = TRexCost(video_prediction.create_encoding, transition_params["g_dim"], params["cost_model"])
    env.visualize_rewards("saved/models/TREX/initial.png", cost_model)

    agent_model = VisualMPC(video_prediction, cost_model.get_value, params["env"]["horizon"], env)
    for iteration in range(1000):
        all_states = []
        state = env.reset()
        t = 1
        total_reward = 0
        done = False
        while not done:
            action, generated_traj, generated_actions = agent_model.act(state)
            next_state, reward, done, info = env.step(action)
            print(action, t, reward)
            total_reward += reward

            # from matplotlib import pyplot as plt
            # plt.imshow(state)
            # plt.show()

            if t % 4 == 0:
                query_states.append(generated_traj)
                query_actions.append(generated_actions)
            all_states.append(state)
            state = next_state
            t += 1
        wandb.log({"Total Reward": total_reward})

        all_states = np.array(all_states)
        all_states = np.reshape(all_states, (all_states.shape[0] * all_states.shape[1], all_states.shape[2], all_states.shape[3]))

        gen_images = wandb.Image(all_states, caption="Sequences")
        wandb.log({"Sequences": gen_images})
        for states, actions in zip(query_states, query_actions):
            print(states.shape, actions.shape)

        query_states_np = np.transpose(np.array(query_states), (0, 1, 3, 4, 2))
        query_actions_np = np.array(query_actions)
        indices = list(combinations(range(len(query_states_np)), 2))
        np.random.shuffle(indices)
        indices = indices[:10]

        paired_states1, paired_actions1, paired_states2, paired_actions2 = [], [], [], []
        for index in indices:
            paired_states1.append(query_states_np[index[0]])
            paired_states2.append(query_states_np[index[1]])
            paired_actions1.append(query_actions_np[index[0]])
            paired_actions2.append(query_actions_np[index[1]])
        labels = human.query_preference(paired_states1, paired_states2)

        paired_states1 = np.array(paired_states1)
        paired_states2 = np.array(paired_states2)
        labels = np.array(labels)
        cost_model.train(paired_states1, paired_actions1, paired_states2, paired_actions2, labels, 10)  # TODO fix
        env.visualize_rewards("saved/models/TREX/image{}.png".format(iteration), cost_model)
