import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from parser import create_env
from models.dynamics import PtModel



if __name__ == '__main__':
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    else:
        seed = 123456
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)
    name = params["env"]["type"]
    logdir = os.path.join("saved/models/state/", name)
    os.makedirs(logdir, exist_ok=True)

    env = create_env(params["env"])
    transition_params = params["dynamics_model"]

    if transition_params["use_cached"]:
        observations = np.load(os.path.join("saved/data/state/", name, "observations.npy"))
        actions = np.load(os.path.join("saved/data/state/", name, "actions.npy"))
    else:
        obs = env.reset()
        observations = [obs]
        actions = []
        print("Collecting dynamics data...")
        for i in range(transition_params["num_dynamics_iters"]):
            action = env.action_space.sample()
            actions.append(action)
            next_obs, _, _, _ = env.step(action)
            obs = next_obs
            if i % 20 == 0:
                obs = env.reset(difficulty=None, check_constraint=False)
            observations.append(obs)
        observations = np.array(observations)
        actions = np.array(actions)

    print("Beginning training model...")
    dynamics = PtModel(params["env"]["state_dim"], params["env"]["action_dim"], lr=transition_params["lr"]).to(device)
    val_loss = dynamics.train_dynamics(observations, actions, transition_params["epochs"],
                                       val_split=transition_params["train_test_split"])

    plt.figure()
    plt.plot(val_loss)
    plt.xlabel("epoch")
    plt.ylabel("Val loss")
    plt.title("Dynamics training")
    plt.savefig(os.path.join(logdir, "dynamics.png"))
    plt.close()
