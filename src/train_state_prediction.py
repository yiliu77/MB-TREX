import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from parser import create_env
from models.dynamics import PtModel
import tqdm



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
    print("Logging to", logdir)
    env = create_env(params["env"])
    transition_params = params["dynamics_model"]

    if transition_params["use_cached"]:
        observations = np.load(os.path.join(logdir, transition_params["save_obs"] + ".npy"))
        actions = np.load(os.path.join(logdir, transition_params["save_acs"] + ".npy"))
    else:
        obs = env.reset()
        observations = [[obs]]
        actions = [[]]
        print("Collecting dynamics data...")
        for i in tqdm.trange(transition_params["num_dynamics_iters"]):
            action = env.action_space.sample()
            actions[-1].append(action)
            next_obs, _, _, _ = env.step(action)
            obs = next_obs
            if (i + 1) % 200 == 0:
                observations[-1] = np.vstack(observations[-1])
                actions[-1] = np.vstack(actions[-1])
                observations.append([])
                actions.append([])
                obs = env.reset()#difficulty=None, check_constraint=False)
            observations[-1].append(obs)
        observations.pop()
        actions.pop()
        observations = np.array(observations)
        actions = np.array(actions)
        np.save(os.path.join(logdir, transition_params["save_obs"]), observations)
        np.save(os.path.join(logdir, transition_params["save_acs"]), actions)

    print("Beginning training model...")
    dynamics = PtModel(params["env"]["state_dim"], params["env"]["action_dim"], lr=transition_params["lr"]).to(device)
    # val_loss = dynamics.train_dynamics(np.concatenate([observations[:, :7], observations[:, 10:13], observations[:, -1:]], axis=1), actions, transition_params["epochs"],
    #                                    val_split=transition_params["train_test_split"])
    val_loss = dynamics.train_dynamics(observations[:100000], actions[:100000],
                                       transition_params["epochs"],
                                       val_split=transition_params["train_test_split"])

    torch.save(dynamics, os.path.join(logdir, transition_params["model_file"]))

    plt.figure()
    plt.plot(val_loss)
    plt.xlabel("epoch")
    plt.ylabel("Val loss")
    plt.title("Dynamics training")
    plt.savefig(os.path.join(logdir, "{}.png".format(os.path.splitext(transition_params["model_file"])[0])))
    plt.close()
