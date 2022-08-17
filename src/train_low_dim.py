import sys

import numpy as np
import torch
import yaml

from models.trex import TRexCost, GTCost
from models.mpc import MPC
from parser import create_env, create_human
from models.dynamics import PtModel, predict_gt_dynamics
import os
import datetime
import matplotlib.pyplot as plt
import cv2
from models.rnd import RND, RunningMeanStd
from functools import partial
from bc import BCEnsemble
from itertools import combinations, product
from PIL import Image

if __name__ == "__main__":
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
    logdir = os.path.join("saved/models/TREX/", params["env"]["type"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(logdir, exist_ok=True)
    print("Logging to", logdir)

    with open(os.path.join(logdir, "cfg.yaml"), "w") as f:
        yaml.dump(params, f)

    state_dim, action_dim = params["env"]["state_dim"], params["env"]["action_dim"]
    env = create_env(params["env"])
    human = create_human(params["human"], env)
    rnd = RND(state_dim, device=device)

    transition_params = params["dynamics_model"]
    tmp_env = create_env(params["env"])

    if transition_params["gt_dynamics"]:
        dynamics = partial(predict_gt_dynamics, tmp_env, 2)
    else:
        dynamics = torch.load(os.path.join("saved/models/state", params["env"]["type"], "model_sac.pth")).to(device)
        # rnd.update_stats_from_states(torch.from_numpy(transitions[np.random.permutation(np.arange(transitions.shape[0]))[:100]][:, 0]).to(device))

    if params["cost_model"]["gt_cost"]:
        cost_model = GTCost(env)
    else:
        cost_model = TRexCost(human, state_dim, action_dim, params["mpc"]["keep_gen_traj"], params["cost_model"])

    agent_model = MPC(dynamics, cost_model, env, params["mpc"], rnd, dynamics_cuda=not transition_params["gt_dynamics"])
    
    demo_trajs = []
    demo_states = []
    demo_acs = []

    for i in range(params["cost_model"]["offline_demos"]):
        demo = human.get_demo()
        demo_states.extend(demo["obs"][:-1])
        demo_acs.extend(demo["acs"])
        demo_trajs.append([demo["obs"][:-1], demo["acs"], demo["obs"][1:]])

    if params["cost_model"]["drex_pretrain"]:
        states_np = np.vstack(demo_states)
        actions_np = np.vstack(demo_acs)
        shuffle_idx = np.random.permutation(len(states_np))
        states = torch.from_numpy(states_np[shuffle_idx]).to(device).float()
        actions = torch.from_numpy(actions_np[shuffle_idx]).to(device).float()

        bc = BCEnsemble(device)
        epochs = 10000
        val_losses, loss_mean_pred = bc.train(states, actions, epochs)

        n_trajs = 10
        states1, acs1, states2, acs2, bc_labels = [], [], [], [], []
        trajectories = [[] for _ in range(params["cost_model"]["offline_demos"]))]
        costs = [[] for _ in range(len(demo_trajs))]
        bc_trajs = []
        # Collect BC rollouts from the starting state of each demonstration at every noice level
        for traj_i in range(len(demo_trajs)):
            start = demo_trajs[traj_i][0][0]
            for i, e, in enumerate(eps):
                trajectories[traj_i].append([])
                costs[traj_i].append([])
                for n in range(n_trajs):
                    states = []
                    acs = []
                    s = start
                    traj = []
                    for t in range(env.horizon):
                        states.append(s) 
                        state_torch = torch.from_numpy(s).float().to(device)
                        if np.random.random() < e:
                            a = torch.from_numpy(env.action_space.sample()).float().to(device)
                        else:
                            a = bc.predict(state_torch).squeeze()
                        acs.append(a.detach().cpu().squeeze().numpy())
                        next_s = dynamics(torch.unsqueeze(torch.cat((state_torch, a)), dim=0))
                        s = next_s.detach().cpu().squeeze().numpy()
                    traj = [np.array(states), np.array(acs)]
                    trajectories[traj_i][i].append(traj)
                    costs[traj_i][i].append(env.get_expert_cost([traj[0][-1]])[0])
            for traj_set_1, traj_set_2 in combinations([demo_trajs[traj_i]] + trajectories[traj_i], 2):
                for traj1, traj2 in product(traj_set_1, traj_set_2):
                    states1.append(np.array(traj1)[0, :])
                    acs1.append(np.array(traj1)[1, :])
                    states2.append(np.array(traj2)[0, :])
                    acs2.append(np.array(traj2)[0, :])
                    bc_labels.append(0)
            costs = np.array(costs)           
        cost_model.train(np.concatenate([states1, acs1], axis=1), np.concatenate([states2, acs2], axis=1), bc_labels, params["cost_model"]["pretrain_epochs"])
        
        env.visualize_rewards(os.path.join(logdir, "cost_init.png"), cost_model)
        paired_states1, paired_actions1, paired_states2, paired_actions2, labels = [], [], [], [], []
    elif params["cost_model"]["offline_demos"] > 0:
        paired_states1, paired_actions1, paired_states2, paired_actions2, labels = [demo_trajs[0][0]], [demo_trajs[0][1]], [demo_trajs[1][0]], [demo_trajs[1][1]], [int(np.sum(env.get_expert_cost(demo_trajs[0][0])) > np.sum(env.get_expert_cost(demo_trajs[1][0])))]
        cost_model.train(np.concatenate([paired_states1, paired_actions1], axis=1), np.concatenate([paired_states2, paired_actions2], axis=1), labels, params["cost_model"]["pretrain_epochs"])
        env.visualize_rewards(os.path.join(logdir, "cost_init.png"), cost_model)
    else:
        paired_states1, paired_actions1, paired_states2, paired_actions2, labels = [], [], [], [], []
    
    eval_success_rate = []
    num_success = 0
    plot = True #params["env"]["type"] == "maze" or params["env"]["type"] == "hopper"
    true_costs = []
    for iteration in range(1, params["cost_model"]["episodes"] + 1):
        all_states = []
        all_actions = []
        state = env.reset()
        if plot:
            frames = [env._get_obs(use_images=True)]
        t = 0
        done = False
        cem_actions = []
        cem_states = []
        episode_reward = 0
        learned_cost = 0
        while not done:
            action, generated_actions, generated_states = agent_model.act(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            learned_cost += cost_model.get_value(state.reshape(1, 1, -1), action.reshape(1,1, -1)).cpu().numpy().item()
            cem_actions.append(generated_actions)
            cem_states.append(generated_states)
            all_states.append(state)
            all_actions.append(action.detach().cpu().numpy())

            state = next_state
            if plot:
                frames.append(env._get_obs(use_images=True))
            t += 1
        true_costs.append(-episode_reward)
        if "success" in info:
            num_success += int(info["success"])
        elif "task_success" in info:
            num_success += int(info["task_success"])
        print("ITER:", iteration, "COST:", -episode_reward, "LEARNED COST:", learned_cost, "Success", num_success)
        time_step = np.random.choice(np.arange(t))
        query_actions = cem_actions[time_step]
        query_states = cem_states[time_step]
        if plot and iteration % 5 == 0:
            np.save(os.path.join(logdir, f"ep{iteration}"), np.array(frames))
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(os.path.join(logdir, f"ep{iteration}.gif"), save_all=True, append_images=imgs[1:], duration=67, loop=0)
            # tmp_env.reset(pos=all_states[time_step])
            # gt_frames = [tmp_env._get_obs(use_images=True)]
            # pt_frames = [tmp_env._get_obs(use_images=True)]
            # for action in query_actions[0]:
            #     tmp_env.step(action)
            #     gt_frames.append(tmp_env._get_obs(use_images=True))
            # for pt_state in query_states[0]:
            #     tmp_env.reset(pos=pt_state)
            #     pt_frames.append(tmp_env._get_obs(use_images=True))
            # gt = [Image.fromarray(f) for f in gt_frames]
            # gt[0].save(os.path.join(logdir, f"ep{iteration}_step{time_step}_gt.gif"), save_all=True, append_images=gt[1:], duration=67,
            #              loop=0)
            # pt = [Image.fromarray(f) for f in pt_frames]
            # pt[0].save(os.path.join(logdir, f"ep{iteration}_step{time_step}_pt.gif"), save_all=True,
            #            append_images=pt[1:], duration=67,
            #            loop=0)

        new_paired_trajs1, new_paired_trajs2, new_labels = cost_model.gen_queries(np.array(query_states), np.array(query_actions))
        cost_model.train(new_paired_trajs1, new_paired_trajs2, new_labels, 1)

        #dynamics update
        if iteration % transition_params["online_update_freq"] == 0:
            data = np.array([all_states[:-1], all_actions[:-1], all_states[1:]]).transpose(1, 0, 2)
            dynamics.train_dynamics(data, 1, update_stats=False)
        # rnd update
        rnd.train(torch.tensor(np.array(all_states)).to(device).float())
        if iteration % 10 == 0:
            eval_successes = 0
            rnd_w = agent_model.rnd_weight
            agent_model.rnd_weight = 0
            true_eval_costs = []
            learned_eval_costs = []
            for i in range(10):
                done = False
                state = env.reset()
                true_cost = 0
                learned_cost = 0
                while not done:
                    action, _, _ = agent_model.act(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    true_cost -= reward
                    learned_cost += cost_model.get_value(state.reshape(1, 1, -1), action.reshape(1,1, -1)).cpu().numpy().item()
                if "success" in info and info["success"]:
                    eval_successes += 1
                true_eval_costs.append(true_cost)
                learned_eval_costs.append(learned_cost)
            plt.figure()
            plt.scatter(true_eval_costs, learned_eval_costs)
            plt.xlabel("True Cost")
            plt.ylabel("Learned Cost")
            plt.savefig(os.path.join(logdir, f"ep{iteration}_eval_cost.png"))
            plt.close()
            eval_success_rate.append(eval_successes / 50)
            agent_model.rnd_weight = rnd_w
    # np.save(logdir, "eval_success_rate")/
    np.save(os.path.join(logdir, "true_cost"), true_costs)
    plt.figure()
    plt.plot(true_costs)
    plt.xlabel("Episode")
    plt.ylabel("True Cost")
    plt.savefig(os.path.join(logdir, f"true_cost.png"))
    for i, model in enumerate(cost_model.cost_models):
        torch.save(model, os.path.join(logdir, f"cost_network_{i}.pt"))

