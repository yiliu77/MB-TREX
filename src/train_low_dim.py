import sys

import numpy as np
import torch
import yaml

from itertools import combinations
from models.trex import TRexCost
from models.mpc import MPC
from parser import create_env
from models.human import LowDimHuman
from models.dynamics import PtModel, predict_gt_dynamics
import os
import datetime
import matplotlib.pyplot as plt
import cv2
from models.rnd import RND, RunningMeanStd
from functools import partial
from bc import BCEnsemble
from itertools import combinations, product


def plot_cost_function(cost_model, num_pts=100, **kwargs):
    x_bounds = [-0.3, 0.3]
    y_bounds = [-0.3, 0.3]

    states = []
    x_pts = num_pts
    y_pts = int(
        x_pts * (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0]))
    for x in np.linspace(x_bounds[0], x_bounds[1], y_pts):
        for y in np.linspace(y_bounds[0], y_bounds[1], x_pts):
            states.append([x, y])

    return cost_model(states, **kwargs).reshape(y_pts, x_pts).T

def dynamics_error(states, env=None, dynamics=None):
    angles = np.arange(0, 2 * np.pi, np.pi / 3)
    acs = np.stack([np.cos(angles), np.sin(angles)]).T
    state_acs = np.concatenate((np.repeat(states, len(acs), axis=0), np.tile(acs, (len(states), 1))), axis=1)
    predictions = dynamics(torch.from_numpy(state_acs).to(device).float()).detach().cpu().numpy()
    actual = []
    for sa in state_acs:
        env.reset(pos=sa[:2])
        actual.append(env.step(sa[2:])[0])
    actual = np.array(actual)
    return np.linalg.norm(predictions - actual, axis=1).reshape(-1, len(acs)).T.mean(axis=0)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    else:
        seed = 123456
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)
    logdir = os.path.join("saved/models/TREX/maze", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, "cfg.yaml"), "w") as f:
        yaml.dump(params, f)
    
    env = create_env(params["env"])
    human = LowDimHuman(env, 0.005)
    rnd = RND(2, device=device)

    transition_params = params["dynamics_model"]
    tmp_env = create_env(params["env"])

    if transition_params["gt_dynamics"]:
        dynamics = partial(predict_gt_dynamics, tmp_env, 2)
    else:
        obs = env.reset()
        transitions = []
        print("Collecting dynamics data...")
        for i in range(transition_params["num_dynamics_iters"]):
            action = env.action_space.sample()
            next_obs, _, _, _ = env.step(action)
            transitions.append([obs, action, next_obs])
            obs = next_obs
            if i % 20 == 0:
                obs = env.reset(difficulty=None, check_constraint=False)
        transitions = np.array(transitions, dtype='float32')
        

        print("Begin training dynamics...")
        # dir_name = f"./saved/models/{transition_params['type']}/{transition_params['note']}/"
        dynamics = PtModel(2, 2, lr=transition_params["lr"]).to(device)
        
        val_loss = dynamics.train_dynamics(transitions, transition_params["epochs"], val_split=transition_params["train_test_split"])
        plt.figure()
        plt.plot(val_loss)
        plt.xlabel("epoch")
        plt.ylabel("Val loss")
        plt.title("Dynamics training")
        plt.savefig(os.path.join(logdir, "dynamics.png"))
        plt.close()

        dynamics_viz = plot_cost_function(dynamics_error, env=tmp_env, dynamics=dynamics)

        plt.figure()
        # plt.imshow(background)
        plt.imshow(dynamics_viz, alpha=0.6)
        plt.colorbar()
        plt.savefig(os.path.join(logdir, "dynamics_error.png"))
        plt.close()

        rnd.update_stats_from_states(torch.from_numpy(transitions[np.random.permutation(np.arange(transitions.shape[0]))[:100]][:, 0]).to(device))

    cost_model = TRexCost(lambda x, y: x, 2, params["cost_model"], num_nets=1)
    env.visualize_rewards(os.path.join(logdir, "initial.png"), cost_model)

    agent_model = MPC(dynamics, cost_model, env, params["mpc"], rnd, dynamics_cuda=not transition_params["gt_dynamics"], logdir=logdir)
    
    demo_trajs = []
    demo_states = []
    demo_acs = []

    for i in range(params["cost_model"]["offline_demos"]):
        demo = human.get_demo()
        demo_states.extend(demo["obs"][:-1])
        demo_acs.extend(demo["acs"])
        demo_trajs.append([demo["obs"][:-1], demo["acs"], demo["obs"][1:]])

    if params["cost_model"]["bc_warm_start"]:
        states_np = np.vstack(demo_states)
        actions_np = np.vstack(demo_acs)
        shuffle_idx = np.random.permutation(len(states_np))
        states = torch.from_numpy(states_np[shuffle_idx]).to(device).float()
        actions = torch.from_numpy(actions_np[shuffle_idx]).to(device).float()

        bc = BCEnsemble(device)
        epochs = 10000
        val_losses, loss_mean_pred = bc.train(states, actions, epochs)

        plt.plot(np.arange(1, len(loss_mean_pred) + 1), loss_mean_pred, label="ensemble prediction validation loss")
        plt.plot(np.arange(1, len(loss_mean_pred)  + 1), np.array(val_losses).mean(axis=0), label="mean validation loss")
        plt.legend()    
        plt.savefig(os.path.join(logdir, "train_bc.png"))
        plt.close()

        bc_trajs = []
        eps = np.linspace(0, 0.75, num=4)

        if params["cost_model"]["bc_comp_mode"] == "all":
            start = env.reset()
            costs = [[] for _ in range(len(eps))]
            for i, e in enumerate(eps):
                bc_trajs.append([])
                for k in range(5):
                    states, acs = [], []
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
                    bc_trajs[-1].append(traj)
                    costs[i].append(env.get_expert_cost([traj[0][-1]])[0])

            costs = np.array(costs)
            avg_costs = np.mean(costs, axis=1)
            std_costs = np.std(costs, axis=1)
            plt.figure()
            plt.errorbar(eps, avg_costs, yerr=std_costs)
            plt.xlabel("epsilon greedy")
            plt.ylabel("cost at final state")
            plt.savefig(os.path.join(logdir, "degredation.png"))
            plt.close()

            background = cv2.resize(env._get_obs(images=True), (100, 100))
            plt.figure()
            plt.imshow(background, extent=[0, 100, 100, 0])
            plt.plot(demo_trajs[0][0][:, 0] * 100 / 0.6 + 50, demo_trajs[0][0][:, 1]  * 100 / 0.6 + 50, label="Demo")
            for i, traj_level in enumerate(bc_trajs):
                plt.plot(traj_level[0][0][:, 0] * 100 / 0.6 + 50, traj_level[0][0][:, 1]  * 100 / 0.6 + 50, label="epsilon = " + str(eps[i]))
            plt.legend()
            plt.savefig(os.path.join(logdir, "demos.png"))
            plt.close()
            states1 = []
            acs1 = []
            states2 = []
            acs2 = []
            bc_labels = []
            
            for traj_set_1, traj_set_2 in combinations([demo_trajs] + bc_trajs, 2):
                for traj1, traj2 in product(traj_set_1, traj_set_2):
                    states1.append(np.array(traj1)[0, :])
                    acs1.append(np.array(traj1)[1, :])
                    states2.append(np.array(traj2)[0, :])
                    acs2.append(np.array(traj2)[1, :])
                    bc_labels.append(0)
        elif params["cost_model"]["bc_comp_mode"] == "same" or params["cost_model"]["bc_comp_mode"] == "other":
            n_trajs = 10
            states1, acs1, states2, acs2, bc_labels = [], [], [], [], []
            trajectories = [[] for _ in range(len(demo_trajs))]
            costs = [[] for _ in range(len(demo_trajs))]
            bc_trajs = []
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
            if params["cost_model"]["bc_comp_mode"] != "other":
                costs = np.array(costs)
                avg_costs = np.mean(costs, axis=-1)
                std_costs = np.std(costs, axis=-1)
                plt.figure()
                for i, cost_from_start in enumerate(avg_costs):
                    plt.errorbar(eps, cost_from_start, yerr=std_costs[i])
                plt.xlabel("epsilon greedy")
                plt.ylabel("cost at final state")
                plt.savefig(os.path.join(logdir, "degredation.png"))
                plt.close()

                
                background = cv2.resize(env._get_obs(images=True), (100, 100))
                plt.figure()
                plt.imshow(background, extent=[0, 100, 100, 0])
                plt.plot(demo_trajs[0][0][:, 0] * 100 / 0.6 + 50, demo_trajs[0][0][:, 1]  * 100 / 0.6 + 50, label="Demo")
                for i, traj_level in enumerate(trajectories[0]):
                    plt.plot(traj_level[0][0][:, 0] * 100 / 0.6 + 50, traj_level[0][0][:, 1]  * 100 / 0.6 + 50, label="epsilon = " + str(eps[i]))
                plt.legend()
                plt.savefig(os.path.join(logdir, "demos.png"))
                plt.close()
        if params["cost_model"]["bc_comp_mode"] == "other":
            old_costs = costs
            costs = costs = [[] for _ in range(len(demo_trajs))]
            for traj_i in range(len(demo_trajs)):
                start = env.reset()
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
                for traj_set_1, traj_set_2 in combinations(trajectories[traj_i], 2):
                    for traj1, traj2 in product(traj_set_1, traj_set_2):
                        states1.append(np.array(traj1)[0, :])
                        acs1.append(np.array(traj1)[1, :])
                        states2.append(np.array(traj2)[0, :])
                        acs2.append(np.array(traj2)[0, :])
                        bc_labels.append(0)
                    
        cost_model.train(states1, acs1, states2, acs2, bc_labels, params["cost_model"]["pretrain_epochs"])
        
        env.visualize_rewards(os.path.join(logdir, "cost_init.png"), cost_model)
        paired_states1, paired_actions1, paired_states2, paired_actions2, labels = [], [], [], [], []
    else: 
        paired_states1, paired_actions1, paired_states2, paired_actions2, labels = [demo_trajs[0][0]], [demo_trajs[0][1]], [demo_trajs[1][0]], [demo_trajs[1][1]], [int(np.sum(env.get_expert_cost(demo_trajs[0][0])) > np.sum(env.get_expert_cost(demo_trajs[1][0])))]
        cost_model.train(paired_states1, paired_actions1, paired_states2, paired_actions2, labels, params["cost_model"]["pretrain_epochs"])
        env.visualize_rewards(os.path.join(logdir, "cost_init.png"), cost_model)

        background = cv2.resize(env._get_obs(images=True), (100, 100))
        plt.figure()
        plt.imshow(background, extent=[0, 100, 100, 0])
        for demo in demo_trajs:
            plt.plot(demo[0][:, 0] * 100 / 0.6 + 50, demo[0][:, 1]  * 100 / 0.6 + 50)
        plt.savefig(os.path.join(logdir, "demos.png"))
    
    eval_success_rate = []
    num_success = 0
    for iteration in range(params["cost_model"]["episodes"]):
        all_states = []
        all_actions = []
        state = env.reset()
        t = 0
        done = False
        plot_cem = "CEM trajectories ep " + str(iteration) if iteration % 2 == 0 else False
        cem_actions = []
        cem_states = []
        while not done:
            action, generated_actions, generated_states = agent_model.act(state, plot= (t==10) and plot_cem)
            next_state, reward, done, info = env.step(action)
            cem_actions.append(generated_actions)
            cem_states.append(generated_states)
            all_states.append(state)
            all_actions.append(action.detach().cpu().numpy())

            state = next_state
            t += 1

        if info['success']:
            num_success += 1
        print("ITER:", iteration, "NUM SUCCESSES:", num_success)
        time_step = np.random.choice(np.arange(t))
        query_actions = cem_actions[time_step]
        query_states = cem_states[time_step]


        indices = list(combinations(range(len(query_states)), 2))
        np.random.shuffle(indices)
        new_paired_states1, new_paired_actions1, new_paired_states2, new_paired_actions2, new_labels = [], [], [], [], []
        plot_queries = iteration % 5 == 0
        background = cv2.resize(env._get_obs(images=True), (100, 100))
        
        if plot_queries:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 16))
            axs[0].imshow(background, extent=[0, 100, 100, 0])
            axs[0].plot(np.array(query_states).transpose(1, 0, 2)[:, :, 0] * 100 / 0.6 + 50, np.array(query_states).transpose(1, 0, 2)[:, :, 1]  * 100 / 0.6 + 50)
            axs[1].imshow(background, extent=[0, 100, 100, 0])
            acs = torch.stack(query_actions)
            query_states_gt = torch.tensor(all_states[time_step]).float().repeat(1, acs.shape[0], 1)
            for ac in acs.permute(1, 0, 2):
                obs = query_states_gt[-1].clone().float()
                next_obs = predict_gt_dynamics(tmp_env, 2, torch.cat((obs, ac), dim=1))
                query_states_gt = torch.cat((query_states_gt, torch.unsqueeze(next_obs, dim=0)), dim=0)
            axs[1].plot(query_states_gt[:, :, 0] * 100 / 0.6 + 50, query_states_gt[:, :, 1] * 100 / 0.6 + 50)
            plt.savefig(os.path.join(logdir, "Queried Trajs ep " + str(iteration)))
            plt.close()

        if plot_queries:
            fig, axs = plt.subplots(nrows=1, ncols=len(indices), figsize=(4 * len(indices), 4))
            fig.suptitle("Trajectory pairs at step " + str(time_step) + " of episode " + str(iteration))
            expert_cost = plot_cost_function(env.get_expert_cost)


        for i, index in enumerate(indices):
            new_paired_states1.append(query_states[index[0]])
            new_paired_states2.append(query_states[index[1]])
            new_paired_actions1.append(query_actions[index[0]])
            new_paired_actions2.append(query_actions[index[1]])
            label = human.query_preference(query_states[index[0]], query_states[index[1]])
            new_labels.append(label)
            if plot_queries:
                axs[i].imshow(background, extent=[0, 100, 100, 0])
                axs[i].imshow(expert_cost, alpha=0.6)

                traj1_x = query_states[index[0]][:, 0].T * 100 / 0.6 + 50
                traj1_y = query_states[index[0]][:, 1].T * 100 / 0.6 + 50
                traj2_x = query_states[index[1]][:, 0].T * 100 / 0.6 + 50
                traj2_y = query_states[index[1]][:, 1].T * 100 / 0.6 + 50
                if label == 0.5:
                    axs[i].plot(traj1_x, traj1_y, color="b")
                    axs[i].plot(traj2_x, traj2_y, color="b")
                else:
                    axs[i].plot(traj1_x, traj1_y, color="r" if label else "g")
                    axs[i].plot(traj2_x, traj2_y, color="g" if label else "r")
        if plot_queries:
            plt.savefig(os.path.join(logdir, "TREX_pairs_ep" + str(iteration) + ".png"), bbox_inches='tight')
            plt.close()

        paired_states1 += new_paired_states1
        paired_actions1 += new_paired_actions1
        paired_states2 += new_paired_states2
        paired_actions2 += new_paired_actions2
        labels += new_labels

        cost_model.train(paired_states1, paired_actions1, paired_states2, paired_actions2, labels, 1)
        if iteration % 5 == 0:
            env.visualize_rewards(os.path.join(logdir, "cost_ep" + str(iteration) + ".png"), cost_model)

        #dynamics update
        # if iteration % transition_params["online_update_freq"] == 0:
        #     data = np.array([all_states[:-1], all_actions[:-1], all_states[1:]]).transpose(1, 0, 2)
        #     dynamics.train_dynamics(data, 1, update_stats=False)
        # rnd update
        rnd.train(torch.tensor(all_states).to(device).float())
        if iteration % 10 == 0:
            rnd_cost = plot_cost_function(lambda x: rnd.get_value(torch.tensor(x).to(device)).cpu().numpy())
            plt.figure()
            plt.imshow(background, extent=[0, 100, 100, 0])
            plt.imshow(rnd_cost, alpha=0.6)
            plt.colorbar()
            plt.savefig(os.path.join(logdir, "rnd_cost" + str(iteration) + ".png"))
            plt.close()

        if iteration % 10 == 0:
            eval_successes = 0
            rnd_w = agent_model.rnd_weight
            agent_model.rnd_weight = 0
            for i in range(50):
                done = False
                state = env.reset()
                while not done:
                    action, _, _ = agent_model.act(state, plot=False)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                if info["success"]:
                    eval_successes += 1
            eval_success_rate.append(eval_successes / 50)
            agent_model.rnd_weight = rnd_w
    torch.save(cost_model.cost_model, os.path.join(logdir, "cost_network.pt"))
    env.visualize_rewards(os.path.join(logdir, "cost_final.png"), cost_model)

    eval_successes = 0
    rnd_w = agent_model.rnd_weight
    agent_model.rnd_weight = 0
    for i in range(50):
        done = False
        state = env.reset()
        while not done:
            action, _, _ = agent_model.act(state, plot=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
        if info["success"]:
            eval_successes += 1
    eval_success_rate.append(eval_successes / 50)
    agent_model.rnd_weight = rnd_w

    np.save(os.path.join(logdir, "eval_success"), np.array(eval_success_rate))

