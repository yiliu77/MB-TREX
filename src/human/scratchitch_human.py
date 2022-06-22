from .human import Human


class ScratchItchHuman(Human):

    def __init__(self, env, same_margin):
        self.env = env
        self.same_margin = same_margin

    def query_preference(self, traj1, traj2):
        states1 = traj1[:, :-self.env.action_space.shape[0]]
        states2 = traj2[:, :-self.env.action_space.shape[0]]
        actions1 = traj1[:, -self.env.action_space.shape[0]:]
        actions2 = traj2[:, -self.env.action_space.shape[0]:]
        cost1 = self.env.get_expert_cost(states1, actions1).sum()
        cost2 = self.env.get_expert_cost(states2, actions2).sum()
        if abs(cost1 - cost2) > self.same_margin:
            label = int(cost1 > cost2)
        else:
            label = 0.5
        return label