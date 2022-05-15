from envs.maze import Maze


def create_env(params):
    if params["type"] == "maze":
        return Maze(params["horizon"], params["use_images"], params["dense_rewards"], params["max_force"], params["goal_thresh"], params["walls"])
    else:
        raise NotImplementedError
