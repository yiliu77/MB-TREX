from envs.maze import Maze
from envs.hopper import Hopper

def create_env(params):
    if params["type"] == "maze":
        return Maze(params["horizon"], params["use_images"], params["dense_rewards"], params["max_force"], params["goal_thresh"], params["walls"])
    elif params["type"] == "hopper":
        return Hopper(params)
    else:
        raise NotImplementedError
