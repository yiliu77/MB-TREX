from envs.maze import Maze
from envs.hopper import Hopper
from envs.scratchitch import ScratchItchPR2EnvMB
from human.human import LowDimHuman
from human.hopper_human import HopperHuman
from human.scratchitch_human import ScratchItchHuman

def create_env(params):
    if params["type"] == "maze":
        return Maze(params["horizon"], params["use_images"], params["dense_rewards"], params["max_force"], params["goal_thresh"], params["walls"])
    elif params["type"] == "hopper":
        return Hopper(params)
    elif params["type"] == "scratchitch":
        import os; os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        return ScratchItchPR2EnvMB()
    else:
        raise NotImplementedError

def create_human(params, env):
    if params["type"] == "maze_human":
        return LowDimHuman(env, params["same_margin"])
    elif params["type"] == "hopper_human":
        return HopperHuman(env, params["same_margin"], params["actual_human"])
    elif params["type"] == "scratchitch_human":
        return ScratchItchHuman(env, params["same_margin"])
    else:
        raise NotImplementedError
