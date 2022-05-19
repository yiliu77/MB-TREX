from envs.simple_point import SimplePoint
from human.simple_point_human import SimplePointHuman
from envs.maze import Maze
from human.maze_human import MazeHuman
from envs.hopper import Hopper
from human.hopper_human import HopperHuman


def create_env(params):
    if params["type"] == "simple_point":
        return SimplePoint(params["horizon"], params["max_force"])
    elif params["type"] == "maze":
        return Maze(params["horizon"], params["start"], params["max_force"])
    elif params["type"] == "hopper":
        return Hopper(params)
    else:
        raise NotImplementedError


def create_human(params, env):
    if params["type"] == "simple_point_human":
        return SimplePointHuman(env)
    elif params["type"] == "maze_human":
        return MazeHuman(env)
    elif params["type"] == "hopper_human":
        return HopperHuman(env)
    else:
        raise NotImplementedError
