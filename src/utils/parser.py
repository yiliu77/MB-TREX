from envs.simple_point import SimplePoint
from human.simple_point_human import SimplePointHuman
from envs.maze import Maze
from human.maze_human import MazeHuman
from envs.hopper import Hopper


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
    if params["type"] == "maze_human":
        return MazeHuman(env)
    else:
        raise NotImplementedError
