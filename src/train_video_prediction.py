import os
import sys

import h5py
import numpy as np
import torch
import tqdm
import yaml

import wandb
from data.maze_dataset import MazeDataset
from models.svg import SVG


def create_dirs(path):
    if os.path.isdir(path):
        return
    os.makedirs(path)


def create_virtual_dataset(folder_path):
    layout_images, layout_actions = None, None
    dataset_size = 0
    for i, file_name in enumerate(os.listdir(folder_path)):
        if "virtual" not in file_name:
            dataset_size += len(h5py.File(os.path.join(folder_path, file_name), 'r')["images"])
    print("Dataset Size", dataset_size)

    f = h5py.File(os.path.join(folder_path, "virtual_transition_data.h5"), 'w')
    curr_size = 0
    for i, file_name in enumerate(os.listdir(folder_path)):
        if "virtual" not in file_name:
            path = os.path.join(folder_path, file_name)
            start = h5py.File(path, 'r')
            if i == 0:
                layout_images = h5py.VirtualLayout(shape=(dataset_size,) + start["images"].shape[1:],
                                                   dtype=np.float32)
                layout_actions = h5py.VirtualLayout(shape=(dataset_size,) + start["actions"].shape[1:],
                                                    dtype=np.float32)

            vsource_images = h5py.VirtualSource(path, "images", start["images"].shape)
            layout_images[curr_size: curr_size + vsource_images.shape[0], :, :, :, :] = vsource_images
            vsource_actions = h5py.VirtualSource(path, "actions", start["actions"].shape)
            layout_actions[curr_size: curr_size + vsource_actions.shape[0], :, :] = vsource_actions

            curr_size += vsource_images.shape[0]
    f.create_virtual_dataset("images", layout_images, fillvalue=255)
    f.create_virtual_dataset("actions", layout_actions, fillvalue=255)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)
    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    model_params = params["video_model"]
    model = SVG(model_params)

    create_virtual_dataset("saved/maze/")
    dataset = MazeDataset("saved/maze/virtual_transition_data.h5", model_params["n_past"] + model_params["n_future"])

    # Model directory
    dir_name = f"./saved/models/{model_params['type']}/{model_params['note']}/"
    create_dirs(dir_name)

    # Train model
    model.train(model_params["epochs"], dataset)
    model.save(dir_name + "saved_svg")
