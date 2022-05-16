import os
import sys

import h5py
import numpy as np
import torch
import yaml

import wandb
from utils.datasets import SimplePointDataset
from models.svg import SVG
from utils.parser import create_env


def create_virtual_dataset(folder_path):
    layout_images, layout_actions, layout_lengths = None, None, None
    dataset_size = 0
    max_length = 0
    for i, file_name in enumerate(os.listdir(folder_path)):
        if "virtual" not in file_name:
            dataset_size += len(h5py.File(os.path.join(folder_path, file_name), 'r')["images"])
            max_length = max(max_length, h5py.File(os.path.join(folder_path, file_name), 'r')["images"].shape[1])
            print(file_name, max_length, len(h5py.File(os.path.join(folder_path, file_name), 'r')["images"]))
    print("Dataset Size", dataset_size)

    f = h5py.File(os.path.join(folder_path, "virtual_transition_data.h5"), 'w')
    curr_size, virtual_layout_created = 0, False
    for i, file_name in enumerate(os.listdir(folder_path)):
        if "virtual" not in file_name:
            path = os.path.join(folder_path, file_name)
            start = h5py.File(path, 'r')
            if not virtual_layout_created:
                layout_images = h5py.VirtualLayout(shape=(dataset_size, max_length,) + start["images"].shape[2:], dtype=np.float32)
                layout_actions = h5py.VirtualLayout(shape=(dataset_size, max_length,) + start["actions"].shape[2:],
                                                    dtype=np.float32)
                layout_lengths = h5py.VirtualLayout(shape=(dataset_size,), dtype=np.int32)
                virtual_layout_created = True

            vsource_images = h5py.VirtualSource(path, "images", start["images"].shape)
            layout_images[curr_size: curr_size + vsource_images.shape[0], :start["images"].shape[1], :, :, :] = vsource_images
            vsource_actions = h5py.VirtualSource(path, "actions", start["actions"].shape)
            layout_actions[curr_size: curr_size + vsource_actions.shape[0], :start["actions"].shape[1], :] = vsource_actions
            vsource_lengths = h5py.VirtualSource(path, "lengths", start["lengths"].shape)
            layout_lengths[curr_size: curr_size + vsource_lengths.shape[0]] = vsource_lengths

            curr_size += vsource_images.shape[0]
    f.create_virtual_dataset("images", layout_images, fillvalue=255)
    f.create_virtual_dataset("actions", layout_actions, fillvalue=255)
    f.create_virtual_dataset("lengths", layout_lengths)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)
    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    env = create_env(params["env"])

    model = SVG(env.observation_space, env.action_space.shape[0], params["video_model"])
    video_dir_name = f"./saved/{params['env']['type']}/{model.type}/"
    model.load(video_dir_name + "svg.pt")  # TODO  remove

    create_virtual_dataset("saved/{}/data/".format(params["env"]["type"]))
    dataset = SimplePointDataset("saved/{}/data/virtual_transition_data.h5".format(params["env"]["type"]), params["video_model"]["n_past"] + params["video_model"]["n_future"])

    # Model directory
    dir_name = f"./saved/{params['env']['type']}/{model.type}/"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # Train model
    model.train(params["video_model"]["epochs"], dataset)
    model.save(dir_name + "svg.pt")
