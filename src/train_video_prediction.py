import os
import sys

import torch
import tqdm
import wandb
import yaml

from data.maze_dataset import MazeDataset
from models.svg import SVG


def create_dirs(path):
    if os.path.isdir(path):
        return
    os.makedirs(path)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    model_params = params["video_model"]
    model = SVG(model_params)
    dataset = MazeDataset("saved/maze45/", model_params["n_past"] + model_params["n_future"])

    # Model directory
    dir_name = f"./saved/models/{model_params['type']}/{model_params['note']}/"
    create_dirs(dir_name)

    # Train model
    train_loss = []
    val_loss = []
    epochs = tqdm.trange(model_params["epochs"], desc='Loss', leave=True)
    for epoch in epochs:
        train_epoch_loss = model.train(dataset)
        train_loss.append(train_epoch_loss)
        epochs.set_description("Train Loss: {:.4f}".format(train_epoch_loss))
        epochs.refresh()

        model.save(dir_name + "saved_svg")