import sys

import torch
import yaml
import wandb
import numpy as np

from models.svg import SVG
from models.trex import TREX
from models.visualmpc import VisualMPC
import pygame
from utils.parser import create_env, create_human

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(sys.argv[1], 'r') as stream:
        params = yaml.safe_load(stream)

    wandb.init(project="MB-TREX", entity="yiliu77", config=params)

    env = create_env(params["env"])
    human = create_human(params["human"], env)

    video_model = SVG(env.observation_space, env.action_space.shape[0], params["video_model"])
    video_dir_name = f"./saved/{params['env']['type']}/{video_model.type}/"
    video_model.load(video_dir_name + "svg.pt")

    pygame.init()
    screen_width, screen_height = 400, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    line_start = np.array([50, 300])

    while True:
        state = env.reset()
        start_state = torch.as_tensor(state / 255).to("cuda").float()
        start_state = start_state.permute((2, 0, 1)).unsqueeze(0)
        start_state = start_state.repeat(video_model.batch_size, 1, 1, 1)

        actions = []
        done = False
        while not done:
            screen.fill((255, 255, 255))
            obs_image = pygame.surfarray.make_surface(np.transpose(state, (1, 0, 2)))
            obs_image = pygame.transform.scale2x(obs_image)
            obs_image = pygame.transform.scale2x(obs_image)
            screen.blit(obs_image, (0, 0))

            mouse_pos = np.array(pygame.mouse.get_pos())
            action_delta = np.clip((mouse_pos - line_start) / 100, env.action_space.low,
                                   env.action_space.high)
            pygame.draw.line(screen, (255, 0, 0), tuple(line_start), tuple(line_start + action_delta * 100), 7)

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    torch.manual_seed(123)
                    actions.append(action_delta)

                    action_tensors = torch.as_tensor(np.array(actions)).to("cuda").float().unsqueeze(1).repeat(1, video_model.batch_size, 1)
                    obs = video_model.predict_states(start_state, action_tensors, len(actions) + 1)
                    obs_np = obs.detach().cpu().numpy()[:, 0, ...]
                    state = obs[-1, 0, ...].permute((1, 2, 0)) * 255
                    state = state.cpu().numpy()
            pygame.display.update()
        pygame.quit()
