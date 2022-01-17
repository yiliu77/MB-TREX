import pygame
import numpy as np

class Human:
    def get_demo(self, n):
        return NotImplementedError

    def query_preference(self, traj1, traj2):
        raise NotImplementedError


class MazeHuman(Human):
    def __init__(self, env):
        self.env = env

    def get_demo(self, n):
        pygame.init()
        screen_width, screen_height = 400, 400
        screen = pygame.display.set_mode((screen_width, screen_height))
        all_traj_obs = []
        all_traj_actions = []

        line_start = np.array([50, 300])
        for _ in range(n):
            done = False
            all_obs, all_actions = [], []
            obs = self.env.reset()
            while not done:
                screen.fill((255, 255, 255))
                obs_image = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                obs_image = pygame.transform.scale2x(obs_image)
                obs_image = pygame.transform.scale2x(obs_image)
                screen.blit(obs_image, (0, 0))

                mouse_pos = np.array(pygame.mouse.get_pos())
                action_delta = np.clip((mouse_pos - line_start) / 100, self.env.action_space.low, self.env.action_space.high)
                pygame.draw.line(screen, (255, 0, 0), tuple(line_start), tuple(line_start + action_delta * 100), 7)

                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONUP:
                        next_obs, reward, done, _ = self.env.step(action_delta)
                        all_obs.append(obs)
                        all_actions.append(action_delta)
                        obs = next_obs
                pygame.display.update()

            all_traj_obs.append(all_obs)
            all_traj_actions.append(all_actions)

        pygame.quit()
        return all_traj_obs, all_traj_actions

    def query_preference(self, paired_states1, paired_states2):
        # Trajectories are L x C x H x W
        labels = []
        curr_image = 0

        pygame.init()
        screen_width, screen_height = 400, 300
        screen = pygame.display.set_mode((screen_width, screen_height))
        done = False
        t = 0
        while not done:
            screen.fill((255, 255, 255))

            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[0] < screen_width / 2 - 50:
                pygame.draw.rect(screen, (135, 206, 235), (0, 0, screen_width / 2 - 50, screen_height))
            elif mouse_pos[0] > screen_width / 2 + 50:
                pygame.draw.rect(screen, (10, 0, 205), (screen_width / 2 + 50, 0, screen_width / 2 - 50, screen_height))
            else:
                pygame.draw.rect(screen, (123, 104, 238), (screen_width / 2 - 50, 0, 100, screen_height))

            _, w, h, _ = paired_states1[curr_image].shape
            left_image = pygame.surfarray.make_surface(np.transpose(paired_states1[curr_image][t], (1, 0, 2)) * 255)
            right_image = pygame.surfarray.make_surface(np.transpose(paired_states2[curr_image][t], (1, 0, 2)) * 255)
            screen.blit(left_image, (50, (screen_height - h) / 2))
            screen.blit(right_image, (screen_width - 50 - w, (screen_height - h) / 2))
            pygame.display.update()

            t = (t + 1) % len(paired_states1[curr_image])
            pygame.time.wait(200)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    if pos[0] < screen_width / 2 - 50:
                        labels.append(0)
                    elif pos[0] > screen_width / 2 + 50:
                        labels.append(1)
                    else:
                        labels.append(0.5)
                    curr_image += 1
                    if len(paired_states1) == curr_image:
                        done = True
        pygame.quit()
        return labels

