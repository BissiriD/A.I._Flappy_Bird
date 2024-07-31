import gym
from gym import spaces
import numpy as np
import pygame
import random
from flappy_bird import FlappyBirdGame

class FlappyBirdEnv(gym.Env):
    def __init__(self, num_birds=5):
        super(FlappyBirdEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(4,), dtype=np.float32)
        self.game = FlappyBirdGame(num_birds)
        self.num_birds = 1
        self.clock = pygame.time.Clock()

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return True
        return False

    def step(self, actions):
        if self.check_quit():
            return None, None, [True] * self.num_birds, {"scores": self.game.scores, "quit": True}

        rewards = []
        dones = []
        states = []

        for i, action in enumerate(actions):
            if action == 1:  # Flap
                self.game.bird_movement[i] = 0
                self.game.bird_movement[i] -= 12
                self.game.flap_sound.play()

            # Update game state for each bird
            self.game.bird_movement[i] += self.game.gravity
            self.game.bird_rects[i].centery += self.game.bird_movement[i]

        # Update pipes (only need to do this once)
        self.game.pipe_list = self.game.move_pipes(self.game.pipe_list)
        self.game.pipe_list = self.game.remove_pipes(self.game.pipe_list)

        # Check collisions and calculate rewards for each bird
        for i in range(self.num_birds):
            self.game.game_active[i] = self.game.check_collision(self.game.pipe_list, i)
            reward = self._get_reward(i)
            rewards.append(reward)
            dones.append(not self.game.game_active[i])
            states.append(self._get_state(i))

        return states, rewards, dones, {"scores": self.game.scores, "quit": False}

    def reset(self):
        self.game.reset_all_birds()
        return [self._get_state(i) for i in range(self.num_birds)]

    def _get_state(self, bird_index):
        bird_y = self.game.bird_rects[bird_index].centery
        bird_velocity = self.game.bird_movement[bird_index]
        pipe_x = self.game.pipe_list[0].centerx if self.game.pipe_list else 700
        pipe_y = self.game.pipe_list[0].centery if self.game.pipe_list else random.choice(self.game.pipe_height)
        return np.array([bird_y, bird_velocity, pipe_x, pipe_y], dtype=np.float32)

    def render(self, mode='human'):
        if self.check_quit():
            return True  # Indicate that we should quit

        self.game.screen.blit(self.game.bg_surface, (0, 0))
        self.game.draw_pipes(self.game.pipe_list)
        for i in range(self.num_birds):
            if self.game.game_active[i]:
                rotated_bird = self.game.rotate_bird(self.game.bird_surfaces[i], i)
                self.game.screen.blit(rotated_bird, self.game.bird_rects[i])
        self.game.draw_floor()
        self.game.score_display('main_game')
        self.clock.tick(30)
        pygame.display.update()
        return False  # Indicate that we should continue

    def _get_reward(self, bird_index):
        if not self.game.game_active[bird_index]:
            return -1  # Penalize for collision
        reward = 0.1  # Small reward for staying alive
        for pipe in self.game.pipe_list:
            if pipe.centerx == 95 and pipe not in self.game.scored_pipes[bird_index]:
                self.game.scores[bird_index] += 0.5
                self.game.scored_pipes[bird_index].append(pipe)
                reward += 1  # Reward for passing a pipe
                self.game.score_sound.play()
        return reward

    def get_states(self):
        return [self._get_state(i) for i in range(self.num_birds)]

    def close(self):
        pygame.quit()