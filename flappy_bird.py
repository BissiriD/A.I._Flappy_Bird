import pygame
import sys
import random

class FlappyBirdGame:
    def __init__(self, num_birds=1):
        pygame.mixer.pre_init(frequency=44100, size=16, channels=1, buffer=512)
        pygame.init()
        self.screen = pygame.display.set_mode((576, 1024))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.Font('04B_19.ttf', 40)

        # Game Variables
        self.gravity = 0.25
        self.bird_movement = [0] * num_birds
        self.game_active = [True] * num_birds
        self.scores = [0] * num_birds
        self.high_score = 0

        self.bg_surface = pygame.image.load('assets/background-day.png').convert()
        self.bg_surface = pygame.transform.scale2x(self.bg_surface)

        self.floor_surface = pygame.image.load('assets/base.png').convert()
        self.floor_surface = pygame.transform.scale2x(self.floor_surface)
        self.floor_x_pos = 0

        bird_downflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-downflap.png').convert_alpha())
        bird_midflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-midflap.png').convert_alpha())
        bird_upflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-upflap.png').convert_alpha())
        self.bird_frames = [bird_downflap, bird_midflap, bird_upflap]
        self.bird_index = 0
        self.bird_surfaces = [self.bird_frames[self.bird_index] for _ in range(num_birds)]
        self.bird_rects = [self.bird_surfaces[0].get_rect(center=(100, 512)) for _ in range(num_birds)]

        self.BIRDFLAP = pygame.USEREVENT + 1
        pygame.time.set_timer(self.BIRDFLAP, 200)

        self.pipe_surface = pygame.image.load('assets/pipe-green.png')
        self.pipe_surface = pygame.transform.scale2x(self.pipe_surface)
        self.pipe_list = []
        self.SPAWNPIPE = pygame.USEREVENT
        pygame.time.set_timer(self.SPAWNPIPE, 1200)
        self.pipe_height = [400, 600, 800]

        self.scored_pipes = [[] for _ in range(num_birds)]

        self.game_over_surface = pygame.transform.scale2x(pygame.image.load('assets/message.png').convert_alpha())
        self.game_over_rect = self.game_over_surface.get_rect(center=(288, 512))

        self.flap_sound = pygame.mixer.Sound('sound/sfx_wing.wav')
        self.death_sound = pygame.mixer.Sound('sound/sfx_hit.wav')
        self.score_sound = pygame.mixer.Sound('sound/sfx_point.wav')

    def draw_floor(self):
        self.screen.blit(self.floor_surface, (self.floor_x_pos, 900))
        self.screen.blit(self.floor_surface, (self.floor_x_pos + 576, 900))

    def create_pipe(self):
        random_pipe_pos = random.choice(self.pipe_height)
        bottom_pipe = self.pipe_surface.get_rect(midtop=(700, random_pipe_pos))
        top_pipe = self.pipe_surface.get_rect(midbottom=(700, random_pipe_pos - 300))
        bottom_pipe.inflate_ip(-20, -20)
        top_pipe.inflate_ip(-20, -20)
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        for pipe in pipes:
            pipe.centerx -= 5
        return pipes

    def draw_pipes(self, pipes):
        for pipe in pipes:
            if pipe.bottom >= 1024:
                self.screen.blit(self.pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
                self.screen.blit(flip_pipe, pipe)

    def remove_pipes(self, pipes):
        return [pipe for pipe in pipes if pipe.centerx > -50]

    def check_collision(self, pipes, bird_index):
        for pipe in pipes:
            if self.bird_rects[bird_index].colliderect(pipe):
                self.death_sound.play()
                return False
        if self.bird_rects[bird_index].top <= -100 or self.bird_rects[bird_index].bottom >= 900:
            return False
        return True

    def rotate_bird(self, bird, bird_index):
        return pygame.transform.rotozoom(bird, -self.bird_movement[bird_index] * 3, 1)

    def bird_animation(self):
        new_bird = self.bird_frames[self.bird_index]
        new_bird_rects = [new_bird.get_rect(center=(100, rect.centery)) for rect in self.bird_rects]
        return new_bird, new_bird_rects

    def score_display(self, game_state):
        if game_state == 'main_game':
            score_surface = self.game_font.render(str(max(self.scores)), True, (255, 255, 255))
            score_rect = score_surface.get_rect(center=(288, 100))
            self.screen.blit(score_surface, score_rect)
        if game_state == 'game_over':
            score_surface = self.game_font.render(f'Score: {max(self.scores)}', True, (255, 255, 255))
            score_rect = score_surface.get_rect(center=(288, 100))
            self.screen.blit(score_surface, score_rect)

            high_score_surface = self.game_font.render(f'High score: {int(self.high_score)}', True, (255, 255, 255))
            high_score_rect = high_score_surface.get_rect(center=(288, 850))
            self.screen.blit(high_score_surface, high_score_rect)

    def update_score(self):
        self.high_score = max(max(self.scores), self.high_score)

    def reset_all_birds(self):
        self.bird_movement = [0] * len(self.bird_movement)
        self.game_active = [True] * len(self.game_active)
        self.scores = [0] * len(self.scores)
        self.bird_rects = [self.bird_surfaces[0].get_rect(center=(100, 512)) for _ in range(len(self.bird_rects))]
        self.scored_pipes = [[] for _ in range(len(self.scored_pipes))]
        self.pipe_list.clear()

if __name__ == "__main__":
    game = FlappyBirdGame()