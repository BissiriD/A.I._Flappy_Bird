import torch
import random
import numpy as np
from flappy_bird_env import FlappyBirdEnv
from model import Linear_QNet, Agent as ModelAgent
from helper import plot

# Hyperparameters (match those in model.py)
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 100000
UPDATE_TARGET_EVERY = 100
HIDDEN_SIZE = 256

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    num_birds = 1  # Using single bird as in model.py
    env = FlappyBirdEnv(num_birds)
    agent = ModelAgent(input_size=4, hidden_size=HIDDEN_SIZE, output_size=2)
    n_games = 2000

    for episode in range(n_games):
        state = env.reset()
        done = False
        score = 0
        if episode % 5 == 0:
            env.render()

        while not done:
            action = agent.get_action(state[0])  # state[0] because env.reset() returns a list
            next_state, reward, done, info = env.step([action])
            score += reward[0]  # reward is a list, we take the first element
            agent.memory.push(state[0], action, reward[0], next_state[0], done[0])

            state = next_state

            if len(agent.memory) > BATCH_SIZE:
                batch = agent.memory.sample(BATCH_SIZE)
                agent.train_step(batch)

            if env.check_quit():
                print("Quitting...")
                return

            should_quit = env.render()
            if should_quit:
                print("Quitting...")
                return

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / (episode + 1)
        plot_mean_scores.append(mean_score)

        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_network()

        if episode % 50 == 0:
            agent.model.save(f'model_episode_{episode}.pth')

        print(f'Game {episode}, Score: {score}, Average Score: {mean_score:.2f}, Epsilon: {agent.epsilon:.2f}')

        if score > record:
            record = score
            agent.model.save('record_model.pth')

        plot(plot_scores, plot_mean_scores)

    agent.model.save('final_model.pth')
    env.close()

if __name__ == '__main__':
    train()