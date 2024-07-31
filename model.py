import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from collections import deque
import numpy as np
from flappy_bird_env import FlappyBirdEnv

# Hyperparameters
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 100000
UPDATE_TARGET_EVERY = 100
HIDDEN_SIZE = 256

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model = Linear_QNet(input_size, hidden_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)  # Assuming 2 possible actions
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones.float()) * GAMMA * next_q_values

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

def train():
    env = FlappyBirdEnv(num_birds=1)
    agent = Agent(input_size=4, hidden_size=HIDDEN_SIZE, output_size=2)
    scores = []
    n_games = 2000

    for episode in range(n_games):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step([action])  # Assuming env.step expects a list of actions
            score += reward
            agent.memory.push(state, action, reward, next_state[0], done)  # Assuming next_state is a list with one state

            state = next_state[0]

            if len(agent.memory) > BATCH_SIZE:
                batch = agent.memory.sample(BATCH_SIZE)
                agent.train_step(batch)

            if env.check_quit():
                print("Quitting...")
                return

        scores.append(score)
        mean_score = np.mean(scores[-100:])  # moving average of last 100 scores

        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_network()

        if episode % 50 == 0:
            agent.model.save(f'model_episode_{episode}.pth')

        print(f'Game {episode}, Score: {score}, Average Score: {mean_score:.2f}, Epsilon: {agent.epsilon:.2f}')

    agent.model.save('final_model.pth')

if __name__ == '__main__':
    train()