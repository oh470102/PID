import torch
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def showcase(agent, env, n_showcase=3):

    n_episodes = n_showcase
    scores = []
    for _ in range(n_episodes):
        state, _ = env.reset()

        score = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            state = torch.tensor(state, dtype=torch.float32)
            action = agent.get_action(state, deterministic=True)
            action = np.clip(action, -agent.action_bound, agent.action_bound)
            action = list(action)
            print(action)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
        
        scores.append(score)

    print(f"Showcase average score: {sum(scores)/len(scores):10.2f}")
    env.close() 

def analyze(scores):
    scores = np.array(scores)
    print(f"Mean: {np.mean(scores):10.2f}")
    print(f"Median: {np.median(scores):10.2f}")
    print(f"Best: {np.max(scores):10.2f}")
    print(f"Worst: {np.min(scores):10.2f}")
    print(f"Std: {np.std(scores):10.2f}")

def final_plot(g1):
    import numpy as np
    resolve_matplotlib_error()
    plt.ioff()

    window_size = 10
    moving_average = np.convolve(g1, np.ones(window_size) / window_size, mode='valid')

    plt.plot(moving_average, label='mAverage',color='red', linewidth=2.5)
    plt.legend()
    plt.draw()
    plt.show()

def live_plot(scores):
    
    plt.plot(scores)
    plt.grid(True)
    plt.draw()
    plt.gca().grid(True)
    plt.xlabel('episodes')
    plt.ylabel('scores')
    plt.pause(0.001)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: 
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample_batch(self, batch_size):

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])

        return states, actions, rewards, next_states, dones

    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0