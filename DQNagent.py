from anothernetwork import *
from utils import *
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np

class DQNAgent:

    def __init__(self, env):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 200
        self.BUFFER_SIZE = 1_000_000
        self.LEARNING_RATE = 3e-4
        self.synch_freq = 500
        self.synch_i = 0
        self.epsilon = 1

        self.env = env
        self.state_dim = 3
        self.action_dim = 3
        self.action_bound = 0.5

        self.PID = np.array([None, None, None])
        
        self.model = DQN(self.action_dim, self.state_dim)
        self.target_model = deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)
        self.save_epi_reward = []

    def get_action(self, state):
        
        q = self.model(state)

        if np.random.random() < self.epsilon:
            action = np.random.randint(-1, 2, size=3)
        else:
            action = np.argmax(q.detach().numpy(), axis=1)

        print(action)

        return action
    
    def update_target_network(self):
        if self.synch_i % self.synch_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
    def learn(self, states, q_targets):
        q = self.model(states)
        loss = torch.mean( (q - q_targets) ** 2)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


    def q_target(self, rewards, q_values, dones):
        y_k = torch.from_numpy(q_values).clone().detach().numpy()

        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]

        return torch.from_numpy(y_k)
    
    def train(self, max_episode_num):

        for ep in tqdm(range(max_episode_num)):
            done, time = False, 0 
            state = self.env.reset()
            self.PID = state

            print(f"Initial PID: {self.PID}")

            if ep % 2 == 0: 
                live_plot(self.save_epi_reward)

            while not done:
                action = self.get_action(torch.from_numpy(state).to(torch.float32))
                action = np.array(action)
                self.PID += action
                
                next_state, reward, mean_score = self.env.linstep(action)
                next_state = np.clip(next_state, 0, 200)
                self.save_epi_reward.append(mean_score)

                print(f"Changed PID to: {next_state}, recording score: {mean_score:2f}, and got better by {reward}")
                time += 1; done = True if time >= 500 else False

                self.buffer.add_buffer(state=state, action=action, reward=reward, next_state=next_state, done=done)

                if self.buffer.count > 100:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    with torch.no_grad():

                        target_q = self.target_model(torch.tensor(next_states).to(torch.float32))
                    
                    y_i = self.q_target(rewards, target_q.numpy(), dones)
                    self.learn(torch.tensor(states).to(torch.float32), torch.tensor(y_i).to(torch.float32))

                    self.update_target_network()

                state = next_state
                if self.epsilon > 0.1: self.epsilon -= (1/(10*max_episode_num))

        return self.save_epi_reward