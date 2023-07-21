from network import *
from utils import *
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np

class SACAgent:

    def __init__(self, env):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 128
        self.BUFFER_SIZE = 1_000_000
        self.ACTOR_LEARNING_RATE = 3e-4
        self.CRITIC_LEARNING_RATE = 3e-4
        self.TAU = 5e-3
        self.ALPHA = 1/3

        self.env = env
        self.state_dim = 3
        self.action_dim = 3
        self.action_bound = 0.1

        self.PID = np.array([None, None, None])

        self.actor = Actor(self.action_dim, self.action_bound, self.state_dim)
        
        self.critic_1 = Critic(self.action_dim, self.state_dim)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(self.action_dim, self.state_dim)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.CRITIC_LEARNING_RATE)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []

    def PID_reset(self):
        self.PID[0] = np.random.randint(10, 100)
        self.PID[1] = np.random.randint(10, 100)
        self.PID[2] = np.random.randint(10, 100)

        # self.PID[0] = 50
        # self.PID[0] = 10        
        # self.PID[0] = 10

        return self.PID
    
    def save_agent(self):
        from datetime import datetime
        import os 

        time = datetime.now().time().strftime('%H:%M')
        PATH = f'./saved_models/{time}'
        os.makedirs(PATH, exist_ok=True)

        torch.save(self.actor.state_dict(), PATH + '/actor.pth')
        torch.save(self.critic_1.state_dict(), PATH + '/critic_1.pth')
        torch.save(self.critic_2.state_dict(), PATH + '/critic_2.pth')
        torch.save(self.target_critic_1.state_dict(), PATH + "/target_critic_1.pth")
        torch.save(self.target_critic_2.state_dict(), PATH + "/target_critic_2.pth")

        print("------Agent Saved------")

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            mu, std = self.actor(state)
            if deterministic:
                action = mu
            else:
                action, _ = self.actor.sample_normal(mu, std, reparam=False)
        return action.numpy()[0]
    
    def update_target_network(self):
        phi_1 = self.critic_1.state_dict()
        phi_2 = self.critic_2.state_dict()
        target_phi_1 = self.target_critic_1.state_dict()
        target_phi_2 = self.target_critic_2.state_dict()

        for key in phi_1.keys():
            target_phi_1[key] = self.TAU * phi_1[key] + (1 - self.TAU) * target_phi_1[key]
            target_phi_2[key] = self.TAU * phi_2[key] + (1 - self.TAU) * target_phi_2[key]

        self.target_critic_1.load_state_dict(target_phi_1)
        self.target_critic_2.load_state_dict(target_phi_2)

    def critic_learn(self, states, actions, q_targets):
        q_1 = self.critic_1([states, actions])
        loss_1 = torch.mean( (q_1 - q_targets) ** 2)

        self.critic_1_opt.zero_grad()
        loss_1.backward()
        self.critic_1_opt.step()

        q_2 = self.critic_2([states, actions])
        loss_2 = torch.mean( (q_2 - q_targets) ** 2)

        self.critic_2_opt.zero_grad()
        loss_2.backward()
        self.critic_2_opt.step()

    def actor_learn(self, states):
        mu, std = self.actor(states)
        actions, log_pdfs = self.actor.sample_normal(mu, std, reparam=True)
        log_pdfs = log_pdfs.squeeze(1)
        soft_q_1 = self.critic_1([states, actions])
        soft_q_2 = self.critic_2([states, actions])
        soft_q = torch.min(soft_q_1, soft_q_2)

        loss = torch.mean(self.ALPHA * log_pdfs - soft_q)

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

    def q_target(self, rewards, q_values, dones):
        y_k = torch.from_numpy(q_values).clone().detach().numpy()

        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]

        return torch.from_numpy(y_k)
    
    def train(self, max_episode_num, save=False):

        for ep in tqdm(range(max_episode_num)):
            episode_reward, done = 0, False, False
            state = self.PID_reset()

            if ep % 10 == 0: 
                live_plot(self.save_epi_reward)

            while not done:
                action = self.get_action(torch.from_numpy(state).to(torch.float32))
                action = np.clip(action, -self.action_bound, self.action_bound)
                action = np.array(action)
                self.PID += action
                
                next_state, reward, done = self.env.linstep(self.PID)

                self.buffer.add_buffer(state, action, reward, next_state, done)

                if self.buffer.count > 100:
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    with torch.no_grad():
                        next_mu, next_std = self.actor(torch.tensor(next_states).to(torch.float32))
                        next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std, reparam=True)

                        target_qs_1 = self.target_critic_1([torch.tensor(next_states).to(torch.float32), next_actions])
                        target_qs_2 = self.target_critic_2([torch.tensor(next_states).to(torch.float32), next_actions])
                        target_qs = torch.min(target_qs_1, target_qs_2)

                        target_qi = target_qs - self.ALPHA * next_log_pdf
                    
                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    self.critic_learn(torch.tensor(states).to(torch.float32), torch.tensor(actions).to(torch.float32), torch.tensor(y_i).to(torch.float32))
                    self.actor_learn(torch.tensor(states).to(torch.float32))
                    
                    self.update_target_network()

                state = next_state
                episode_reward += reward

            self.save_epi_reward.append(episode_reward)
            
        if save == True:
            self.save_agent()

        return self.save_epi_reward