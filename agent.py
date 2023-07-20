from copy import *
from tqdm import *
from network import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

class SACAgent:

    def __init__(self, env, load=False):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 48
        self.BUFFER_SIZE = 100_000
        self.ACTOR_LEARNING_RATE = 9e-4
        self.CRITIC_LEARNING_RATE = 9e-4
        self.TAU = 5e-3
        self.ALPHA = 0.25      # sensitive

        self.P_list = [50]
        self.D_list = [10]
        self.I_list = [10]
 
        self.env = env
        self.state_dim = 3+4+4 # PID, Sp, Sp-CV
        self.action_dim = 3
        self.action_bound = 0.1 # sensitive

        self.actor = Actor(self.action_dim, self.action_bound)
        
        self.critic_1 = Critic(self.action_dim, self.state_dim)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(self.action_dim, self.state_dim)
        self.target_critic_2 = deepcopy(self.critic_2)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=self.CRITIC_LEARNING_RATE)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=self.CRITIC_LEARNING_RATE)

        if load is True:
            t = input("Latest Model Time (HH:MM): ")
            PATH = f"./saved_models/{t}/"
            self.actor.load_state_dict(torch.load(PATH + 'actor.pth'))
            self.critic_1.load_state_dict(torch.load(PATH + "critic_1.pth"))
            self.critic_2.load_state_dict(torch.load(PATH + "critic_2.pth"))
            self.target_critic_1.load_state_dict(torch.load(PATH + "target_critic_1.pth"))
            self.target_critic_2.load_state_dict(torch.load(PATH + "target_critic_2.pth"))

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)
        self.save_epi_reward = []

    def train(self, max_episode_num, save=False):
        '''
        Let s = [P, I, D, SP, CV]
        reset returns (CV), {}
        step returns (CV), return, {}

        a = [dP, dI, dD]
        '''
        for ep in tqdm(range(max_episode_num)):

            if ep % 10 == 0: 
                live_plot(self.save_epi_reward)

            init_state, _ = self.env.reset() 
            curr_PID = np.array([x[-1] for x in [self.P_list, self.I_list, self.D_list]])
            SP = np.array([0,0,0,0]) # desired state
            init_state = np.concatenate((curr_PID, SP, init_state)) 

            action = self.get_action(torch.from_numpy(init_state).to(torch.float32))
            action = np.clip(action, -self.action_bound, self.action_bound)

            P, I, D = curr_PID + np.array(action)
            self.P_list.append(P); self.I_list.append(I); self.D_list.append(D)

            _, reward, _ = self.env.linstep([P, I, D])

            self.buffer.add_buffer(init_state, [P, I, D], reward, True)

            if self.buffer.count > 100:
                init_states, actions, rewards, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                target_qi = rewards
                
                y_i = self.q_target(rewards, target_qi, dones)

                self.critic_learn(torch.tensor(init_states).to(torch.float32).to(self.actor.device), torch.tensor(actions).to(torch.float32).to(self.actor.device),
                                    torch.tensor(y_i).to(torch.float32).to(self.actor.device))
                self.actor_learn(torch.tensor(init_states).to(torch.float32).to(self.actor.device))
                
                self.update_target_network()

            self.save_epi_reward.append(reward)
            
        if save == True:
            self.save_agent()

        return self.save_epi_reward
    
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

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            mu, std = self.actor(state.to(self.actor.device))
            if deterministic:
                action = mu
            else:
                action, _ = self.actor.sample_normal(mu, std, reparam=False)
        return action.cpu().numpy()#[0]

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