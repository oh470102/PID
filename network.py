'''
Define Actor & Critic Networks
Consider employing two actor networks in future work, maybe...
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class Actor(nn.Module):

    def __init__(self, action_dim, action_bound):
        # action_dim = 3 if pidcontrol1 else 6 if pidcontrol2
        super().__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-6, 1.0]

        # state_dim = 4
        self.l1 = nn.Linear(4, 64) 
        self.ln_1 = nn.LayerNorm(64)
        self.l2 = nn.Linear(64, 64)
        self.ln_2 = nn.LayerNorm(64)
        self.l3 = nn.Linear(64, 16)
        self.ln_3 = nn.LayerNorm(16)

        self.mu = nn.Linear(16, self.action_dim)
        self.std = nn.Linear(16, self.action_dim)

        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
        x = F.relu(self.ln_1(self.l1(x)))
        x = F.relu(self.ln_2(self.l2(x)))
        x = F.relu(self.ln_3(self.l3(x)))

        mu = torch.tanh(self.mu(x))
        std = F.softplus(self.std(x))

        mu = (lambda x: x * self.action_bound)(mu)
        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])

        return mu, std

    def sample_normal(self, mu, std, reparam):
        normal_dist = D.Normal(mu, std)
        if reparam:
            action = normal_dist.rsample()
        else:
            action = normal_dist.sample()
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        log_pdf = normal_dist.log_prob(action)
        log_pdf = torch.sum(log_pdf, dim=-1, keepdim=True)

        return action, log_pdf

class Critic(nn.Module):
    
    def __init__(self, action_dim, state_dim):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.v1 = nn.Linear(self.state_dim, 32)
        self.a1 = nn.Linear(self.action_dim, 32)

        self.l2 = nn.Linear(64, 32) # NOTE: 64 = v1.size + a1.size
        self.ln_2 = nn.LayerNorm(32)
        self.l3 = nn.Linear(32, 32)
        self.ln_3 = nn.LayerNorm(32)
        self.q =  nn.Linear(32, 1)

        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state_action):
        state, action = state_action[0], state_action[1]

        v = F.relu(self.v1(state))
        v = F.normalize(v, p=2, dim=-1)
        a = F.relu(self.a1(action))
        a = F.normalize(a, p=2, dim=-1)

        x = torch.cat([v, a], dim=-1)
        x = F.relu(self.ln_2(self.l2(x)))
        x = F.relu(self.ln_3(self.l3(x)))
        x = self.q(x)

        return x