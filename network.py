import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class Actor(nn.Module):

    def __init__(self, action_dim, action_bound, state_dim):
        super().__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-6, 1.0]
        self.state_dim = state_dim

        self.l1 = nn.Linear(self.state_dim, 128) 
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 16)

        self.mu = nn.Linear(16, self.action_dim)
        self.std = nn.Linear(16, self.action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

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

        self.v1 = nn.Linear(self.state_dim, 128)
        self.a1 = nn.Linear(self.action_dim, 128)

        self.l2 = nn.Linear(256, 128) # NOTE: 256 = v1.size + a1.size
        self.l3 = nn.Linear(128, 64)
        self.q =  nn.Linear(64, 1)

    def forward(self, state_action):
        state, action = state_action[0], state_action[1]

        v = F.relu(self.v1(state))
        a = F.relu(self.a1(action))
        x = torch.cat([v, a], dim=-1)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.q(x)

        return x