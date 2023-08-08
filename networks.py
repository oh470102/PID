import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class DQNModel(nn.Module):

    '''
    Standard neural network architecture.
        - 2 hidden layers (4 total), all fully-connected (FC).
        - composition: (in, 64, 150, 100, out).
        - Leaky ReLU was used throughout as acitvation.
    '''

    def __init__(self, input_dim: int, output_dim: int) -> None:

        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 150)
        self.layer3 = nn.Linear(150, 100)
        self.layer4 = nn.Linear(100, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = self.layer4(x)

        return x

class DuelDQNModel(nn.Module):

    '''
    Neural network with dueling architecture (Duel DQN).
        - composed of:
            - common layer (in, 64, 150)
            - advantage layer (150, 100, out), value layer (150, 100, 1) (NOTE: they are two separate streamlines)
            - final Q-layer (the separate streamlines are combined for final output)
        - Leaky ReLU was used throughout as acitvation.
    '''
    def __init__(self, input_dim: int, output_dim: int) -> None:

        super().__init__()

        # common layer(s)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 150),
            nn.LeakyReLU(),
        )

        # advantage layer(s)
        self.adv_layer = nn.Sequential(
            nn.Linear(150, 100),
            nn.LeakyReLU(),
            nn.Linear(100, output_dim),
        )

        # value layer
        self.val_layer = nn.Sequential(
            nn.Linear(150, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)

        a = self.adv_layer(x)
        v = self.val_layer(x)

        '''
        Duel DQN
            - Estimates Q as: Q(s,a) = V(s) + A(s, a) - mean(A(s,a))
            - Introduces bias by using mean() but computationally more efficient
        '''

        q = v + a - a.mean(dim=-1, keepdim=True)

        return q
    
class NoisyDQNModel(nn.Module):

    '''
    The Noisy Network architecture proposed by DeepMind (2017).
        - introduces noise to network parameters (W, b) to encourage exploration
        - an alternative to e-greedy
        - the dueling architecture is also incorporated here.

        - composed of:
            - common noisy layers (in, 64, 150)
            - advantage layer with noise (150, 100, out), value layer with noise (150, 100, 1) (NOTE: they are two separate streamlines)
            - final Q-layer (the separate streamlines are combined for final output)

        - Leaky ReLU was used throughout as acitvation.

    NOTE: if NoisyDQNModel is used, do not use e-greedy.
    '''

    def __init__(self, input_dim: int, output_dim: int) -> None:

        super().__init__()

        # noisy linear layer(s)
        self.noisy_layer1 = NoisyLinear(64, 150)
        self.noisy_layer2 = NoisyLinear(150, 100)
        self.noisy_layer3 = NoisyLinear(150, 100)

        # common layer(s)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            self.noisy_layer1,
            nn.LeakyReLU(),
        )

        # advantage layer(s)
        self.adv_layer = nn.Sequential(
            self.noisy_layer2,
            nn.LeakyReLU(),
            nn.Linear(100, output_dim),
        )

        # value layer
        self.val_layer = nn.Sequential(
            self.noisy_layer3,
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer(x)

        a = self.adv_layer(x)
        v = self.val_layer(x)

        q = v + a - a.mean(dim=-1, keepdim=True)

        return q
    
    def reset_noise(self) -> None:

        '''
        noise for each layer must be reset after each time-step.
        '''

        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()
        self.noisy_layer3.reset_noise()

        return 

class NoisyLinear(nn.Module):
    
    '''
    Refer to https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb for an explanation of what's going on...
    Refer to https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb for original code.
    '''
    
    def __init__(self, input_dim: int, output_dim: int, std_init = 0.5) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.Tensor(output_dim, input_dim))

        self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim))
        self.register_buffer('bias_epsilon', torch.Tensor(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.linear(
            x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.input_dim)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.input_dim))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.output_dim))

    def reset_noise(self) -> None:
        epsilon_in = self.scale_noise(self.input_dim)
        epsilon_out = self.scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())